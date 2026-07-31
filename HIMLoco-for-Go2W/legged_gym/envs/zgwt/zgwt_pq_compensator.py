"""GPU-parallel simplified neural P/Q composite adaptive compensation.

This module deliberately has no Isaac Gym dependency so its tensor dynamics can
be unit-tested on CPU.  The fixed feature network is shared by all environments;
P, Q, W_hat and all filter/reference states are per-environment runtime state.

The elementwise weight clamps below are engineering safety guards.  They are not
the spectral-norm projection operator from the full paper and do not provide the
paper's complete projection-based stability guarantees.
"""

import math
import os
from typing import Dict, Optional

import torch
from torch import nn


def _cfg_value(cfg, name):
    if not hasattr(cfg, name):
        raise ValueError(f"missing PQ configuration field: {name}")
    return getattr(cfg, name)


class FixedPQFeatureNetwork(nn.Module):
    """Deterministically initialized MLP whose parameters never update."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        feature_dim: int,
        activation: str = "tanh",
        seed: int = 0,
    ):
        super().__init__()
        if input_dim <= 0 or feature_dim <= 0:
            raise ValueError("input_dim and feature_dim must be positive")
        hidden_dims = [int(width) for width in hidden_dims]
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("all hidden_dims must be positive")
        if activation.lower() != "tanh":
            raise ValueError("the first PQ implementation supports activation='tanh'")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        dims = [int(input_dim)] + hidden_dims + [int(feature_dim)]
        layers = []
        # nn.Linear initializes itself from the global RNG before we overwrite
        # its parameters. Preserve that RNG so enabling PQ cannot perturb PPO or
        # environment randomization streams.
        global_rng_state = torch.random.get_rng_state()
        try:
            for in_features, out_features in zip(dims[:-1], dims[1:]):
                layer = nn.Linear(in_features, out_features)
                weight_bound = math.sqrt(6.0 / (in_features + out_features))
                bias_bound = 1.0 / math.sqrt(in_features)
                with torch.no_grad():
                    layer.weight.copy_(
                        torch.empty(
                            out_features, in_features, dtype=torch.float32
                        ).uniform_(
                            -weight_bound, weight_bound, generator=generator
                        )
                    )
                    layer.bias.copy_(
                        torch.empty(out_features, dtype=torch.float32).uniform_(
                            -bias_bound, bias_bound, generator=generator
                        )
                    )
                layers.extend((layer, nn.Tanh()))
        finally:
            torch.random.set_rng_state(global_rng_state)
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.activation_name = activation.lower()
        self.feature_seed = int(seed)
        self.requires_grad_(False)
        self.eval()

    def forward(self, inputs):
        return self.network(inputs)


class ZGWTPQCompensator:
    """Simplified P/Q composite learner for twelve ZGWT leg joints."""

    LEG_DOF_COUNT = 12

    def __init__(
        self,
        num_envs: int,
        cfg,
        device,
        dtype: torch.dtype = torch.float32,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        self.enabled = bool(_cfg_value(cfg, "enabled"))
        self.shadow_mode = bool(_cfg_value(cfg, "shadow_mode"))
        self.compensate_wheels = bool(_cfg_value(cfg, "compensate_wheels"))
        if self.compensate_wheels:
            raise ValueError("the first PQ implementation compensates legs only")

        self.input_dim = int(_cfg_value(cfg, "input_dim"))
        if self.input_dim != 3 * self.LEG_DOF_COUNT:
            raise ValueError("input_dim must be 36 for [q, qd, q_ref]")
        self.base_feature_dim = int(_cfg_value(cfg, "feature_dim"))
        self.append_bias = bool(
            _cfg_value(cfg, "append_constant_bias_feature")
        )
        self.actual_feature_dim = self.base_feature_dim + int(self.append_bias)

        self.filter_time_constant = float(
            _cfg_value(cfg, "filter_time_constant")
        )
        self.forgetting_rate = float(_cfg_value(cfg, "forgetting_rate"))
        self.adaptation_gain = float(_cfg_value(cfg, "adaptation_gain"))
        self.weight_leak = float(_cfg_value(cfg, "weight_leak"))
        self.compensation_torque_ratio = float(
            _cfg_value(cfg, "compensation_torque_ratio")
        )
        self.compensation_ramp_time = float(
            _cfg_value(cfg, "compensation_ramp_time")
        )
        self.max_weight_abs = float(_cfg_value(cfg, "max_weight_abs"))
        self.max_weight_update_abs = float(
            _cfg_value(cfg, "max_weight_update_abs")
        )
        self._validate_scalar_configuration()

        m0 = torch.as_tensor(
            _cfg_value(cfg, "m0_diag"), dtype=dtype, device=self.device
        )
        if m0.shape != (self.LEG_DOF_COUNT,) or not torch.all(m0 > 0):
            raise ValueError("m0_diag must contain twelve positive values")
        self.m0_diag = m0
        self.m0_inv = 1.0 / m0

        self.joint_position_scale = float(
            _cfg_value(cfg, "joint_position_scale")
        )
        self.joint_velocity_scale = float(
            _cfg_value(cfg, "joint_velocity_scale")
        )
        self.joint_reference_scale = float(
            _cfg_value(cfg, "joint_reference_scale")
        )
        for name, value in (
            ("joint_position_scale", self.joint_position_scale),
            ("joint_velocity_scale", self.joint_velocity_scale),
            ("joint_reference_scale", self.joint_reference_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self.feature_network = FixedPQFeatureNetwork(
            input_dim=self.input_dim,
            hidden_dims=_cfg_value(cfg, "hidden_dims"),
            feature_dim=self.base_feature_dim,
            activation=_cfg_value(cfg, "activation"),
            seed=int(_cfg_value(cfg, "feature_seed")),
        ).to(device=self.device, dtype=dtype)
        self.feature_network.requires_grad_(False)

        n, p, m = self.num_envs, self.actual_feature_dim, self.LEG_DOF_COUNT
        zeros = lambda *shape: torch.zeros(
            *shape, dtype=dtype, device=self.device, requires_grad=False
        )
        self.W_hat = zeros(n, p, m)
        self.P = zeros(n, p, p)
        self.Q = zeros(n, p, m)
        self.e2_f = zeros(n, m)
        self.H_f = zeros(n, m)
        self.u_f = zeros(n, m)
        self.phi_f = zeros(n, p)
        self.last_q_ref = zeros(n, m)
        self.q_ref = zeros(n, m)
        self.qd_ref = zeros(n, m)
        self.qd_ref_target = zeros(n, m)
        self.reference_initialized = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self.just_reset = torch.ones(n, dtype=torch.bool, device=self.device)
        self.elapsed_time = zeros(n)

        self.last_tau_comp = zeros(n, m)
        self.last_tau_comp_used = zeros(n, m)
        self.last_W_dot = zeros(n, p, m)
        self.last_composite_error = zeros(n, p, m)
        self.last_tracking_update = zeros(n, p, m)
        self.last_blend = zeros(n)
        self.last_freeze = torch.ones(n, dtype=torch.bool, device=self.device)
        self.last_freeze_contact = torch.zeros_like(self.last_freeze)
        self.last_freeze_saturation = torch.zeros_like(self.last_freeze)
        self.last_freeze_tilt = torch.zeros_like(self.last_freeze)
        self.last_nan_fault = torch.zeros_like(self.last_freeze)
        self.nan_count = zeros(n)

        self._metric_names = (
            "pq_comp_torque_mean",
            "pq_comp_torque_rms",
            "pq_comp_torque_ratio",
            "pq_weight_norm_mean",
            "pq_weight_update_norm",
            "pq_composite_error_norm",
            "pq_tracking_update_norm",
            "pq_P_trace",
            "pq_P_frobenius_norm",
            "pq_Q_frobenius_norm",
            "pq_freeze_fraction",
            "pq_freeze_contact",
            "pq_freeze_saturation",
            "pq_freeze_tilt",
        )
        self.metric_sums = {name: zeros(n) for name in self._metric_names}
        self.metric_samples = zeros(n)
        self.metric_comp_max = zeros(n)
        self.metric_weight_max = zeros(n)

    def _validate_scalar_configuration(self):
        positive = (
            ("filter_time_constant", self.filter_time_constant),
            ("forgetting_rate", self.forgetting_rate),
            ("adaptation_gain", self.adaptation_gain),
            ("max_weight_abs", self.max_weight_abs),
            ("max_weight_update_abs", self.max_weight_update_abs),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_leak) or self.weight_leak < 0.0:
            raise ValueError("weight_leak must be finite and non-negative")
        if not 0.0 <= self.compensation_torque_ratio <= 1.0:
            raise ValueError("compensation_torque_ratio must be in [0, 1]")
        if (
            not math.isfinite(self.compensation_ramp_time)
            or self.compensation_ramp_time < 0.0
        ):
            raise ValueError(
                "compensation_ramp_time must be finite and non-negative"
            )
        if bool(_cfg_value(self.cfg, "use_spectral_regularization")):
            raise ValueError("spectral regularization is reserved for a future version")
        if bool(_cfg_value(self.cfg, "use_projection")):
            raise ValueError("projection is reserved for a future version")

    def _check_joint_tensor(self, name, value):
        expected = (self.num_envs, self.LEG_DOF_COUNT)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

    def _features(self, q, qd, q_ref, default_q):
        q_input = (q - default_q) * self.joint_position_scale
        qd_input = qd * self.joint_velocity_scale
        q_ref_input = (q_ref - default_q) * self.joint_reference_scale
        inputs = torch.cat((q_input, qd_input, q_ref_input), dim=-1)
        phi = self.feature_network(inputs)
        if self.append_bias:
            phi = torch.cat(
                (
                    phi,
                    torch.ones(
                        self.num_envs,
                        1,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
        return phi

    @torch.no_grad()
    def begin_policy_step(self, q_ref, policy_dt: float):
        """Update q_ref once per actor step; never difference it per substep."""
        self._check_joint_tensor("q_ref", q_ref)
        if not math.isfinite(policy_dt) or policy_dt <= 0.0:
            raise ValueError("policy_dt must be finite and positive")
        finite = torch.isfinite(q_ref).all(dim=1)
        valid_previous = (
            finite
            & self.reference_initialized
            & torch.isfinite(self.last_q_ref).all(dim=1)
        )
        raw = torch.zeros_like(q_ref)
        raw[valid_previous] = (
            q_ref[valid_previous] - self.last_q_ref[valid_previous]
        ) / float(policy_dt)
        self.last_q_ref[finite] = q_ref[finite]
        self.q_ref[finite] = q_ref[finite]
        self.qd_ref_target[finite] = raw[finite]
        self.qd_ref_target[~finite] = 0.0
        self.reference_initialized[finite] = True

    @torch.no_grad()
    def reset(
        self,
        env_ids,
        q,
        qd,
        q_ref,
        default_q,
        kp,
        kd,
    ):
        """Reset selected runtime states and initialize filters from current input."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        for name, value in (
            ("q", q),
            ("qd", qd),
            ("q_ref", q_ref),
            ("default_q", default_q),
            ("kp", kp),
            ("kd", kd),
        ):
            self._check_joint_tensor(name, value)

        if bool(_cfg_value(self.cfg, "reset_weights_on_episode_reset")):
            self.W_hat[env_ids] = 0.0
            self.P[env_ids] = 0.0
            self.Q[env_ids] = 0.0

        self.q_ref[env_ids] = q_ref[env_ids]
        self.last_q_ref[env_ids] = q_ref[env_ids]
        self.qd_ref[env_ids] = 0.0
        self.qd_ref_target[env_ids] = 0.0
        self.reference_initialized[env_ids] = True
        self.elapsed_time[env_ids] = 0.0
        self.just_reset[env_ids] = True

        e1 = q - q_ref
        e2 = qd
        phi = self._features(q, qd, q_ref, default_q)
        delta_hat = torch.einsum("bpm,bp->bm", self.W_hat, phi)
        H = self.m0_inv * (-kp * e1 - kd * e2)
        u = -delta_hat
        self.e2_f[env_ids] = e2[env_ids]
        self.H_f[env_ids] = H[env_ids]
        self.u_f[env_ids] = u[env_ids]
        self.phi_f[env_ids] = phi[env_ids]
        self._clear_metrics(env_ids)
        self.last_tau_comp[env_ids] = 0.0
        self.last_tau_comp_used[env_ids] = 0.0

    def _clear_metrics(self, env_ids):
        for values in self.metric_sums.values():
            values[env_ids] = 0.0
        self.metric_samples[env_ids] = 0.0
        self.metric_comp_max[env_ids] = 0.0
        self.metric_weight_max[env_ids] = 0.0
        self.nan_count[env_ids] = 0.0

    @torch.no_grad()
    def step(
        self,
        q,
        qd,
        default_q,
        kp,
        kd,
        torque_limits,
        low_level_dt: float,
        contact_ok: Optional[torch.Tensor] = None,
        torque_saturated: Optional[torch.Tensor] = None,
        tilt: Optional[torch.Tensor] = None,
    ):
        """Run filters, P/Q/W adaptation, and return limited leg compensation."""
        for name, value in (
            ("q", q),
            ("qd", qd),
            ("default_q", default_q),
            ("kp", kp),
            ("kd", kd),
        ):
            self._check_joint_tensor(name, value)
        if torque_limits.shape not in (
            (self.LEG_DOF_COUNT,),
            (self.num_envs, self.LEG_DOF_COUNT),
        ):
            raise ValueError("torque_limits must have shape [12] or [N, 12]")
        if not math.isfinite(low_level_dt) or low_level_dt <= 0.0:
            raise ValueError("low_level_dt must be finite and positive")

        zeros_bool = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if contact_ok is None:
            contact_ok = ~zeros_bool
        if torque_saturated is None:
            torque_saturated = zeros_bool
        if tilt is None:
            tilt = torch.zeros(
                self.num_envs, dtype=self.dtype, device=self.device
            )

        if not self.enabled:
            self.last_tau_comp.zero_()
            self.last_tau_comp_used.zero_()
            self.last_blend.zero_()
            self.last_freeze[:] = True
            return self.last_tau_comp_used

        alpha = 1.0 - math.exp(
            -float(low_level_dt) / self.filter_time_constant
        )
        self.qd_ref += alpha * (self.qd_ref_target - self.qd_ref)
        e1 = q - self.q_ref
        e2 = qd - self.qd_ref
        phi = self._features(q, qd, self.q_ref, default_q)
        delta_hat = torch.einsum("bpm,bp->bm", self.W_hat, phi)
        H = self.m0_inv * (-kp * e1 - kd * e2)
        u = -delta_hat

        input_finite = (
            torch.isfinite(q).all(dim=1)
            & torch.isfinite(qd).all(dim=1)
            & torch.isfinite(self.q_ref).all(dim=1)
            & torch.isfinite(phi).all(dim=1)
            & torch.isfinite(H).all(dim=1)
        )
        state_finite = (
            torch.isfinite(self.W_hat).flatten(1).all(dim=1)
            & torch.isfinite(self.P).flatten(1).all(dim=1)
            & torch.isfinite(self.Q).flatten(1).all(dim=1)
            & torch.isfinite(self.e2_f).all(dim=1)
            & torch.isfinite(self.H_f).all(dim=1)
            & torch.isfinite(self.u_f).all(dim=1)
            & torch.isfinite(self.phi_f).all(dim=1)
            & torch.isfinite(self.qd_ref).all(dim=1)
            & torch.isfinite(self.qd_ref_target).all(dim=1)
        )
        nan_fault = ~(input_finite & state_finite)
        if torch.any(nan_fault):
            bad_ids = nan_fault.nonzero(as_tuple=False).flatten()
            self.W_hat[bad_ids] = 0.0
            self.P[bad_ids] = 0.0
            self.Q[bad_ids] = 0.0
            self.e2_f[bad_ids] = 0.0
            self.H_f[bad_ids] = 0.0
            self.u_f[bad_ids] = 0.0
            self.phi_f[bad_ids] = 0.0
            self.qd_ref[bad_ids] = 0.0
            self.qd_ref_target[bad_ids] = 0.0
            self.nan_count[bad_ids] += 1.0

        valid = input_finite & state_finite
        safe_e2 = torch.where(valid[:, None], e2, self.e2_f)
        safe_H = torch.where(valid[:, None], H, self.H_f)
        safe_u = torch.where(valid[:, None], u, self.u_f)
        safe_phi = torch.where(valid[:, None], phi, self.phi_f)
        self.e2_f += alpha * (safe_e2 - self.e2_f)
        self.H_f += alpha * (safe_H - self.H_f)
        self.u_f += alpha * (safe_u - self.u_f)
        self.phi_f += alpha * (safe_phi - self.phi_f)
        e2_f_dot = (safe_e2 - self.e2_f) / self.filter_time_constant
        residual = e2_f_dot - self.H_f - self.u_f

        phi_outer = torch.einsum("bi,bj->bij", self.phi_f, self.phi_f)
        residual_outer = torch.einsum(
            "bi,bj->bij", self.phi_f, residual
        )
        p_dot = (
            -self.forgetting_rate * self.P
            + self.forgetting_rate * phi_outer
        )
        q_dot = (
            -self.forgetting_rate * self.Q
            + self.forgetting_rate * residual_outer
        )

        ramp_incomplete = (
            self.elapsed_time < self.compensation_ramp_time
            if self.compensation_ramp_time > 0.0
            else zeros_bool
        )
        freeze_contact = (
            (~contact_ok)
            if bool(_cfg_value(self.cfg, "freeze_on_contact_loss"))
            else zeros_bool
        )
        freeze_saturation = (
            torque_saturated
            if bool(_cfg_value(self.cfg, "freeze_on_torque_saturation"))
            else zeros_bool
        )
        freeze_tilt = (
            tilt > float(_cfg_value(self.cfg, "excessive_tilt_threshold"))
            if bool(_cfg_value(self.cfg, "freeze_on_excessive_tilt"))
            else zeros_bool
        )
        freeze_reset = (
            self.just_reset
            if bool(_cfg_value(self.cfg, "freeze_on_reset"))
            else zeros_bool
        )
        freeze_ramp = (
            ramp_incomplete
            if bool(_cfg_value(self.cfg, "freeze_during_ramp"))
            else zeros_bool
        )
        freeze = (
            freeze_contact
            | freeze_saturation
            | freeze_tilt
            | freeze_reset
            | freeze_ramp
            | nan_fault
        )
        update_mask = (~freeze).to(self.dtype)
        self.P += float(low_level_dt) * p_dot * update_mask[:, None, None]
        self.Q += float(low_level_dt) * q_dot * update_mask[:, None, None]
        self.P.copy_(0.5 * (self.P + self.P.transpose(-1, -2)))

        composite_error = self.Q - torch.bmm(self.P, self.W_hat)
        tracking_outer = torch.einsum("bi,bj->bij", safe_phi, safe_e2)
        w_dot = (
            self.adaptation_gain * (composite_error + tracking_outer)
            - self.weight_leak * self.W_hat
        )
        w_dot.clamp_(
            -self.max_weight_update_abs, self.max_weight_update_abs
        )
        w_dot *= update_mask[:, None, None]
        self.W_hat += float(low_level_dt) * w_dot
        self.W_hat.clamp_(-self.max_weight_abs, self.max_weight_abs)

        delta_hat = torch.einsum("bpm,bp->bm", self.W_hat, safe_phi)
        tau_comp = -self.m0_diag * delta_hat
        limits = self.compensation_torque_ratio * torque_limits
        tau_comp = torch.maximum(torch.minimum(tau_comp, limits), -limits)
        if self.compensation_ramp_time > 0.0:
            blend = torch.clamp(
                self.elapsed_time / self.compensation_ramp_time, 0.0, 1.0
            )
        else:
            blend = torch.ones_like(self.elapsed_time)
        tau_used = blend[:, None] * tau_comp
        if self.shadow_mode:
            tau_used.zero_()
        tau_used[nan_fault] = 0.0

        self.last_tau_comp.copy_(tau_comp)
        self.last_tau_comp_used.copy_(tau_used)
        self.last_W_dot.copy_(w_dot)
        self.last_composite_error.copy_(composite_error)
        self.last_tracking_update.copy_(tracking_outer)
        self.last_blend.copy_(blend)
        self.last_freeze.copy_(freeze)
        self.last_freeze_contact.copy_(freeze_contact)
        self.last_freeze_saturation.copy_(freeze_saturation)
        self.last_freeze_tilt.copy_(freeze_tilt)
        self.last_nan_fault.copy_(nan_fault)
        self._accumulate_metrics(torque_limits)
        self.elapsed_time += float(low_level_dt)
        self.just_reset.zero_()
        return self.last_tau_comp_used

    def _accumulate_metrics(self, torque_limits):
        tau_abs = torch.abs(self.last_tau_comp)
        weight_norm = torch.linalg.vector_norm(self.W_hat, dim=(1, 2))
        values = {
            "pq_comp_torque_mean": torch.mean(tau_abs, dim=1),
            "pq_comp_torque_rms": torch.sqrt(
                torch.mean(torch.square(self.last_tau_comp), dim=1)
            ),
            "pq_comp_torque_ratio": torch.max(
                tau_abs / torch.clamp(torque_limits, min=1.0e-8), dim=1
            ).values,
            "pq_weight_norm_mean": weight_norm,
            "pq_weight_update_norm": torch.linalg.vector_norm(
                self.last_W_dot, dim=(1, 2)
            ),
            "pq_composite_error_norm": torch.linalg.vector_norm(
                self.last_composite_error, dim=(1, 2)
            ),
            "pq_tracking_update_norm": torch.linalg.vector_norm(
                self.last_tracking_update, dim=(1, 2)
            ),
            "pq_P_trace": torch.diagonal(
                self.P, dim1=-2, dim2=-1
            ).sum(dim=1),
            "pq_P_frobenius_norm": torch.linalg.vector_norm(
                self.P, dim=(1, 2)
            ),
            "pq_Q_frobenius_norm": torch.linalg.vector_norm(
                self.Q, dim=(1, 2)
            ),
            "pq_freeze_fraction": self.last_freeze.to(self.dtype),
            "pq_freeze_contact": self.last_freeze_contact.to(self.dtype),
            "pq_freeze_saturation": self.last_freeze_saturation.to(self.dtype),
            "pq_freeze_tilt": self.last_freeze_tilt.to(self.dtype),
        }
        for name, value in values.items():
            self.metric_sums[name] += value
        self.metric_samples += 1.0
        self.metric_comp_max = torch.maximum(
            self.metric_comp_max, torch.max(tau_abs, dim=1).values
        )
        self.metric_weight_max = torch.maximum(
            self.metric_weight_max, weight_norm
        )

    @torch.no_grad()
    def episode_metrics(self, env_ids) -> Dict[str, torch.Tensor]:
        """Return low-frequency aggregates, including P eigendiagnostics."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return {}
        denom = torch.clamp(self.metric_samples[env_ids], min=1.0)
        result = {
            "pq_enabled": torch.tensor(
                float(self.enabled), device=self.device
            ),
            "pq_shadow_mode": torch.tensor(
                float(self.shadow_mode), device=self.device
            ),
        }
        for name, total in self.metric_sums.items():
            result[name] = torch.mean(total[env_ids] / denom)
        result["pq_comp_torque_max"] = torch.max(
            self.metric_comp_max[env_ids]
        )
        result["pq_weight_norm_max"] = torch.max(
            self.metric_weight_max[env_ids]
        )
        result["pq_nan_count"] = torch.sum(self.nan_count[env_ids])

        # This is intentionally called at episode/report frequency, never in the
        # physics loop.
        symmetric_p = 0.5 * (
            self.P[env_ids] + self.P[env_ids].transpose(-1, -2)
        )
        eigenvalues = torch.linalg.eigvalsh(symmetric_p)
        min_eigenvalue = torch.min(eigenvalues, dim=1).values
        max_eigenvalue = torch.max(eigenvalues, dim=1).values
        condition = max_eigenvalue / torch.clamp(
            min_eigenvalue, min=1.0e-8
        )
        result["pq_P_min_eigenvalue"] = torch.mean(min_eigenvalue)
        result["pq_P_condition_estimate"] = torch.mean(condition)
        return result


def export_pq_feature_network(compensator: ZGWTPQCompensator, path: str):
    """Save deterministic feature weights and deployment-relevant metadata."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    cfg = compensator.cfg
    payload = {
        "format_version": 1,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in compensator.feature_network.state_dict().items()
        },
        "activation": compensator.feature_network.activation_name,
        "input_dim": compensator.input_dim,
        "hidden_dims": list(_cfg_value(cfg, "hidden_dims")),
        "feature_dim": compensator.base_feature_dim,
        "append_constant_bias_feature": compensator.append_bias,
        "actual_feature_dim": compensator.actual_feature_dim,
        "feature_seed": int(_cfg_value(cfg, "feature_seed")),
        "input_normalization": {
            "joint_position_scale": compensator.joint_position_scale,
            "joint_velocity_scale": compensator.joint_velocity_scale,
            "joint_reference_scale": compensator.joint_reference_scale,
        },
        "m0_diag": compensator.m0_diag.detach().cpu(),
        "filter_time_constant": compensator.filter_time_constant,
        "forgetting_rate": compensator.forgetting_rate,
        "adaptation_gain": compensator.adaptation_gain,
        "weight_leak": compensator.weight_leak,
        "compensation_torque_ratio": compensator.compensation_torque_ratio,
    }
    torch.save(payload, path)
    return path
