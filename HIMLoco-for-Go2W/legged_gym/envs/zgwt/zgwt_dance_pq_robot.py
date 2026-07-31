"""Isolated ZGWT dance environment with a twelve-leg-joint PQ compensator."""

import torch

from .zgwt_dance_pq_config import ZGWTDancePQCfg
from .zgwt_dance_robot import ZgwtDance
from .zgwt_dance_utils import effective_actions, select_wheel_factors
from .zgwt_pq_compensator import (
    ZGWTPQCompensator,
    export_pq_feature_network,
)


class ZgwtDancePQ(ZgwtDance):
    """ZGWT load-stand task with substep-rate neural P/Q compensation."""

    cfg: ZGWTDancePQCfg

    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.control.control_type != "P":
            raise ValueError("ZGWT dance PQ currently requires position-PD control")
        if self.num_dof != 16 or len(self.wheel_indices) != 4:
            raise ValueError("ZGWT dance PQ requires 16 DOFs including four wheels")

        leg_mask = torch.ones(
            self.num_dof, dtype=torch.bool, device=self.device
        )
        leg_mask[self.wheel_indices] = False
        self.leg_indices = leg_mask.nonzero(as_tuple=False).flatten()
        if self.leg_indices.numel() != ZGWTPQCompensator.LEG_DOF_COUNT:
            raise ValueError("ZGWT dance PQ requires exactly twelve leg joints")
        self.leg_dof_names = [self.dof_names[i] for i in self.leg_indices.tolist()]

        self.pq_compensator = ZGWTPQCompensator(
            self.num_envs, self.cfg.pq, self.device
        )
        q_ref = self.default_dof_pos[:, self.leg_indices].expand(
            self.num_envs, -1
        )
        kp, kd = self._effective_leg_gains()
        all_envs = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.pq_compensator.reset(
            all_envs,
            self.dof_pos[:, self.leg_indices],
            self.dof_vel[:, self.leg_indices],
            q_ref,
            self.default_dof_pos[:, self.leg_indices].expand(
                self.num_envs, -1
            ),
            kp,
            kd,
        )
        print(
            "ZGWT PQ timing: "
            f"policy_dt={self.dt:.6f}s, "
            f"physics_dt={self.sim_params.dt:.6f}s, "
            f"control_decimation={self.cfg.control.decimation}, "
            f"pq_update_frequency={1.0 / self.sim_params.dt:.1f}Hz"
        )
        print("ZGWT PQ leg order:", self.leg_dof_names)

    def step(self, actions):
        """Latch a clipped leg reference once, then update PQ in each substep."""
        masked = effective_actions(actions, self.wheel_indices)
        clipped = torch.clamp(
            masked,
            -float(self.cfg.normalization.clip_actions),
            float(self.cfg.normalization.clip_actions),
        ).to(self.device)
        q_ref = (
            self.default_dof_pos[:, self.leg_indices]
            + float(self.cfg.control.action_scale)
            * clipped[:, self.leg_indices]
        )
        self.pq_compensator.begin_policy_step(q_ref, self.dt)
        return super().step(masked)

    def reset_idx(self, env_ids):
        pq_metrics = {}
        if hasattr(self, "pq_compensator") and len(env_ids) > 0:
            pq_metrics = self.pq_compensator.episode_metrics(env_ids)

        super().reset_idx(env_ids)

        if not hasattr(self, "pq_compensator") or len(env_ids) == 0:
            return
        q_ref = self.default_dof_pos[:, self.leg_indices].expand(
            self.num_envs, -1
        )
        kp, kd = self._effective_leg_gains()
        self.pq_compensator.reset(
            env_ids,
            self.dof_pos[:, self.leg_indices],
            self.dof_vel[:, self.leg_indices],
            q_ref,
            self.default_dof_pos[:, self.leg_indices].expand(
                self.num_envs, -1
            ),
            kp,
            kd,
        )
        if "episode" in self.extras:
            self.extras["episode"].update(pq_metrics)

    def _select_leg_factors(self, factors):
        if factors.ndim != 2 or factors.shape[0] != self.num_envs:
            raise RuntimeError("gain factors must have shape [num_envs, 1|num_dof]")
        if factors.shape[1] == 1:
            return factors
        if factors.shape[1] == self.num_dof:
            return factors[:, self.leg_indices]
        raise RuntimeError("gain factors must have shape [num_envs, 1|num_dof]")

    def _effective_leg_gains(self):
        kp = self.p_gains[self.leg_indices].unsqueeze(0) * self._select_leg_factors(
            self.Kp_factors
        )
        kd = self.d_gains[self.leg_indices].unsqueeze(0) * self._select_leg_factors(
            self.Kd_factors
        )
        return kp.expand(self.num_envs, -1), kd.expand(self.num_envs, -1)

    def _nominal_torques_before_clip_and_strength(self, actions):
        """Reproduce the dance P controller before either torque clip."""
        actions = effective_actions(actions, self.wheel_indices)
        actions_scaled = actions * float(self.cfg.control.action_scale)
        pos_actions_scaled = actions_scaled.clone()
        pos_actions_scaled[:, self.wheel_indices] = 0.0
        dof_error = self.default_dof_pos - self.dof_pos
        dof_error = dof_error.clone()
        dof_error[:, self.wheel_indices] = 0.0
        vel_ref = torch.zeros_like(actions_scaled)

        torques = (
            self.p_gains
            * self.Kp_factors
            * (pos_actions_scaled + dof_error)
            + self.d_gains
            * self.Kd_factors
            * (vel_ref - self.dof_vel)
        )

        wheel_error = (
            self.wheel_park_targets - self.dof_pos[:, self.wheel_indices]
        )
        try:
            wheel_kd_factors = select_wheel_factors(
                self.Kd_factors, self.wheel_indices, self.num_dof
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        torques[:, self.wheel_indices] = (
            float(self.cfg.control.wheel_park_stiffness) * wheel_error
            - float(self.cfg.control.wheel_park_damping)
            * wheel_kd_factors
            * self.dof_vel[:, self.wheel_indices]
        )
        return torques

    def _compute_torques(self, actions):
        # Mandatory exact disabled regression path: no duplicated approximation.
        if not bool(self.cfg.pq.enabled):
            self.pq_compensator.enabled = False
            return super()._compute_torques(actions)

        self.pq_compensator.enabled = True
        self.pq_compensator.shadow_mode = bool(self.cfg.pq.shadow_mode)
        nominal = self._nominal_torques_before_clip_and_strength(actions)
        kp, kd = self._effective_leg_gains()
        leg_strength = self._select_leg_factors(self.motor_strength_factors)
        leg_nominal_physical = nominal[:, self.leg_indices] * leg_strength
        torque_saturated = torch.any(
            torch.abs(leg_nominal_physical)
            >= 0.99 * self.torque_limits[self.leg_indices],
            dim=1,
        )
        contact_ok = torch.all(
            self.contact_forces[:, self.feet_indices, 2] >= 5.0, dim=1
        )
        roll, pitch = self._current_roll_pitch()
        tilt = torch.maximum(torch.abs(roll), torch.abs(pitch))

        tau_comp = self.pq_compensator.step(
            q=self.dof_pos[:, self.leg_indices],
            qd=self.dof_vel[:, self.leg_indices],
            default_q=self.default_dof_pos[:, self.leg_indices].expand(
                self.num_envs, -1
            ),
            kp=kp,
            kd=kd,
            torque_limits=self.torque_limits[self.leg_indices],
            low_level_dt=float(self.sim_params.dt),
            contact_ok=contact_ok,
            torque_saturated=torque_saturated,
            tilt=tilt,
        )

        # Shadow mode must reproduce the original physics path even if future
        # experiments re-enable motor-strength randomization.
        if bool(self.cfg.pq.shadow_mode):
            return super()._compute_torques(actions)

        nominal[:, self.leg_indices] += tau_comp
        if self.motor_strength_factors.ndim != 2 or self.motor_strength_factors.shape[
            0
        ] != self.num_envs or self.motor_strength_factors.shape[1] not in (
            1,
            self.num_dof,
        ):
            raise RuntimeError(
                "motor_strength_factors must have shape [num_envs, 1|num_dof]"
            )
        torques = nominal * self.motor_strength_factors
        return torch.clamp(torques, -self.torque_limits, self.torque_limits)

    def set_pq_mode(self, mode):
        if mode not in ("off", "shadow", "active"):
            raise ValueError("PQ mode must be 'off', 'shadow', or 'active'")
        enabled = mode != "off"
        shadow = mode == "shadow"
        self.cfg.pq.enabled = enabled
        self.cfg.pq.shadow_mode = shadow
        self.pq_compensator.enabled = enabled
        self.pq_compensator.shadow_mode = shadow

    def export_pq_features(self, path):
        return export_pq_feature_network(self.pq_compensator, path)

    def _process_rigid_body_props(self, props, env_id):
        nominal_com = (
            float(props[0].com.x),
            float(props[0].com.y),
            float(props[0].com.z),
        )
        if env_id == 0:
            self._pq_nominal_base_com = nominal_com
        props = super()._process_rigid_body_props(props, env_id)

        fixed_payload = getattr(self.cfg.pq, "fixed_payload_mass", None)
        if fixed_payload is not None:
            payload = float(fixed_payload)
            if payload < 0.0:
                raise ValueError("fixed_payload_mass must be non-negative")
            props[0].mass = float(self.default_rigid_body_mass[0]) + payload
            self.payload[env_id, 0] = payload

        fixed_com = getattr(self.cfg.pq, "fixed_com_offset", None)
        if fixed_com is not None:
            if len(fixed_com) != 3:
                raise ValueError("fixed_com_offset must contain x, y, z")
            from isaacgym import gymapi

            props[0].com = gymapi.Vec3(
                nominal_com[0] + float(fixed_com[0]),
                nominal_com[1] + float(fixed_com[1]),
                nominal_com[2] + float(fixed_com[2]),
            )
            self.com_displacement[env_id] = torch.tensor(
                fixed_com, dtype=torch.float, device=self.device
            )
        return props

    def set_payload_mass_and_com(self, payload_mass, com_offset=None, env_ids=None):
        """Apply a true runtime base mass/COM step for Phase-A evaluation."""
        if payload_mass < 0.0:
            raise ValueError("payload_mass must be non-negative")
        if com_offset is not None and len(com_offset) != 3:
            raise ValueError("com_offset must contain x, y, z")
        if env_ids is None:
            env_ids = range(self.num_envs)
        nominal_com = self._pq_nominal_base_com
        from isaacgym import gymapi

        for env_id in env_ids:
            env_id = int(env_id)
            props = self.gym.get_actor_rigid_body_properties(
                self.envs[env_id], self.actor_handles[env_id]
            )
            props[0].mass = (
                float(self.default_rigid_body_mass[0]) + float(payload_mass)
            )
            if com_offset is not None:
                props[0].com = gymapi.Vec3(
                    nominal_com[0] + float(com_offset[0]),
                    nominal_com[1] + float(com_offset[1]),
                    nominal_com[2] + float(com_offset[2]),
                )
            self.gym.set_actor_rigid_body_properties(
                self.envs[env_id],
                self.actor_handles[env_id],
                props,
                recomputeInertia=True,
            )
            self.payload[env_id, 0] = float(payload_mass)
            if com_offset is not None:
                self.com_displacement[env_id] = torch.tensor(
                    com_offset, dtype=torch.float, device=self.device
                )
