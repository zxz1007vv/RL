import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.utils.math import get_scale_shift

from .zgwt_robot import Zgwt
from .zgwt_dance_config import ZGWTDanceCfg
from .zgwt_dance_utils import (
    effective_actions,
    floored_soft_multiplier,
    low_pass_alpha,
    neutral_command_weight,
    normalize_dance_commands,
    select_wheel_factors,
    target_projected_gravity,
)


class ZgwtDance(Zgwt):
    """Fixed-support task tracking bounded body yaw, roll, pitch, and height."""

    cfg: ZGWTDanceCfg

    ROLL_COMMAND = 3
    PITCH_COMMAND = 4
    HEIGHT_COMMAND = 5

    # Pose commands are the primary task. Stationary support is tightened only
    # after the commanded pose is approached; all remaining terms are costs.
    TRACK_REWARD_NAMES = frozenset(
        {
            "tracking_body_orientation",
            "tracking_body_yaw",
            "tracking_body_height",
        }
    )
    HOLD_REWARD_NAMES = frozenset(
        {
            "tracking_lin_vx",
            "tracking_lin_vy",
            "tracking_support_position",
            "tracking_feet_position",
            "tracking_max_foot_position",
        }
    )
    SOFT_REWARD_NAMES = frozenset(
        {
            "yaw_rate",
            "feet_vertical_motion",
            "action_rate",
            "action_smoothness",
            "torques",
            "neutral_joint_pose",
        }
    )
    HARD_REWARD_NAMES = frozenset(
        {
            "collision",
            "feet_contact",
            "feet_stumble",
            "dof_pos_limits",
            "torque_limits",
            "severe_wheel_park",
            "severe_support_loss",
        }
    )

    def _init_buffers(self):
        super()._init_buffers()
        expected_actor = 3 + 3 + 6 + 3 * self.num_actions
        expected_privileged = expected_actor + 3 + 3 + 3 * len(self.feet_indices)
        if self.cfg.env.num_one_step_observations != 60 or expected_actor != 60:
            raise ValueError(
                "ZGWT dance actor observation must be 60-D "
                f"(configured={self.cfg.env.num_one_step_observations}, actual={expected_actor})"
            )
        if self.cfg.env.num_observations != 360:
            raise ValueError("ZGWT dance history observation must be 6 x 60 = 360-D")
        if self.cfg.env.num_one_step_privileged_obs != expected_privileged:
            raise ValueError(
                "ZGWT dance privileged observation mismatch: "
                f"configured={self.cfg.env.num_one_step_privileged_obs}, "
                f"actual={expected_privileged}"
            )
        if self.cfg.terrain.measure_heights:
            raise ValueError("ZGWT dance is a plane task and must not measure heights")
        self.pose_command_targets = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.yaw_command_targets = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.commands[:, self.HEIGHT_COMMAND] = self.cfg.rewards.default_body_height
        self.pose_command_targets[:, 2] = self.cfg.rewards.default_body_height
        self.episode_start_xy = self.root_states[:, :2].clone()
        self.episode_start_yaw = self._current_yaw().clone()
        self.episode_start_support_xy = torch.mean(
            self.feet_pos[:, :, :2], dim=1
        ).clone()
        self.episode_start_feet_xy = self.feet_pos[:, :, :2].clone()
        self.support_anchor_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_park_targets = self.dof_pos[:, self.wheel_indices].clone()
        self._resampling_for_reset = False

        self.command_mode_probabilities = self._normalized_probabilities(
            self.cfg.commands.mode_probabilities, "manual"
        )

    def step(self, actions):
        """Discard policy wheel channels before delay, control, reward and history."""
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"expected actions [{self.num_envs}, {self.num_actions}], "
                f"got {list(actions.shape)}"
            )
        return super().step(effective_actions(actions, self.wheel_indices))

    def _normalized_probabilities(self, values, label):
        probabilities = torch.tensor(
            values, dtype=torch.float, device=self.device
        )
        if probabilities.numel() != 16 or torch.any(probabilities < 0):
            raise ValueError(
                f"dance {label}_mode_probabilities must contain "
                "16 non-negative values"
            )
        probability_sum = torch.sum(probabilities)
        if not torch.isclose(
            probability_sum,
            torch.tensor(1.0, dtype=torch.float, device=self.device),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"dance {label}_mode_probabilities must sum to 1.0, "
                f"got {probability_sum.item():.8f}"
            )
        return probabilities

    def _get_noise_scale_vec(self, cfg):
        command_dim = cfg.commands.num_commands
        actor_obs_dim = 6 + command_dim + 3 * self.num_actions
        height_dim = 187 if cfg.terrain.measure_heights else 0
        noise_vec = torch.zeros(actor_obs_dim + height_dim, device=self.device)

        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        obs_scales = cfg.normalization.obs_scales

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        command_end = 6 + command_dim
        noise_vec[6:command_end] = 0.0
        noise_vec[command_end : command_end + self.num_actions] = (
            noise_scales.dof_pos * noise_level * obs_scales.dof_pos
        )
        noise_vec[
            command_end + self.num_actions : command_end + 2 * self.num_actions
        ] = noise_scales.dof_vel * noise_level * obs_scales.dof_vel
        if height_dim:
            noise_vec[actor_obs_dim:] = (
                noise_scales.height_measurements
                * noise_level
                * obs_scales.height_measurements
            )
        return noise_vec

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        roll, pitch = self._current_roll_pitch()
        height = self._current_base_height()
        roll_abs_error = torch.abs(
            self.commands[env_ids, self.ROLL_COMMAND] - roll[env_ids]
        )
        pitch_abs_error = torch.abs(
            self.commands[env_ids, self.PITCH_COMMAND] - pitch[env_ids]
        )
        yaw_abs_error = torch.abs(self._body_yaw_error()[env_ids])
        height_abs_error = torch.abs(
            self.commands[env_ids, self.HEIGHT_COMMAND] - height[env_ids]
        )

        roll_active = (
            torch.abs(self.pose_command_targets[env_ids, 0])
            >= self.cfg.rewards.active_orientation_threshold
        ).float()
        pitch_active = (
            torch.abs(self.pose_command_targets[env_ids, 1])
            >= self.cfg.rewards.active_orientation_threshold
        ).float()
        yaw_active = (
            torch.abs(self.yaw_command_targets[env_ids])
            >= self.cfg.rewards.active_yaw_threshold
        ).float()
        height_active = (
            torch.abs(
                self.pose_command_targets[env_ids, 2]
                - self.cfg.rewards.default_body_height
            )
            >= self.cfg.rewards.active_height_threshold
        ).float()

        xy_drift = torch.mean(
            torch.norm(
                self.root_states[env_ids, :2] - self.episode_start_xy[env_ids],
                dim=1,
            )
        )
        support_xy = torch.mean(self.feet_pos[env_ids, :, :2], dim=1)
        support_xy_drift = torch.mean(
            torch.norm(
                support_xy - self.episode_start_support_xy[env_ids], dim=1
            )
        )
        individual_wheel_drift = torch.norm(
            self.feet_pos[env_ids, :, :2]
            - self.episode_start_feet_xy[env_ids],
            dim=2,
        )
        feet_xy_drift = torch.mean(individual_wheel_drift)
        max_individual_wheel_drift = torch.mean(
            torch.max(individual_wheel_drift, dim=1).values
        )

        aux_episode_mean = torch.zeros(
            len(env_ids), dtype=torch.float, device=self.device
        )
        if "aux_multiplier" in self.episode_sums:
            elapsed = torch.clamp(
                self.episode_length_buf[env_ids].float() * self.dt,
                min=self.dt,
            )
            aux_episode_mean = (
                self.episode_sums["aux_multiplier"][env_ids] / elapsed
            )

        fall_rate = torch.mean((~self.time_out_buf[env_ids]).float())
        contact_loss_ratio = torch.mean(
            (
                self.contact_forces[
                    env_ids[:, None], self.feet_indices, 2
                ]
                < 5.0
            ).float()
        )
        collision_ratio = torch.mean(
            torch.any(
                torch.norm(
                    self.contact_forces[
                        env_ids[:, None],
                        self.penalised_contact_indices,
                        :,
                    ],
                    dim=-1,
                )
                > 0.1,
                dim=1,
            ).float()
        )
        joint_limit_hit_ratio = torch.mean(
            (
                (self.dof_pos[env_ids] <= self.dof_pos_limits[:, 0])
                | (self.dof_pos[env_ids] >= self.dof_pos_limits[:, 1])
            ).float()
        )
        torque_saturation_ratio = torch.mean(
            (
                torch.abs(self.torques[env_ids])
                >= 0.99 * self.torque_limits
            ).float()
        )
        action_saturation_ratio = torch.mean(
            (
                torch.abs(self.actions[env_ids])
                >= 0.99 * float(self.cfg.normalization.clip_actions)
            ).float()
        )

        self._resampling_for_reset = True
        super().reset_idx(env_ids)
        self._resampling_for_reset = False

        self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.episode_start_yaw[env_ids] = self._current_yaw()[env_ids]
        self.wheel_park_targets[env_ids] = self.dof_pos[
            env_ids[:, None], self.wheel_indices
        ]
        self.support_anchor_pending[env_ids] = True

        episode = self.extras["episode"]
        episode["mean_abs_roll_error_active"] = self._masked_mean(
            roll_abs_error, roll_active
        )
        episode["mean_abs_pitch_error_active"] = self._masked_mean(
            pitch_abs_error, pitch_active
        )
        episode["mean_abs_yaw_error_active"] = self._masked_mean(
            yaw_abs_error, yaw_active
        )
        episode["mean_abs_height_error_active"] = self._masked_mean(
            height_abs_error, height_active
        )
        episode["mean_abs_roll_error_inactive"] = self._masked_mean(
            roll_abs_error, 1.0 - roll_active
        )
        episode["mean_abs_pitch_error_inactive"] = self._masked_mean(
            pitch_abs_error, 1.0 - pitch_active
        )
        episode["mean_abs_yaw_error_inactive"] = self._masked_mean(
            yaw_abs_error, 1.0 - yaw_active
        )
        episode["mean_abs_height_error_inactive"] = self._masked_mean(
            height_abs_error, 1.0 - height_active
        )
        episode["mean_xy_drift"] = xy_drift
        episode["mean_support_xy_drift"] = support_xy_drift
        episode["mean_feet_xy_drift"] = feet_xy_drift
        episode["max_individual_wheel_drift"] = (
            max_individual_wheel_drift
        )
        episode["aux_multiplier_mean"] = torch.mean(aux_episode_mean)
        episode["aux_multiplier_p05"] = torch.quantile(
            aux_episode_mean, 0.05
        )
        episode["fall_rate"] = fall_rate
        episode["contact_loss_ratio"] = contact_loss_ratio
        episode["collision_ratio"] = collision_ratio
        episode["joint_limit_hit_ratio"] = joint_limit_hit_ratio
        episode["torque_saturation_ratio"] = torque_saturation_ratio
        episode["action_saturation_ratio"] = action_saturation_ratio

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        yaw_range = self.command_ranges["body_yaw"]
        roll_range = self.command_ranges["body_roll"]
        pitch_range = self.command_ranges["body_pitch"]
        height_range = self.command_ranges["body_height"]
        count = len(env_ids)

        sampled_yaw = torch_rand_float(
            yaw_range[0], yaw_range[1], (count, 1), device=self.device
        ).squeeze(1)
        sampled_roll = torch_rand_float(
            roll_range[0], roll_range[1], (count, 1), device=self.device
        ).squeeze(1)
        sampled_pitch = torch_rand_float(
            pitch_range[0], pitch_range[1], (count, 1), device=self.device
        ).squeeze(1)
        sampled_height = torch_rand_float(
            height_range[0], height_range[1], (count, 1), device=self.device
        ).squeeze(1)

        # Mix isolated and combined commands so each degree of freedom is learned
        # before the policy sees the hardest corner combinations.
        mode = torch.multinomial(
            self.command_mode_probabilities, count, replacement=True
        )
        targets = torch.zeros(count, 3, dtype=torch.float, device=self.device)
        targets[:, 2] = self.cfg.rewards.default_body_height
        yaw_targets = torch.zeros(count, dtype=torch.float, device=self.device)
        height_only = mode == 1
        roll_only = mode == 2
        pitch_only = mode == 3
        yaw_only = mode == 4
        yaw_targets[yaw_only] = sampled_yaw[yaw_only]
        targets[height_only, 2] = sampled_height[height_only]
        targets[roll_only, 0] = sampled_roll[roll_only]
        targets[pitch_only, 1] = sampled_pitch[pitch_only]
        roll_pitch = mode == 5
        targets[roll_pitch, 0] = sampled_roll[roll_pitch]
        targets[roll_pitch, 1] = sampled_pitch[roll_pitch]
        roll_yaw = mode == 6
        targets[roll_yaw, 0] = sampled_roll[roll_yaw]
        yaw_targets[roll_yaw] = sampled_yaw[roll_yaw]
        pitch_yaw = mode == 7
        targets[pitch_yaw, 1] = sampled_pitch[pitch_yaw]
        yaw_targets[pitch_yaw] = sampled_yaw[pitch_yaw]
        roll_pitch_yaw = mode == 8
        targets[roll_pitch_yaw, 0] = sampled_roll[roll_pitch_yaw]
        targets[roll_pitch_yaw, 1] = sampled_pitch[roll_pitch_yaw]
        yaw_targets[roll_pitch_yaw] = sampled_yaw[roll_pitch_yaw]

        height_roll = mode == 9
        targets[height_roll, 0] = sampled_roll[height_roll]
        targets[height_roll, 2] = sampled_height[height_roll]
        height_pitch = mode == 10
        targets[height_pitch, 1] = sampled_pitch[height_pitch]
        targets[height_pitch, 2] = sampled_height[height_pitch]
        height_yaw = mode == 11
        targets[height_yaw, 2] = sampled_height[height_yaw]
        yaw_targets[height_yaw] = sampled_yaw[height_yaw]
        height_roll_pitch = mode == 12
        targets[height_roll_pitch, 0] = sampled_roll[height_roll_pitch]
        targets[height_roll_pitch, 1] = sampled_pitch[height_roll_pitch]
        targets[height_roll_pitch, 2] = sampled_height[height_roll_pitch]
        height_roll_yaw = mode == 13
        targets[height_roll_yaw, 0] = sampled_roll[height_roll_yaw]
        targets[height_roll_yaw, 2] = sampled_height[height_roll_yaw]
        yaw_targets[height_roll_yaw] = sampled_yaw[height_roll_yaw]
        height_pitch_yaw = mode == 14
        targets[height_pitch_yaw, 1] = sampled_pitch[height_pitch_yaw]
        targets[height_pitch_yaw, 2] = sampled_height[height_pitch_yaw]
        yaw_targets[height_pitch_yaw] = sampled_yaw[height_pitch_yaw]
        all_commands = mode == 15
        targets[all_commands, 0] = sampled_roll[all_commands]
        targets[all_commands, 1] = sampled_pitch[all_commands]
        targets[all_commands, 2] = sampled_height[all_commands]
        yaw_targets[all_commands] = sampled_yaw[all_commands]

        self.yaw_command_targets[env_ids] = yaw_targets
        self.pose_command_targets[env_ids] = targets
        self.commands[env_ids, :2] = 0.0
        if self._resampling_for_reset:
            # Start every reset from the nominal pose and then ramp to the target.
            self.commands[env_ids, 2] = 0.0
            self.commands[env_ids, self.ROLL_COMMAND : self.HEIGHT_COMMAND + 1] = 0.0
            self.commands[env_ids, self.HEIGHT_COMMAND] = (
                self.cfg.rewards.default_body_height
            )

    def _post_physics_step_callback(self):
        pending_ids = self.support_anchor_pending.nonzero(
            as_tuple=False
        ).flatten()
        if len(pending_ids) > 0:
            self.episode_start_support_xy[pending_ids] = torch.mean(
                self.feet_pos[pending_ids, :, :2], dim=1
            )
            self.episode_start_feet_xy[pending_ids] = self.feet_pos[
                pending_ids, :, :2
            ]
            self.support_anchor_pending[pending_ids] = False

        resample_steps = max(
            1, int(self.cfg.commands.resampling_time / self.dt)
        )
        env_ids = (self.episode_length_buf % resample_steps == 0).nonzero(
            as_tuple=False
        ).flatten()
        self._resample_commands(env_ids)

        alpha = low_pass_alpha(
            self.dt, float(self.cfg.commands.command_filter_time_constant)
        )
        max_yaw_step = float(self.cfg.commands.yaw_slew_rate) * self.dt
        yaw_step = torch.clamp(
            self.yaw_command_targets - self.commands[:, 2],
            min=-max_yaw_step,
            max=max_yaw_step,
        )
        self.commands[:, 2] += yaw_step
        pose_commands = self.commands[
            :, self.ROLL_COMMAND : self.HEIGHT_COMMAND + 1
        ]
        pose_commands[:] = torch.lerp(pose_commands, self.pose_command_targets, alpha)
        self.commands[:, :2] = 0.0

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (
            self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0
        ):
            self._disturbance_robots()

    def _build_current_observation(self, add_noise):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0
        observed_commands = self.commands.clone()
        # Absolute yaw is unobservable from projected gravity. Supplying the
        # relative yaw error keeps the actor input Markovian without adding a
        # new observation dimension.
        observed_commands[:, 2] = self._body_yaw_error()
        observed_commands = normalize_dance_commands(
            observed_commands,
            default_height=float(self.cfg.rewards.default_body_height),
            min_height=float(self.cfg.commands.min_height),
            max_abs_yaw=float(self.cfg.commands.max_abs_yaw),
            max_abs_roll=float(self.cfg.commands.max_abs_roll),
            max_abs_pitch=float(self.cfg.commands.max_abs_pitch),
        )

        actor_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                observed_commands,
                dof_err * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        if actor_obs.shape != (self.num_envs, 60):
            raise RuntimeError(
                f"actor observation shape must be [{self.num_envs}, 60], "
                f"got {list(actor_obs.shape)}"
            )
        if add_noise:
            actor_obs += (
                (2 * torch.rand_like(actor_obs) - 1)
                * self.noise_scale_vec[: actor_obs.shape[1]]
            )

        current_obs = torch.cat(
            (
                actor_obs,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.disturbance[:, 0, :],
            ),
            dim=-1,
        )
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1,
            ) * self.obs_scales.height_measurements
            if add_noise:
                heights += (
                    (2 * torch.rand_like(heights) - 1)
                    * self.noise_scale_vec[actor_obs.shape[1] :]
                )
            current_obs = torch.cat((current_obs, heights), dim=-1)

        force_scale, force_shift = get_scale_shift(
            self.cfg.normalization.contact_force_range
        )
        contact_forces = (
            self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
            - force_shift
        ) * force_scale
        privileged_obs = torch.cat((current_obs, contact_forces), dim=-1)
        if privileged_obs.shape != (
            self.num_envs,
            self.num_one_step_privileged_obs,
        ):
            raise RuntimeError(
                "privileged observation shape mismatch: "
                f"got {list(privileged_obs.shape)}"
            )
        return privileged_obs

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        primary_reward_names = self.TRACK_REWARD_NAMES | self.HOLD_REWARD_NAMES
        positive_auxiliary = [
            name
            for name in self.reward_names
            if self.reward_scales[name] > 0
            and name not in primary_reward_names
        ]
        if positive_auxiliary:
            raise ValueError(
                "positive dance rewards must be declared as task terms: "
                + ", ".join(positive_auxiliary)
            )
        configured = set(self.reward_names)
        classified = (
            self.TRACK_REWARD_NAMES
            | self.HOLD_REWARD_NAMES
            | self.SOFT_REWARD_NAMES
            | self.HARD_REWARD_NAMES
        )
        unclassified = configured - classified
        if unclassified:
            raise ValueError(
                "dance rewards must be explicitly classified: "
                + ", ".join(sorted(unclassified))
            )
        if float(self.cfg.rewards.hold_gate_sigma) <= 0.0:
            raise ValueError("hold_gate_sigma must be positive")
        auxiliary_floor = float(self.cfg.rewards.auxiliary_reward_floor)
        if not 0.0 <= auxiliary_floor <= 1.0:
            raise ValueError("auxiliary_reward_floor must be in [0, 1]")
        for name in (
            "command_tracking",
            "stationary_hold",
            "hold_gate",
            "aux_multiplier",
            "hard_safety",
            "final_task",
        ):
            self.episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )

    def compute_reward(self):
        """Task * soft multiplier + hard safety + terminal penalty."""
        tracking_cost = torch.zeros_like(self.rew_buf)
        hold_cost = torch.zeros_like(self.rew_buf)
        soft_reward = torch.zeros_like(self.rew_buf)
        hard_safety_penalty = torch.zeros_like(self.rew_buf)
        tracking_weights = self.cfg.rewards.tracking_weights
        hold_weights = self.cfg.rewards.hold_weights
        tracking_weight_sum = sum(float(v) for v in tracking_weights.values())
        hold_weight_sum = sum(float(v) for v in hold_weights.values())

        for name, reward_function in zip(
            self.reward_names, self.reward_functions
        ):
            if name in self.TRACK_REWARD_NAMES:
                weight = float(tracking_weights[name])
                cost = self._tracking_cost(name)
                tracking_cost += weight * cost
                self.episode_sums[name] += torch.exp(-cost) * self.dt
            elif name in self.HOLD_REWARD_NAMES:
                weight = float(hold_weights[name])
                cost = self._hold_cost(name)
                hold_cost += weight * cost
                self.episode_sums[name] += torch.exp(-cost) * self.dt
            else:
                raw_reward = reward_function()
                scaled_reward = raw_reward * self.reward_scales[name]
                self.episode_sums[name] += scaled_reward
                if name in self.SOFT_REWARD_NAMES:
                    soft_reward += scaled_reward
                elif name in self.HARD_REWARD_NAMES:
                    hard_safety_penalty += scaled_reward

        if tracking_weight_sum <= 0.0 or hold_weight_sum <= 0.0:
            raise RuntimeError("dance task requires tracking and hold rewards")

        tracking_cost /= tracking_weight_sum
        hold_cost /= hold_weight_sum
        tracking_score = torch.exp(-tracking_cost)
        hold_score = torch.exp(-hold_cost)
        hold_gate = torch.exp(
            -tracking_cost / float(self.cfg.rewards.hold_gate_sigma)
        )
        task_score = tracking_score * (
            (1.0 - hold_gate) + hold_gate * hold_score
        )
        auxiliary_multiplier = floored_soft_multiplier(
            soft_reward,
            float(self.cfg.rewards.auxiliary_reward_sigma),
            float(self.cfg.rewards.auxiliary_reward_floor),
        )
        task_reward = (
            float(self.cfg.rewards.task_reward_scale) * self.dt * task_score
        )
        self.rew_buf[:] = (
            task_reward * auxiliary_multiplier + hard_safety_penalty
        )
        self.episode_sums["command_tracking"] += tracking_score * self.dt
        self.episode_sums["stationary_hold"] += hold_score * self.dt
        self.episode_sums["hold_gate"] += hold_gate * self.dt
        self.episode_sums["aux_multiplier"] += auxiliary_multiplier * self.dt
        self.episode_sums["hard_safety"] += hard_safety_penalty
        self.episode_sums["final_task"] += self.rew_buf

        # Keep terminal failure outside the multiplier, as in the base task.
        if "termination" in self.reward_scales:
            termination_reward = (
                self._reward_termination()
                * self.reward_scales["termination"]
            )
            self.rew_buf += termination_reward
            self.episode_sums["termination"] += termination_reward

    def compute_observations(self):
        current_obs = self._build_current_observation(self.add_noise)
        self.obs_buf = torch.cat(
            (
                current_obs[:, : self.num_one_step_obs],
                self.obs_buf[:, : -self.num_one_step_obs],
            ),
            dim=-1,
        )
        if self.obs_buf.shape != (self.num_envs, 360):
            raise RuntimeError(f"history observation must be [N, 360], got {list(self.obs_buf.shape)}")
        self.privileged_obs_buf = current_obs[:, : self.num_one_step_privileged_obs]

    def get_current_obs(self):
        return self._build_current_observation(self.add_noise)

    def compute_termination_observations(self, env_ids):
        current_obs = self._build_current_observation(self.add_noise)
        return current_obs[env_ids, : self.num_one_step_privileged_obs]

    def _compute_torques(self, actions):
        actions = effective_actions(actions, self.wheel_indices)
        torques = super()._compute_torques(actions)
        # Position-hold the wheel axle at its reset angle. Damping alone only
        # suppresses speed and still allows slow rolling during a body-yaw pose.
        wheel_error = (
            self.wheel_park_targets - self.dof_pos[:, self.wheel_indices]
        )
        if self.Kd_factors.ndim != 2 or self.Kd_factors.shape[0] != self.num_envs:
            raise RuntimeError("Kd_factors must have shape [num_envs, 1|num_dof]")
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
        if self.motor_strength_factors.ndim != 2 or self.motor_strength_factors.shape[
            0
        ] != self.num_envs or self.motor_strength_factors.shape[1] not in (
            1,
            self.num_dof,
        ):
            raise RuntimeError(
                "motor_strength_factors must have shape [num_envs, 1|num_dof]"
            )
        torques *= self.motor_strength_factors
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        super().check_termination()
        max_tilt = float(self.cfg.rewards.termination_tilt)
        excessive_tilt = self.projected_gravity[:, 2] > -math.cos(max_tilt)
        too_low = self._current_base_height() < self.cfg.rewards.termination_min_height
        self.reset_buf |= excessive_tilt | too_low

    def _current_roll_pitch(self):
        gravity = self.projected_gravity
        pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        return roll, pitch

    def _target_projected_gravity(self):
        """Gravity vector expected in the body frame at commanded roll/pitch."""
        return target_projected_gravity(
            self.commands[:, self.ROLL_COMMAND],
            self.commands[:, self.PITCH_COMMAND],
        )

    def _current_yaw(self):
        quat = self.root_states[:, 3:7]
        qx, qy, qz, qw = quat.unbind(dim=1)
        return torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def _current_relative_yaw(self):
        delta = self._current_yaw() - self.episode_start_yaw
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _body_yaw_error(self):
        error = self.commands[:, 2] - self._current_relative_yaw()
        return torch.atan2(torch.sin(error), torch.cos(error))

    def _current_base_height(self):
        if not self.cfg.terrain.measure_heights:
            return self.root_states[:, 2]
        if self.measured_heights is None:
            raise RuntimeError("measure_heights=True but measured_heights is unavailable")
        return torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    @staticmethod
    def _masked_mean(values, mask):
        """Mean over an active command subset, safe when a batch has none."""
        return torch.sum(values * mask) / torch.clamp(torch.sum(mask), min=1.0)

    def _tracking_cost(self, name):
        """Direct normalized squared costs; no log(exp()) reconstruction."""
        if name == "tracking_body_orientation":
            error = torch.sum(
                torch.square(
                    self.projected_gravity[:, :2]
                    - self._target_projected_gravity()[:, :2]
                ),
                dim=1,
            )
            return error / float(self.cfg.rewards.orientation_tracking_sigma)
        if name == "tracking_body_yaw":
            return torch.square(self._body_yaw_error()) / float(
                self.cfg.rewards.yaw_tracking_sigma
            )
        if name == "tracking_body_height":
            error = self._current_base_height() - self.commands[:, self.HEIGHT_COMMAND]
            return torch.square(error) / float(
                self.cfg.rewards.height_tracking_sigma
            )
        raise KeyError(name)

    def _hold_cost(self, name):
        if name == "tracking_lin_vx":
            return torch.square(self.base_lin_vel[:, 0]) / float(
                self.cfg.rewards.tracking_sigma
            )
        if name == "tracking_lin_vy":
            return torch.square(self.base_lin_vel[:, 1]) / float(
                self.cfg.rewards.tracking_sigma
            )
        if name == "tracking_support_position":
            return torch.square(self._support_anchor_excess()) / float(
                self.cfg.rewards.support_position_tracking_sigma
            )
        if name == "tracking_feet_position":
            return self._reward_feet_position_drift() / float(
                self.cfg.rewards.feet_position_tracking_sigma
            )
        if name == "tracking_max_foot_position":
            return self._reward_max_foot_position_drift() / float(
                self.cfg.rewards.max_foot_position_tracking_sigma
            )
        raise KeyError(name)

    def _reward_tracking_body_orientation(self):
        error = torch.sum(
            torch.square(
                self.projected_gravity[:, :2]
                - self._target_projected_gravity()[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-error / self.cfg.rewards.orientation_tracking_sigma)

    def _reward_tracking_body_yaw(self):
        error = torch.square(self._body_yaw_error())
        return torch.exp(-error / self.cfg.rewards.yaw_tracking_sigma)

    def _reward_tracking_body_height(self):
        error = torch.square(
            self._current_base_height() - self.commands[:, self.HEIGHT_COMMAND]
        )
        return torch.exp(-error / self.cfg.rewards.height_tracking_sigma)

    def _reward_support_center_drift(self):
        return torch.square(self._support_anchor_excess())

    def _reward_tracking_support_position(self):
        """Primary stationary reference; tighter than individual wheel anchors."""
        error = torch.square(self._support_anchor_excess())
        return torch.exp(
            -error / self.cfg.rewards.support_position_tracking_sigma
        )

    def _support_anchor_excess(self):
        support_xy = torch.mean(self.feet_pos[:, :, :2], dim=1)
        drift = torch.norm(
            support_xy - self.episode_start_support_xy, dim=1
        )
        return torch.clamp(
            drift - float(self.cfg.rewards.support_anchor_deadzone), min=0.0
        )

    def _feet_anchor_excess(self):
        drift = torch.norm(
            self.feet_pos[:, :, :2] - self.episode_start_feet_xy, dim=2
        )
        return torch.clamp(
            drift - float(self.cfg.rewards.feet_anchor_deadzone), min=0.0
        )

    def _reward_feet_position_drift(self):
        """Penalize meaningful support movement, not contact jitter."""
        return torch.sum(torch.square(self._feet_anchor_excess()), dim=1)

    def _reward_tracking_feet_position(self):
        """Dense bonus for keeping all four wheel centers at their reset points."""
        error = self._reward_feet_position_drift()
        return torch.exp(
            -error / self.cfg.rewards.feet_position_tracking_sigma
        )

    def _reward_tracking_max_foot_position(self):
        """Do not let one drifting wheel hide behind the four-wheel average."""
        error = self._reward_max_foot_position_drift()
        return torch.exp(
            -error / self.cfg.rewards.max_foot_position_tracking_sigma
        )

    def _reward_max_foot_position_drift(self):
        """Prevent one wheel from moving while the other three hide the average."""
        drift_sq = torch.square(self._feet_anchor_excess())
        return torch.max(drift_sq, dim=1).values

    def _reward_yaw_rate(self):
        """Settle at the requested body-yaw angle instead of continuously turning."""
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_feet_vertical_motion(self):
        """Suppress wheel lift and vertical chatter during body poses."""
        contact = (
            self.contact_forces[:, self.feet_indices, 2] > 5.0
        ).float()
        return torch.sum(
            contact * torch.square(self.feet_vel[:, :, 2]), dim=1
        )

    def _reward_neutral_joint_pose(self):
        """Keep a symmetric nominal stance only near the neutral pose command."""
        neutral_weight = self._neutral_pose_weight()

        joint_error = self.dof_pos - self.default_dof_pos
        joint_error = joint_error.clone()
        joint_error[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(joint_error), dim=1) * neutral_weight

    def _neutral_pose_weight(self):
        """Continuous gate based on filtered commands, never raw targets."""
        normalized = self.commands.clone()
        normalized[:, 2] = self.commands[:, 2]
        normalized = normalize_dance_commands(
            normalized,
            default_height=float(self.cfg.rewards.default_body_height),
            min_height=float(self.cfg.commands.min_height),
            max_abs_yaw=float(self.cfg.commands.max_abs_yaw),
            max_abs_roll=float(self.cfg.commands.max_abs_roll),
            max_abs_pitch=float(self.cfg.commands.max_abs_pitch),
        )
        return neutral_command_weight(
            normalized, float(self.cfg.rewards.neutral_command_sigma)
        )

    def _reward_severe_wheel_park(self):
        wheel_error = torch.abs(
            self.wheel_park_targets - self.dof_pos[:, self.wheel_indices]
        )
        return torch.any(
            wheel_error > float(self.cfg.rewards.severe_wheel_park_error), dim=1
        ).float()

    def _reward_severe_support_loss(self):
        support_xy = torch.mean(self.feet_pos[:, :, :2], dim=1)
        drift = torch.norm(support_xy - self.episode_start_support_xy, dim=1)
        return (
            drift > float(self.cfg.rewards.severe_support_loss_distance)
        ).float()

    def _reward_torque_limits(self):
        """Penalize saturation; post-clip excess is otherwise identically zero."""
        return torch.sum(
            (
                torch.abs(self.torques) >= 0.99 * self.torque_limits
            ).float(),
            dim=1,
        )
