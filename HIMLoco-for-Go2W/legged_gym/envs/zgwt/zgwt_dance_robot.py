import math

import torch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.utils.math import get_scale_shift

from .zgwt_robot import Zgwt
from .zgwt_dance_config import ZGWTDanceCfg


class ZgwtDance(Zgwt):
    """Stationary ZGWT task tracking roll, pitch, and body-height commands."""

    cfg: ZGWTDanceCfg

    ROLL_COMMAND = 3
    PITCH_COMMAND = 4
    HEIGHT_COMMAND = 5

    def _init_buffers(self):
        super()._init_buffers()
        self.commands_scale = torch.tensor(
            self.cfg.commands.command_scales,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.pose_command_targets = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.commands[:, self.HEIGHT_COMMAND] = self.cfg.rewards.default_body_height
        self.pose_command_targets[:, 2] = self.cfg.rewards.default_body_height
        self.episode_start_xy = self.root_states[:, :2].clone()
        self.episode_start_support_xy = torch.mean(
            self.feet_pos[:, :, :2], dim=1
        ).clone()
        self.support_anchor_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._resampling_for_reset = False

        pair_a_names = []
        pair_b_names = []
        for pair_a, pair_b in (("FAR", "FBL"), ("RAR", "RBL")):
            for joint in ("ABAD", "HIP", "KNEE"):
                pair_a_names.append(f"{pair_a}_{joint}_JOINT")
                pair_b_names.append(f"{pair_b}_{joint}_JOINT")
        self.neutral_pair_a_indices = torch.tensor(
            [self.dof_names.index(name) for name in pair_a_names],
            dtype=torch.long,
            device=self.device,
        )
        self.neutral_pair_b_indices = torch.tensor(
            [self.dof_names.index(name) for name in pair_b_names],
            dtype=torch.long,
            device=self.device,
        )

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
        roll_error = torch.mean(
            torch.abs(self.commands[env_ids, self.ROLL_COMMAND] - roll[env_ids])
        )
        pitch_error = torch.mean(
            torch.abs(self.commands[env_ids, self.PITCH_COMMAND] - pitch[env_ids])
        )
        height_error = torch.mean(
            torch.abs(self.commands[env_ids, self.HEIGHT_COMMAND] - height[env_ids])
        )
        xy_drift = torch.mean(
            torch.norm(
                self.root_states[env_ids, :2] - self.episode_start_xy[env_ids], dim=1
            )
        )
        support_xy = torch.mean(self.feet_pos[env_ids, :, :2], dim=1)
        support_xy_drift = torch.mean(
            torch.norm(
                support_xy - self.episode_start_support_xy[env_ids], dim=1
            )
        )

        self._resampling_for_reset = True
        super().reset_idx(env_ids)
        self._resampling_for_reset = False
        self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        # Rigid-body state is refreshed on the next physics step after reset.
        self.support_anchor_pending[env_ids] = True

        # Dance-specific diagnostics for TensorBoard.
        self.extras["episode"]["mean_abs_roll_error"] = roll_error
        self.extras["episode"]["mean_abs_pitch_error"] = pitch_error
        self.extras["episode"]["mean_abs_height_error"] = height_error
        self.extras["episode"]["mean_xy_drift"] = xy_drift
        self.extras["episode"]["mean_support_xy_drift"] = support_xy_drift

    def _command_curriculum_fraction(self):
        duration = max(float(self.cfg.commands.curriculum_time), self.dt)
        return min(self.common_step_counter * self.dt / duration, 1.0)

    def _interpolated_range(self, name):
        initial = getattr(self.cfg.commands.initial_ranges, name)
        final = self.command_ranges[name]
        fraction = self._command_curriculum_fraction()
        low = initial[0] + fraction * (final[0] - initial[0])
        high = initial[1] + fraction * (final[1] - initial[1])
        return low, high

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        roll_range = self._interpolated_range("body_roll")
        pitch_range = self._interpolated_range("body_pitch")
        height_range = self._interpolated_range("body_height")
        count = len(env_ids)

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
        mode = torch.randint(0, 5, (count,), device=self.device)
        neutral_mask = torch.rand(count, device=self.device) < float(
            self.cfg.commands.neutral_pose_prob
        )
        active_mask = ~neutral_mask
        targets = torch.zeros(count, 3, dtype=torch.float, device=self.device)
        targets[:, 2] = self.cfg.rewards.default_body_height
        height_only = (mode == 0) & active_mask
        roll_only = (mode == 1) & active_mask
        pitch_only = (mode == 2) & active_mask
        targets[height_only, 2] = sampled_height[height_only]
        targets[roll_only, 0] = sampled_roll[roll_only]
        targets[pitch_only, 1] = sampled_pitch[pitch_only]
        two_axis = (mode == 3) & active_mask
        targets[two_axis, 0] = sampled_roll[two_axis]
        targets[two_axis, 1] = sampled_pitch[two_axis]
        combined = (mode == 4) & active_mask
        targets[combined, 0] = sampled_roll[combined]
        targets[combined, 1] = sampled_pitch[combined]
        targets[combined, 2] = sampled_height[combined]

        self.pose_command_targets[env_ids] = targets
        self.commands[env_ids, :3] = 0.0
        if self._resampling_for_reset:
            # Start every reset from the nominal pose and then ramp to the target.
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
            self.support_anchor_pending[pending_ids] = False

        resample_steps = max(
            1, int(self.cfg.commands.resampling_time / self.dt)
        )
        env_ids = (self.episode_length_buf % resample_steps == 0).nonzero(
            as_tuple=False
        ).flatten()
        self._resample_commands(env_ids)

        transition_time = max(float(self.cfg.commands.transition_time), self.dt)
        alpha = min(self.dt / transition_time, 1.0)
        pose_commands = self.commands[
            :, self.ROLL_COMMAND : self.HEIGHT_COMMAND + 1
        ]
        pose_commands[:] = torch.lerp(pose_commands, self.pose_command_targets, alpha)
        self.commands[:, :3] = 0.0

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

        actor_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands * self.commands_scale,
                dof_err * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
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
        return torch.cat((current_obs, contact_forces), dim=-1)

    def compute_observations(self):
        current_obs = self._build_current_observation(self.add_noise)
        self.obs_buf = torch.cat(
            (
                current_obs[:, : self.num_one_step_obs],
                self.obs_buf[:, : -self.num_one_step_obs],
            ),
            dim=-1,
        )
        self.privileged_obs_buf = current_obs[:, : self.num_one_step_privileged_obs]

    def get_current_obs(self):
        return self._build_current_observation(self.add_noise)

    def compute_termination_observations(self, env_ids):
        current_obs = self._build_current_observation(self.add_noise)
        return current_obs[env_ids, : self.num_one_step_privileged_obs]

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        # Hard park the wheels for the first dance task. The policy still sees
        # wheel states/actions, but cannot translate the robot by driving them.
        torques[:, self.wheel_indices] = (
            -float(self.cfg.control.wheel_park_damping)
            * self.Kd_factors
            * self.dof_vel[:, self.wheel_indices]
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        super().check_termination()
        max_tilt = float(self.cfg.rewards.termination_tilt)
        excessive_tilt = self.projected_gravity[:, 2] > -math.cos(max_tilt)
        too_low = self._current_base_height() < self.cfg.rewards.termination_min_height
        self.reset_buf |= excessive_tilt | too_low

    def _target_projected_gravity(self):
        roll = self.commands[:, self.ROLL_COMMAND]
        pitch = self.commands[:, self.PITCH_COMMAND]
        cos_pitch = torch.cos(pitch)
        return torch.stack(
            (
                torch.sin(pitch),
                -torch.sin(roll) * cos_pitch,
                -torch.cos(roll) * cos_pitch,
            ),
            dim=1,
        )

    def _current_roll_pitch(self):
        gravity = self.projected_gravity
        pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        return roll, pitch

    def _current_base_height(self):
        return torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _reward_tracking_body_orientation(self):
        error = torch.sum(
            torch.square(self.projected_gravity - self._target_projected_gravity()),
            dim=1,
        )
        return torch.exp(-error / self.cfg.rewards.orientation_tracking_sigma)

    def _reward_tracking_body_height(self):
        error = torch.square(
            self._current_base_height() - self.commands[:, self.HEIGHT_COMMAND]
        )
        return torch.exp(-error / self.cfg.rewards.height_tracking_sigma)

    def _reward_base_position_drift(self):
        return torch.sum(
            torch.square(self.root_states[:, :2] - self.episode_start_xy), dim=1
        )

    def _reward_support_center_drift(self):
        support_xy = torch.mean(self.feet_pos[:, :, :2], dim=1)
        return torch.sum(
            torch.square(support_xy - self.episode_start_support_xy), dim=1
        )

    def _reward_feet_horizontal_motion(self):
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.0).float()
        horizontal_speed_sq = torch.sum(
            torch.square(self.feet_vel[:, :, :2]), dim=2
        )
        return torch.sum(horizontal_speed_sq * contact, dim=1)

    def _reward_neutral_joint_pose(self):
        """Keep a symmetric nominal stance only near the neutral pose command."""
        neutral_weight = self._neutral_pose_weight()

        joint_error = self.dof_pos - self.default_dof_pos
        joint_error = joint_error.clone()
        joint_error[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(joint_error), dim=1) * neutral_weight

    def _reward_lateral_leg_symmetry(self):
        """Limit unnecessary left/right joint asymmetry for every pose command."""
        joint_error = self.dof_pos - self.default_dof_pos
        pair_error = (
            joint_error[:, self.neutral_pair_a_indices]
            - joint_error[:, self.neutral_pair_b_indices]
        )
        allowed_asymmetry = (
            torch.abs(self.commands[:, self.ROLL_COMMAND]).unsqueeze(1)
            * self.cfg.rewards.lateral_symmetry_roll_allowance
        )
        excess_asymmetry = torch.clamp(
            torch.abs(pair_error) - allowed_asymmetry, min=0.0
        )
        return torch.sum(excess_asymmetry, dim=1)

    def _reward_lateral_foot_alignment(self):
        """Keep each left/right wheel pair aligned in the body x direction."""
        front_delta = self.feet_pos[:, 0, :] - self.feet_pos[:, 1, :]
        rear_delta = self.feet_pos[:, 2, :] - self.feet_pos[:, 3, :]
        front_delta_body = quat_rotate_inverse(self.base_quat, front_delta)
        rear_delta_body = quat_rotate_inverse(self.base_quat, rear_delta)
        return torch.abs(front_delta_body[:, 0]) + torch.abs(
            rear_delta_body[:, 0]
        )

    def _neutral_pose_weight(self):
        """Return a smooth gate that disables stance rewards during dance."""
        pose_command_error = torch.square(
            self.commands[:, self.ROLL_COMMAND]
        ) + torch.square(self.commands[:, self.PITCH_COMMAND])
        height_command_error = torch.square(
            self.commands[:, self.HEIGHT_COMMAND]
            - self.cfg.rewards.default_body_height
        )
        return torch.exp(
            -pose_command_error / self.cfg.rewards.neutral_orientation_sigma
            -height_command_error / self.cfg.rewards.neutral_height_sigma
        )
