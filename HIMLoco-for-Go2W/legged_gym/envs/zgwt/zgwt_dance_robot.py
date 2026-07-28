import math

import torch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.utils.math import get_scale_shift

from .zgwt_robot import Zgwt
from .zgwt_dance_config import ZGWTDanceCfg


class ZgwtDance(Zgwt):
    """Fixed-support task tracking bounded body yaw, roll, pitch, and height."""

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
        leg_order = ("FAR", "FBL", "RAR", "RBL")
        self.leg_abad_indices = torch.tensor(
            [self.dof_names.index(f"{leg}_ABAD_JOINT") for leg in leg_order],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_hip_indices = torch.tensor(
            [self.dof_names.index(f"{leg}_HIP_JOINT") for leg in leg_order],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_knee_indices = torch.tensor(
            [self.dof_names.index(f"{leg}_KNEE_JOINT") for leg in leg_order],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_joint_indices = torch.cat(
            (self.leg_abad_indices, self.leg_hip_indices, self.leg_knee_indices)
        )
        # Convert front/rear joint signs into one common positive crouch axis.
        self.crouch_hip_sign = torch.tensor(
            [-1.0, -1.0, 1.0, 1.0], device=self.device
        )
        self.crouch_knee_sign = torch.tensor(
            [1.0, 1.0, -1.0, -1.0], device=self.device
        )
        mode_probabilities = torch.tensor(
            self.cfg.commands.mode_probabilities,
            dtype=torch.float,
            device=self.device,
        )
        if mode_probabilities.numel() != 8 or torch.any(mode_probabilities < 0):
            raise ValueError("dance mode_probabilities must contain 8 non-negative values")
        probability_sum = torch.sum(mode_probabilities)
        if probability_sum <= 0:
            raise ValueError("dance mode_probabilities must have a positive sum")
        self.command_mode_probabilities = mode_probabilities / probability_sum

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
        roll_error = torch.mean(roll_abs_error)
        pitch_error = torch.mean(pitch_abs_error)
        roll_active = (
            torch.abs(self.pose_command_targets[env_ids, 0])
            >= self.cfg.rewards.active_orientation_threshold
        ).float()
        pitch_active = (
            torch.abs(self.pose_command_targets[env_ids, 1])
            >= self.cfg.rewards.active_orientation_threshold
        ).float()
        roll_active_error = self._masked_mean(roll_abs_error, roll_active)
        pitch_active_error = self._masked_mean(pitch_abs_error, pitch_active)
        height_abs_error = torch.abs(
            self.commands[env_ids, self.HEIGHT_COMMAND] - height[env_ids]
        )
        height_error = torch.mean(height_abs_error)
        height_active = (
            torch.abs(
                self.pose_command_targets[env_ids, 2]
                - self.cfg.rewards.default_body_height
            )
            >= self.cfg.rewards.active_height_threshold
        ).float()
        height_active_error = self._masked_mean(height_abs_error, height_active)
        # default_dof_pos has shape [1, num_dof] and is shared by every env;
        # indexing it with env_ids > 0 triggers a delayed CUDA device assert.
        joint_error = self.dof_pos[env_ids] - self.default_dof_pos
        lateral_pair_error = torch.mean(
            torch.abs(
                joint_error[:, self.neutral_pair_a_indices]
                - joint_error[:, self.neutral_pair_b_indices]
            )
        )
        neutral_weight = self._neutral_pose_weight()[env_ids]
        neutral_abad_error = torch.sum(
            torch.mean(
                torch.abs(joint_error[:, self.leg_abad_indices]), dim=1
            )
            * neutral_weight
        ) / torch.clamp(torch.sum(neutral_weight), min=1.0)
        hip_flexion = (
            joint_error[:, self.leg_hip_indices] * self.crouch_hip_sign
        )
        knee_flexion = (
            joint_error[:, self.leg_knee_indices] * self.crouch_knee_sign
        )
        default_height_weight = torch.exp(
            -torch.square(
                self.commands[env_ids, self.HEIGHT_COMMAND]
                - self.cfg.rewards.default_body_height
            )
            / self.cfg.rewards.neutral_height_sigma
        )
        default_height_count = torch.clamp(
            torch.sum(default_height_weight), min=1.0
        )
        common_hip_flexion = torch.sum(
            torch.abs(torch.mean(hip_flexion, dim=1)) * default_height_weight
        ) / default_height_count
        common_knee_flexion = torch.sum(
            torch.abs(torch.mean(knee_flexion, dim=1)) * default_height_weight
        ) / default_height_count
        yaw_abs_error = torch.abs(self._body_yaw_error()[env_ids])
        yaw_error = torch.mean(yaw_abs_error)
        yaw_command_active = (
            torch.abs(self.yaw_command_targets[env_ids])
            >= self.cfg.rewards.active_yaw_threshold
        ).float()
        yaw_active_error = self._masked_mean(yaw_abs_error, yaw_command_active)
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
        feet_xy_drift = torch.mean(
            torch.norm(
                self.feet_pos[env_ids, :, :2]
                - self.episode_start_feet_xy[env_ids],
                dim=2,
            )
        )
        yaw_active = (
            torch.abs(self.commands[env_ids, 2])
            >= self.cfg.rewards.body_yaw_in_place_full_scale
        ).float()
        yaw_sample_count = torch.clamp(torch.sum(yaw_active), min=1.0)
        yaw_xy_drift = torch.sum(
            torch.norm(
                self.root_states[env_ids, :2] - self.episode_start_xy[env_ids],
                dim=1,
            )
            * yaw_active
        ) / yaw_sample_count
        yaw_forward_speed = torch.sum(
            torch.abs(self.base_lin_vel[env_ids, 0]) * yaw_active
        ) / yaw_sample_count

        self._resampling_for_reset = True
        super().reset_idx(env_ids)
        self._resampling_for_reset = False
        self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.episode_start_yaw[env_ids] = self._current_yaw()[env_ids]
        self.wheel_park_targets[env_ids] = self.dof_pos[
            env_ids[:, None], self.wheel_indices
        ]
        # Rigid-body state is refreshed on the next physics step after reset.
        self.support_anchor_pending[env_ids] = True

        # Dance-specific diagnostics for TensorBoard.
        self.extras["episode"]["mean_abs_roll_error"] = roll_error
        self.extras["episode"]["mean_abs_pitch_error"] = pitch_error
        self.extras["episode"]["mean_abs_roll_error_active"] = roll_active_error
        self.extras["episode"]["mean_abs_pitch_error_active"] = (
            pitch_active_error
        )
        self.extras["episode"]["mean_roll_active_fraction"] = torch.mean(
            roll_active
        )
        self.extras["episode"]["mean_pitch_active_fraction"] = torch.mean(
            pitch_active
        )
        self.extras["episode"]["mean_abs_height_error"] = height_error
        self.extras["episode"]["mean_abs_height_error_active"] = (
            height_active_error
        )
        self.extras["episode"]["mean_height_active_fraction"] = torch.mean(
            height_active
        )
        self.extras["episode"]["mean_lateral_pair_error"] = lateral_pair_error
        self.extras["episode"]["mean_neutral_abad_error"] = neutral_abad_error
        self.extras["episode"]["mean_default_height_common_hip_flexion"] = (
            common_hip_flexion
        )
        self.extras["episode"]["mean_default_height_common_knee_flexion"] = (
            common_knee_flexion
        )
        # Override the base class' yaw-rate diagnostic: slot 2 is a body-yaw angle.
        self.extras["episode"]["mean_abs_yaw_error"] = yaw_error
        self.extras["episode"]["mean_abs_yaw_error_active"] = yaw_active_error
        self.extras["episode"]["mean_yaw_active_fraction"] = torch.mean(
            yaw_command_active
        )
        self.extras["episode"]["mean_xy_drift"] = xy_drift
        self.extras["episode"]["mean_support_xy_drift"] = support_xy_drift
        self.extras["episode"]["mean_feet_xy_drift"] = feet_xy_drift
        self.extras["episode"]["mean_yaw_xy_drift"] = yaw_xy_drift
        self.extras["episode"]["mean_abs_yaw_lin_vel_x"] = yaw_forward_speed

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

        yaw_range = self._interpolated_range("body_yaw")
        roll_range = self._interpolated_range("body_roll")
        pitch_range = self._interpolated_range("body_pitch")
        height_range = self._interpolated_range("body_height")
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
        yaw_only = mode == 1
        height_only = mode == 2
        roll_only = mode == 3
        pitch_only = mode == 4
        yaw_targets[yaw_only] = sampled_yaw[yaw_only]
        targets[height_only, 2] = sampled_height[height_only]
        targets[roll_only, 0] = sampled_roll[roll_only]
        targets[pitch_only, 1] = sampled_pitch[pitch_only]
        two_axis = mode == 5
        targets[two_axis, 0] = sampled_roll[two_axis]
        targets[two_axis, 1] = sampled_pitch[two_axis]
        pose_combined = mode == 6
        targets[pose_combined, 0] = sampled_roll[pose_combined]
        targets[pose_combined, 1] = sampled_pitch[pose_combined]
        targets[pose_combined, 2] = sampled_height[pose_combined]
        combined = mode == 7
        yaw_targets[combined] = sampled_yaw[combined]
        targets[combined, 0] = sampled_roll[combined]
        targets[combined, 1] = sampled_pitch[combined]
        targets[combined, 2] = sampled_height[combined]

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

        transition_time = max(float(self.cfg.commands.transition_time), self.dt)
        alpha = min(self.dt / transition_time, 1.0)
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

        actor_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                observed_commands * self.commands_scale,
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
        # Position-hold the wheel axle at its reset angle. Damping alone only
        # suppresses speed and still allows slow rolling during a body-yaw pose.
        wheel_error = (
            self.wheel_park_targets - self.dof_pos[:, self.wheel_indices]
        )
        torques[:, self.wheel_indices] = (
            float(self.cfg.control.wheel_park_stiffness) * wheel_error
            - float(self.cfg.control.wheel_park_damping)
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
        return torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    @staticmethod
    def _masked_mean(values, mask):
        """Mean over an active command subset, safe when a batch has none."""
        return torch.sum(values * mask) / torch.clamp(torch.sum(mask), min=1.0)

    def _command_tracking_weight(self, target_magnitude, full_scale):
        """Prioritize active commands while weakly holding inactive axes."""
        active_weight = torch.clamp(
            torch.abs(target_magnitude) / float(full_scale), min=0.0, max=1.0
        )
        inactive_weight = float(
            self.cfg.rewards.inactive_command_tracking_weight
        )
        return inactive_weight + (1.0 - inactive_weight) * active_weight

    def _reward_tracking_body_orientation(self):
        error = torch.sum(
            torch.square(self.projected_gravity - self._target_projected_gravity()),
            dim=1,
        )
        return torch.exp(-error / self.cfg.rewards.orientation_tracking_sigma)

    def _reward_tracking_body_roll(self):
        roll, _ = self._current_roll_pitch()
        error = torch.square(
            roll - self.commands[:, self.ROLL_COMMAND]
        )
        weight = self._command_tracking_weight(
            self.pose_command_targets[:, 0],
            self.cfg.rewards.roll_tracking_full_scale,
        )
        return weight * torch.exp(
            -error / self.cfg.rewards.roll_tracking_sigma
        )

    def _reward_tracking_body_pitch(self):
        _, pitch = self._current_roll_pitch()
        error = torch.square(
            pitch - self.commands[:, self.PITCH_COMMAND]
        )
        weight = self._command_tracking_weight(
            self.pose_command_targets[:, 1],
            self.cfg.rewards.pitch_tracking_full_scale,
        )
        return weight * torch.exp(
            -error / self.cfg.rewards.pitch_tracking_sigma
        )

    def _reward_tracking_body_yaw(self):
        error = torch.square(self._body_yaw_error())
        weight = self._command_tracking_weight(
            self.yaw_command_targets,
            self.cfg.rewards.yaw_tracking_full_scale,
        )
        return weight * torch.exp(-error / self.cfg.rewards.yaw_tracking_sigma)

    def _reward_tracking_body_height(self):
        error = torch.square(
            self._current_base_height() - self.commands[:, self.HEIGHT_COMMAND]
        )
        height_target_offset = (
            self.pose_command_targets[:, 2]
            - self.cfg.rewards.default_body_height
        )
        weight = self._command_tracking_weight(
            height_target_offset,
            self.cfg.rewards.height_tracking_full_scale,
        )
        return weight * torch.exp(
            -error / self.cfg.rewards.height_tracking_sigma
        )

    def _reward_base_position_drift(self):
        return torch.sum(
            torch.square(self.root_states[:, :2] - self.episode_start_xy), dim=1
        )

    def _reward_support_center_drift(self):
        return torch.square(self._support_anchor_excess())

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

    def _reward_max_foot_position_drift(self):
        """Prevent one wheel from moving while the other three hide the average."""
        drift_sq = torch.square(self._feet_anchor_excess())
        return torch.max(drift_sq, dim=1).values

    def _reward_base_linear_motion(self):
        """Suppress fore/aft and lateral pacing during fixed-support poses."""
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

    def _reward_body_yaw_in_place(self):
        """Keep the base and support points fixed during body-yaw poses."""
        yaw_weight = torch.clamp(
            torch.abs(self.commands[:, 2])
            / self.cfg.rewards.body_yaw_in_place_full_scale,
            max=1.0,
        )
        position_error = torch.sum(
            torch.square(self.root_states[:, :2] - self.episode_start_xy), dim=1
        )
        support_error = torch.square(self._support_anchor_excess())
        planar_speed = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        return yaw_weight * (
            position_error + support_error + 0.25 * planar_speed
        )

    def _reward_yaw_rate(self):
        """Settle at the requested body-yaw angle instead of continuously turning."""
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_wheel_stand_still(self):
        """Wheel actions are invalid for every command in fixed-support dance."""
        return torch.sum(
            torch.square(self.actions[:, self.wheel_indices]), dim=1
        )

    def _reward_wheel_vel_stand_still(self):
        """Wheel speed must remain zero during roll, pitch, height, and yaw poses."""
        return torch.sum(
            torch.square(self.dof_vel[:, self.wheel_indices]), dim=1
        )

    def _reward_feet_horizontal_motion(self):
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.0).float()
        horizontal_speed_sq = torch.sum(
            torch.square(self.feet_vel[:, :, :2]), dim=2
        )
        return torch.sum(horizontal_speed_sq * contact, dim=1)

    def _reward_feet_air_horizontal_motion(self):
        """Discourage stepping as an easy workaround during fast pose changes."""
        air = (self.contact_forces[:, self.feet_indices, 2] <= 5.0).float()
        horizontal_speed_sq = torch.sum(
            torch.square(self.feet_vel[:, :, :2]), dim=2
        )
        return torch.sum(horizontal_speed_sq * air, dim=1)

    def _reward_feet_vertical_motion(self):
        """Suppress wheel lift and vertical chatter during body poses."""
        return torch.sum(torch.square(self.feet_vel[:, :, 2]), dim=1)

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
            + torch.abs(self.yaw_command_targets).unsqueeze(1)
            * self.cfg.rewards.lateral_symmetry_yaw_allowance
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
        yaw_gate = torch.exp(
            -torch.square(self.yaw_command_targets)
            / self.cfg.rewards.yaw_symmetry_gate_sigma
        )
        return (
            torch.abs(front_delta_body[:, 0])
            + torch.abs(rear_delta_body[:, 0])
        ) * yaw_gate

    def _reward_height_leg_coordination(self):
        """Match the common leg flexion to a continuous height reference.

        The previous variance-only term allowed all four legs to remain equally
        folded at the nominal 0.54 m command. The common component is now
        constrained for every pose, while the individual-leg shape prior is
        enabled only when roll, pitch, and yaw are near zero.
        """
        crouch_fraction = (
            self.cfg.rewards.default_body_height
            - self.commands[:, self.HEIGHT_COMMAND]
        ) / self.cfg.rewards.height_crouch_range
        crouch_fraction = torch.clamp(
            crouch_fraction,
            min=-self.cfg.rewards.height_extension_fraction_limit,
            max=1.0,
        )
        target_hip_flexion = (
            crouch_fraction * self.cfg.rewards.height_crouch_hip_target
        )
        target_knee_flexion = (
            crouch_fraction * self.cfg.rewards.height_crouch_knee_target
        )
        orientation_error = (
            torch.square(self.commands[:, self.ROLL_COMMAND])
            + torch.square(self.commands[:, self.PITCH_COMMAND])
            + torch.square(self.yaw_command_targets)
        )
        orientation_gate = torch.exp(
            -orientation_error
            / self.cfg.rewards.height_coordination_orientation_sigma
        )

        joint_error = self.dof_pos - self.default_dof_pos
        abad_error = joint_error[:, self.leg_abad_indices]
        hip_flexion = (
            joint_error[:, self.leg_hip_indices] * self.crouch_hip_sign
        )
        knee_flexion = (
            joint_error[:, self.leg_knee_indices] * self.crouch_knee_sign
        )
        common_error = (
            torch.square(
                torch.mean(hip_flexion, dim=1) - target_hip_flexion
            )
            + torch.square(
                torch.mean(knee_flexion, dim=1) - target_knee_flexion
            )
        )
        isolated_shape_error = (
            torch.mean(torch.square(abad_error), dim=1)
            + torch.mean(
                torch.square(
                    hip_flexion - target_hip_flexion.unsqueeze(1)
                ),
                dim=1,
            )
            + torch.mean(
                torch.square(
                    knee_flexion - target_knee_flexion.unsqueeze(1)
                ),
                dim=1,
            )
        )
        return common_error + orientation_gate * isolated_shape_error

    def _reward_leg_action_magnitude(self):
        """Prevent a static saturated leg command from evading action-rate cost."""
        return torch.sum(
            torch.square(self.actions[:, self.leg_joint_indices]), dim=1
        )

    def _reward_pitch_leg_coordination(self):
        """Keep pure pitch in the sagittal plane with paired front/rear legs."""
        pitch_weight = torch.clamp(
            torch.abs(self.commands[:, self.PITCH_COMMAND])
            / self.cfg.rewards.pitch_coordination_full_scale,
            max=1.0,
        )
        other_axis_error = (
            torch.square(self.commands[:, self.ROLL_COMMAND])
            + torch.square(self.yaw_command_targets)
            + torch.square(
                self.commands[:, self.HEIGHT_COMMAND]
                - self.cfg.rewards.default_body_height
            )
        )
        other_axis_gate = torch.exp(
            -other_axis_error
            / self.cfg.rewards.pitch_coordination_other_axis_sigma
        )

        joint_error = self.dof_pos - self.default_dof_pos
        abad_error = joint_error[:, self.leg_abad_indices]
        hip_error = joint_error[:, self.leg_hip_indices]
        knee_error = joint_error[:, self.leg_knee_indices]
        pair_error = (
            torch.square(hip_error[:, 0] - hip_error[:, 1])
            + torch.square(hip_error[:, 2] - hip_error[:, 3])
            + torch.square(knee_error[:, 0] - knee_error[:, 1])
            + torch.square(knee_error[:, 2] - knee_error[:, 3])
        )
        coordination_error = (
            torch.mean(torch.square(abad_error), dim=1) + pair_error
        )
        return pitch_weight * other_axis_gate * coordination_error

    def _neutral_pose_weight(self):
        """Return a smooth gate that disables stance rewards during dance."""
        pose_command_error = torch.square(
            self.commands[:, self.ROLL_COMMAND]
        ) + torch.square(self.commands[:, self.PITCH_COMMAND])
        yaw_command_error = torch.square(self.commands[:, 2])
        height_command_error = torch.square(
            self.commands[:, self.HEIGHT_COMMAND]
            - self.cfg.rewards.default_body_height
        )
        return torch.exp(
            -pose_command_error / self.cfg.rewards.neutral_orientation_sigma
            -yaw_command_error / self.cfg.rewards.yaw_tracking_sigma
            -height_command_error / self.cfg.rewards.neutral_height_sigma
        )
