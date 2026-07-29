import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.utils.math import get_scale_shift

from .zgwt_robot import Zgwt
from .zgwt_dance_config import ZGWTDanceCfg


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
        # Convert front/rear joint signs into one common positive crouch axis.
        self.crouch_hip_sign = torch.tensor(
            [-1.0, -1.0, 1.0, 1.0], device=self.device
        )
        self.crouch_knee_sign = torch.tensor(
            [1.0, 1.0, -1.0, -1.0], device=self.device
        )
        self.initial_command_mode_probabilities = self._normalized_probabilities(
            self.cfg.commands.initial_mode_probabilities, "initial"
        )
        self.final_command_mode_probabilities = self._normalized_probabilities(
            self.cfg.commands.final_mode_probabilities, "final"
        )
        self.command_modes = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

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
        if probability_sum <= 0:
            raise ValueError(
                f"dance {label}_mode_probabilities must have a positive sum"
            )
        return probabilities / probability_sum

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
        roll_inactive_error = self._masked_mean(
            roll_abs_error, 1.0 - roll_active
        )
        pitch_inactive_error = self._masked_mean(
            pitch_abs_error, 1.0 - pitch_active
        )
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
        height_inactive_error = self._masked_mean(
            height_abs_error, 1.0 - height_active
        )
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
        default_height_weight = (
            torch.abs(
                self.pose_command_targets[env_ids, 2]
                - self.cfg.rewards.default_body_height
            ) < self.cfg.rewards.active_height_threshold
        ).float()
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
        yaw_inactive_error = self._masked_mean(
            yaw_abs_error, 1.0 - yaw_command_active
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
        feet_xy_drift = torch.mean(
            torch.norm(
                self.feet_pos[env_ids, :, :2]
                - self.episode_start_feet_xy[env_ids],
                dim=2,
            )
        )
        yaw_active = yaw_command_active
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
        height_pose_mixed_fraction = torch.mean(
            (self.command_modes[env_ids] >= 9).float()
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
        self.extras["episode"]["mean_abs_roll_error_inactive"] = (
            roll_inactive_error
        )
        self.extras["episode"]["mean_abs_pitch_error_inactive"] = (
            pitch_inactive_error
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
        self.extras["episode"]["mean_abs_height_error_inactive"] = (
            height_inactive_error
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
        self.extras["episode"]["mean_abs_yaw_error_inactive"] = (
            yaw_inactive_error
        )
        self.extras["episode"]["mean_yaw_active_fraction"] = torch.mean(
            yaw_command_active
        )
        self.extras["episode"]["mean_xy_drift"] = xy_drift
        self.extras["episode"]["mean_support_xy_drift"] = support_xy_drift
        self.extras["episode"]["mean_feet_xy_drift"] = feet_xy_drift
        self.extras["episode"]["mean_yaw_xy_drift"] = yaw_xy_drift
        self.extras["episode"]["mean_abs_yaw_lin_vel_x"] = yaw_forward_speed
        self.extras["episode"]["mean_height_pose_mixed_fraction"] = (
            height_pose_mixed_fraction
        )
        self.extras["episode"]["aux_multiplier_mean"] = torch.mean(
            aux_episode_mean
        )
        self.extras["episode"]["aux_multiplier_p05"] = torch.quantile(
            aux_episode_mean, 0.05
        )
        self.extras["episode"]["aux_multiplier_min"] = torch.min(
            aux_episode_mean
        )

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

    def _current_mode_probabilities(self):
        curriculum_fraction = self._command_curriculum_fraction()
        start = float(self.cfg.commands.mixed_height_curriculum_start)
        end = float(self.cfg.commands.mixed_height_curriculum_end)
        denominator = max(end - start, 1.0e-6)
        blend = min(max((curriculum_fraction - start) / denominator, 0.0), 1.0)
        return torch.lerp(
            self.initial_command_mode_probabilities,
            self.final_command_mode_probabilities,
            blend,
        )

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
            self._current_mode_probabilities(), count, replacement=True
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

        self.command_modes[env_ids] = mode
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
        # Keep the height input zero-centred like the other pose commands.
        # The reward and public command remain in absolute metres.
        observed_commands[:, self.HEIGHT_COMMAND] -= (
            self.cfg.rewards.default_body_height
        )

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
            "final_task",
        ):
            self.episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )

    def compute_reward(self):
        """Combine tracking as the task and penalties as multiplicative costs.

        Command tracking is always primary. Stationary support is introduced as
        tracking converges, and the auxiliary multiplier has a nonzero floor so
        safety costs cannot erase the command-tracking gradient.
        """
        tracking_log = torch.zeros_like(self.rew_buf)
        tracking_weight_sum = 0.0
        hold_log = torch.zeros_like(self.rew_buf)
        hold_weight_sum = 0.0
        auxiliary_reward = torch.zeros_like(self.rew_buf)

        for name, reward_function in zip(
            self.reward_names, self.reward_functions
        ):
            raw_reward = reward_function()
            scaled_reward = raw_reward * self.reward_scales[name]
            self.episode_sums[name] += scaled_reward
            if name in self.TRACK_REWARD_NAMES:
                tracking_weight = abs(float(self.reward_scales[name]))
                tracking_log += tracking_weight * torch.log(
                    torch.clamp(raw_reward, min=1.0e-6, max=1.0)
                )
                tracking_weight_sum += tracking_weight
            elif name in self.HOLD_REWARD_NAMES:
                hold_weight = abs(float(self.reward_scales[name]))
                hold_log += hold_weight * torch.log(
                    torch.clamp(raw_reward, min=1.0e-6, max=1.0)
                )
                hold_weight_sum += hold_weight
            else:
                auxiliary_reward += scaled_reward

        if tracking_weight_sum <= 0.0 or hold_weight_sum <= 0.0:
            raise RuntimeError("dance task requires tracking and hold rewards")

        tracking_cost = -tracking_log / tracking_weight_sum
        tracking_score = torch.exp(-tracking_cost)
        hold_score = torch.exp(hold_log / hold_weight_sum)
        hold_gate = torch.exp(
            -tracking_cost / float(self.cfg.rewards.hold_gate_sigma)
        )
        task_score = tracking_score * (
            (1.0 - hold_gate) + hold_gate * hold_score
        )
        auxiliary_exponent = torch.clamp(
            auxiliary_reward / float(self.cfg.rewards.auxiliary_reward_sigma),
            min=-20.0,
            max=0.0,
        )
        auxiliary_floor = float(self.cfg.rewards.auxiliary_reward_floor)
        auxiliary_multiplier = auxiliary_floor + (1.0 - auxiliary_floor) * (
            torch.exp(auxiliary_exponent)
        )
        task_reward = (
            float(self.cfg.rewards.task_reward_scale) * self.dt * task_score
        )
        self.rew_buf[:] = task_reward * auxiliary_multiplier
        self.episode_sums["command_tracking"] += tracking_score * self.dt
        self.episode_sums["stationary_hold"] += hold_score * self.dt
        self.episode_sums["hold_gate"] += hold_gate * self.dt
        self.episode_sums["aux_multiplier"] += auxiliary_multiplier * self.dt
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

    def _current_roll_pitch(self):
        gravity = self.projected_gravity
        pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        return roll, pitch

    def _target_projected_gravity(self):
        """Gravity vector expected in the body frame at commanded roll/pitch."""
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
        """One for the explicitly neutral target, zero for dance tasks."""
        orientation_neutral = (
            torch.abs(self.pose_command_targets[:, 0])
            < self.cfg.rewards.active_orientation_threshold
        ) & (
            torch.abs(self.pose_command_targets[:, 1])
            < self.cfg.rewards.active_orientation_threshold
        )
        yaw_neutral = (
            torch.abs(self.yaw_command_targets)
            < self.cfg.rewards.active_yaw_threshold
        )
        height_neutral = (
            torch.abs(
                self.pose_command_targets[:, 2]
                - self.cfg.rewards.default_body_height
            ) < self.cfg.rewards.active_height_threshold
        )
        return (orientation_neutral & yaw_neutral & height_neutral).float()
