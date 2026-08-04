import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.utils.math import get_scale_shift

from .zgwt_robot import Zgwt
from .zgwt_dance_config import ZGWTDanceCfg
from .zgwt_dance_utils import (
    effective_actions,
    height_tracking_score,
    leg_extension_from_knee,
    low_pass_alpha,
    normalize_dance_commands,
    select_wheel_factors,
    stance_coordination_score,
)


class ZgwtDance(Zgwt):
    """Fixed-support task tracking bounded body yaw, roll, pitch, and height."""

    cfg: ZGWTDanceCfg

    ROLL_COMMAND = 3
    PITCH_COMMAND = 4
    HEIGHT_COMMAND = 5

    def _init_buffers(self):
        # 父类会在环境创建后再次采样 payload/COM，但不会把第二次样本写回
        # PhysX。先保存真正用于创建刚体的样本，初始化其余 buffer 后再恢复。
        created_payload = self.payload.clone()
        created_com_displacement = self.com_displacement.clone()
        super()._init_buffers()
        self.payload = created_payload
        self.com_displacement = created_com_displacement
        if not hasattr(self, "policy_dof_indices"):
            self.policy_dof_indices = torch.arange(
                self.num_actions, dtype=torch.long, device=self.device
            )
        if not hasattr(self, "policy_wheel_indices"):
            self.policy_wheel_indices = self.wheel_indices
        if self.policy_dof_indices.numel() != self.num_actions:
            raise ValueError(
                "policy_dof_indices must contain exactly num_actions entries"
            )
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

        # 协调性按 FBL、FAR、RBL、RAR 的膝关节计算，轮关节天然被排除。
        leg_order = ("FBL", "FAR", "RBL", "RAR")
        knee_indices = []
        for leg_name in leg_order:
            joint_name = f"{leg_name}_KNEE_JOINT"
            if joint_name not in self.dof_names:
                raise ValueError(f"资产缺少协调性计算所需关节: {joint_name}")
            knee_indices.append(self.dof_names.index(joint_name))
        self.stance_knee_indices = torch.tensor(
            knee_indices, dtype=torch.long, device=self.device
        )
        self.stance_hip_x = torch.tensor(
            self.cfg.rewards.stance_hip_x,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        self.stance_hip_y = torch.tensor(
            self.cfg.rewards.stance_hip_y,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)

        pd_actions = torch.tensor(
            self.cfg.rewards.pd_equivalent_actions,
            dtype=torch.float,
            device=self.device,
        )
        if pd_actions.numel() != self.num_actions:
            raise ValueError("pd_equivalent_actions 的长度必须等于 num_actions")
        pd_actions = effective_actions(
            pd_actions.unsqueeze(0), self.policy_wheel_indices
        ).squeeze(0)
        self.pd_equivalent_actions = pd_actions.unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self.takeover_elapsed = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.takeover_duration = torch.zeros_like(self.takeover_elapsed)
        self.history_reset_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._sample_takeover_duration(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )

    def step(self, actions):
        """屏蔽轮关节动作，并在 episode 开始时从原 PD 目标平滑接管。"""
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"expected actions [{self.num_envs}, {self.num_actions}], "
                f"got {list(actions.shape)}"
            )
        policy_actions = effective_actions(actions, self.policy_wheel_indices)
        if bool(self.cfg.rewards.takeover_blend_enabled):
            blend = torch.clamp(
                self.takeover_elapsed / self.takeover_duration,
                min=0.0,
                max=1.0,
            )
            executed_actions = torch.lerp(
                self.pd_equivalent_actions, policy_actions, blend
            )
            self.takeover_elapsed += self.dt
        else:
            executed_actions = policy_actions
        return super().step(executed_actions)

    def _sample_takeover_duration(self, env_ids):
        """为每个 episode 采样接管混合时长，避免固定时序过拟合。"""
        duration_range = self.cfg.rewards.takeover_blend_duration_range
        if len(duration_range) != 2 or duration_range[0] <= 0.0:
            raise ValueError("takeover_blend_duration_range 必须为两个正数")
        if duration_range[1] < duration_range[0]:
            raise ValueError("接管混合时长上限不能小于下限")
        self.takeover_duration[env_ids] = torch_rand_float(
            duration_range[0],
            duration_range[1],
            (len(env_ids), 1),
            device=self.device,
        )

    def _process_rigid_body_props(self, props, env_id):
        """把 payload 质量和等效质心关联成同一个物理样本。"""
        if bool(
            getattr(
                self.cfg.domain_rand, "correlate_payload_and_com", False
            )
        ):
            payload_mass = self.payload[env_id, 0]
            base_mass = float(self.default_rigid_body_mass[0])
            nominal_base_com = torch.tensor(
                [props[0].com.x, props[0].com.y, props[0].com.z],
                dtype=torch.float,
                device=self.device,
            )
            payload_com = torch.tensor(
                self.cfg.domain_rand.equivalent_payload_com,
                dtype=torch.float,
                device=self.device,
            )
            # base 刚体的新 COM 是原 base 与等效 payload 的质量加权平均。
            correlated_offset = payload_mass / (
                base_mass + payload_mass
            ) * (payload_com - nominal_base_com)
            delta = torch.tensor(
                self.cfg.domain_rand.com_displacement_delta,
                dtype=torch.float,
                device=self.device,
            )
            if delta.numel() != 3 or torch.any(delta < 0.0):
                raise ValueError("com_displacement_delta 必须是三个非负数")
            sampled_unit = self.com_displacement[env_id]
            self.com_displacement[env_id] = correlated_offset + sampled_unit * delta
        return super()._process_rigid_body_props(props, env_id)

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
        height_deficit = torch.relu(
            self.commands[env_ids, self.HEIGHT_COMMAND]
            - height[env_ids]
            - float(self.cfg.rewards.height_deficit_deadzone)
        )
        _, coordination_residual = self._stance_coordination_score_and_residual()
        coordination_abs = torch.abs(coordination_residual[env_ids])
        leg_extension = self._leg_extension()[env_ids]
        extension_spread = torch.max(leg_extension, dim=1).values - torch.min(
            leg_extension, dim=1
        ).values
        body_stability_score = self._reward_body_stability()[env_ids]
        support_stability_score = self._reward_support_stability()[env_ids]

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
        support_drift = torch.norm(
            support_xy - self.episode_start_support_xy[env_ids], dim=1
        )
        support_xy_drift = torch.mean(support_drift)
        support_loss_termination_rate = torch.mean(
            (
                support_drift
                > float(self.cfg.rewards.severe_support_loss_distance)
            ).float()
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

        self.takeover_elapsed[env_ids] = 0.0
        self._sample_takeover_duration(env_ids)
        self.history_reset_pending[env_ids] = True
        self.actions[env_ids] = self.pd_equivalent_actions[env_ids]
        self.last_actions[env_ids] = self.pd_equivalent_actions[env_ids]
        self.last_last_actions[env_ids] = self.pd_equivalent_actions[env_ids]
        if hasattr(self, "delayed_actions"):
            self.delayed_actions[env_ids] = self.pd_equivalent_actions[
                env_ids, None, :
            ]

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
        episode["support_loss_termination_rate"] = (
            support_loss_termination_rate
        )
        episode["mean_feet_xy_drift"] = feet_xy_drift
        episode["max_individual_wheel_drift"] = (
            max_individual_wheel_drift
        )
        episode["mean_height_deficit"] = torch.mean(height_deficit)
        episode["height_deficit_ratio"] = torch.mean(
            (height_deficit > 0.0).float()
        )
        episode["mean_leg_extension_spread"] = torch.mean(extension_spread)
        episode["p95_coordination_residual"] = torch.quantile(
            coordination_abs.reshape(-1), 0.95
        )
        episode["mean_body_stability_score"] = torch.mean(
            body_stability_score
        )
        episode["mean_support_stability_score"] = torch.mean(
            support_stability_score
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
        policy_dof_pos = self.dof_pos[:, self.policy_dof_indices]
        policy_default_dof_pos = self.default_dof_pos[:, self.policy_dof_indices]
        policy_dof_vel = self.dof_vel[:, self.policy_dof_indices]
        dof_err = policy_dof_pos - policy_default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.policy_wheel_indices] = 0.0
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
                policy_dof_vel * self.obs_scales.dof_vel,
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

    def compute_observations(self):
        current_obs = self._build_current_observation(self.add_noise)
        updated_history = torch.cat(
            (
                current_obs[:, : self.num_one_step_obs],
                self.obs_buf[:, : -self.num_one_step_obs],
            ),
            dim=-1,
        )
        pending_ids = self.history_reset_pending.nonzero(
            as_tuple=False
        ).flatten()
        if len(pending_ids) > 0:
            # reset 后用同一真实帧填满历史，避免混入上一个 episode 或全零历史。
            current_actor = current_obs[pending_ids, : self.num_one_step_obs]
            updated_history[pending_ids] = current_actor.repeat(
                1, self.history_length
            )
            self.history_reset_pending[pending_ids] = False
        self.obs_buf = updated_history
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
        support_lost = self._support_center_drift() > float(
            self.cfg.rewards.severe_support_loss_distance
        )
        self.reset_buf |= excessive_tilt | too_low | support_lost

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

    # ------------ reward functions: Task Tracking / 任务跟踪 ------------
    def _reward_tracking_body_roll(self):
        """跟踪机身 roll 命令。"""
        roll, _ = self._current_roll_pitch()
        error = torch.square(roll - self.commands[:, self.ROLL_COMMAND])
        return torch.exp(-error / float(self.cfg.rewards.roll_tracking_sigma))

    def _reward_tracking_body_pitch(self):
        """跟踪机身 pitch 命令。"""
        _, pitch = self._current_roll_pitch()
        error = torch.square(pitch - self.commands[:, self.PITCH_COMMAND])
        return torch.exp(-error / float(self.cfg.rewards.pitch_tracking_sigma))

    def _reward_tracking_body_yaw(self):
        """跟踪相对 episode 初始朝向的 yaw 角度。"""
        error = torch.square(self._body_yaw_error())
        return torch.exp(-error / float(self.cfg.rewards.yaw_tracking_sigma))

    def _reward_tracking_body_height(self):
        """跟踪当前高度命令，并额外惩罚超过死区的向下塌陷。"""
        return height_tracking_score(
            self._current_base_height(),
            self.commands[:, self.HEIGHT_COMMAND],
            float(self.cfg.rewards.height_tracking_sigma),
            float(self.cfg.rewards.height_deficit_deadzone),
            float(self.cfg.rewards.height_deficit_sigma),
            float(self.cfg.rewards.height_deficit_weight),
        )

    # ------------ reward functions: Stance Stability / 站姿稳定 ------------
    def _reward_tracking_lin_vx(self):
        """跟踪 x 方向零速度命令，防止机器人前后漂移。"""
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(
            -error / float(self.cfg.rewards.lin_velocity_tracking_sigma)
        )

    def _reward_tracking_lin_vy(self):
        """跟踪 y 方向零速度命令，防止机器人左右漂移。"""
        error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(
            -error / float(self.cfg.rewards.lin_velocity_tracking_sigma)
        )

    def _reward_body_stability(self):
        """抑制竖直速度和机身角速度，平面速度由分轴 tracking 负责。"""
        cost = torch.square(self.base_lin_vel[:, 2]) / float(
            self.cfg.rewards.body_lin_vel_z_sigma
        )
        cost += torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) / float(
            self.cfg.rewards.body_ang_vel_xy_sigma
        )
        cost += torch.square(self.base_ang_vel[:, 2]) / float(
            self.cfg.rewards.body_yaw_rate_sigma
        )
        return torch.exp(-cost)

    def _reward_stance_coordination(self):
        """允许四腿共模压缩，惩罚单腿或单侧差模折叠。"""
        score, _ = self._stance_coordination_score_and_residual()
        return score

    def _reward_support_stability(self):
        """奖励稳定的支撑中心和四轮有效接触。"""
        position_cost = torch.square(self._support_anchor_excess()) / float(
            self.cfg.rewards.support_position_tracking_sigma
        )
        min_force = float(self.cfg.rewards.support_min_contact_force)
        contact_deficit = torch.clamp(
            (min_force - self.contact_forces[:, self.feet_indices, 2])
            / min_force,
            min=0.0,
            max=1.0,
        )
        contact_cost = torch.mean(torch.square(contact_deficit), dim=1) / float(
            self.cfg.rewards.support_contact_sigma
        )
        return torch.exp(-(position_cost + contact_cost))

    def _leg_extension(self):
        """计算四条腿的髋到轮心等效伸展量，不使用 wheel joint。"""
        return leg_extension_from_knee(
            self.dof_pos[:, self.stance_knee_indices],
            float(self.cfg.rewards.thigh_length),
            float(self.cfg.rewards.shank_length),
        )

    def _expected_extension_differential(self):
        """计算 roll/pitch 命令在平地上需要的四腿伸展量差。"""
        roll = self.commands[:, self.ROLL_COMMAND].unsqueeze(1)
        pitch = self.commands[:, self.PITCH_COMMAND].unsqueeze(1)
        hip_height_offset = (
            -torch.sin(pitch) * self.stance_hip_x
            + torch.cos(pitch) * torch.sin(roll) * self.stance_hip_y
        )
        return hip_height_offset - torch.mean(
            hip_height_offset, dim=1, keepdim=True
        )

    def _stance_coordination_score_and_residual(self):
        """分离四腿的共模压缩和扣除姿态命令后的差模残差。"""
        roll_ratio = torch.abs(
            self.commands[:, self.ROLL_COMMAND]
        ) / float(self.cfg.commands.max_abs_roll)
        pitch_ratio = torch.abs(
            self.commands[:, self.PITCH_COMMAND]
        ) / float(self.cfg.commands.max_abs_pitch)
        deadzone = float(self.cfg.rewards.stance_coordination_deadzone)
        deadzone += float(
            self.cfg.rewards.stance_command_deadzone_gain
        ) * (roll_ratio + pitch_ratio)
        return stance_coordination_score(
            self._leg_extension(),
            self._expected_extension_differential(),
            deadzone.unsqueeze(1),
            float(self.cfg.rewards.stance_coordination_sigma),
        )

    def _support_anchor_excess(self):
        """返回超过支撑中心允许死区的平面漂移量。"""
        drift = self._support_center_drift()
        return torch.clamp(
            drift - float(self.cfg.rewards.support_anchor_deadzone), min=0.0
        )

    def _support_center_drift(self):
        """计算支撑中心相对 episode 初始位置的平面漂移。"""
        support_xy = torch.mean(self.feet_pos[:, :, :2], dim=1)
        return torch.norm(
            support_xy - self.episode_start_support_xy, dim=1
        )

    # ------------ reward functions: Control Quality / 控制质量 ------------
    def _reward_action_rate(self):
        """惩罚相邻控制步的动作变化。"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # ------------ reward functions: Safety Constraints / 安全约束 ------------
    def _reward_collision(self):
        """惩罚非足端刚体与地面发生碰撞。"""
        contact = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        )
        return torch.sum((contact > 0.1).float(), dim=1)

    def _reward_dof_pos_limits(self):
        """惩罚关节位置超出软限位。"""
        lower = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        upper = (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(lower + upper, dim=1)

    def _reward_torque_limits(self):
        """惩罚达到 99% 力矩上限的饱和关节。"""
        saturated = torch.abs(self.torques) >= 0.99 * self.torque_limits
        return torch.sum(saturated.float(), dim=1)

    def _reward_severe_wheel_park(self):
        """惩罚轮关节严重偏离 episode 初始驻车角。"""
        wheel_error = torch.abs(
            self.wheel_park_targets - self.dof_pos[:, self.wheel_indices]
        )
        return torch.any(
            wheel_error > float(self.cfg.rewards.severe_wheel_park_error), dim=1
        ).float()

    def _reward_severe_support_loss(self):
        """惩罚支撑中心发生不可接受的大幅漂移。"""
        return (
            self._support_center_drift()
            > float(self.cfg.rewards.severe_support_loss_distance)
        ).float()

    def _reward_base_height_too_low(self):
        """惩罚低于 hard threshold 的严重塌陷。"""
        return (
            self._current_base_height()
            < float(self.cfg.rewards.hard_min_height)
        ).float()

    def _reward_excessive_tilt(self):
        """惩罚超过 hard threshold 的严重倾斜。"""
        return (
            self.projected_gravity[:, 2]
            > -math.cos(float(self.cfg.rewards.hard_tilt))
        ).float()

    def _reward_termination(self):
        """只惩罚真实失败，不惩罚正常 episode 超时。"""
        return (self.reset_buf & ~self.time_out_buf).float()
