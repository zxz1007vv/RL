import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from .zgwt_arm_control_utils import (
    advance_limited_joint_reference,
    inertia_normalized_desired_acceleration,
    load_simulation_verified_poses,
    normalized_pose_weights,
    torque_rate_limit,
)
from .zgwt_arm_dynamics import ZgwtArmDynamics
from .zgwt_arm_config import ZGWTDanceArmCfg
from ..zgwt.zgwt_dance_robot import ZgwtDance


class ZgwtDanceArm(ZgwtDance):
    """腿部策略与六轴传统控制器解耦的 ZGWT 带臂舞蹈环境。"""

    SOFT_REWARD_NAMES = ZgwtDance.SOFT_REWARD_NAMES | {
        "neutral_abduction_pose"
    }

    cfg: ZGWTDanceArmCfg

    def _create_envs(self):
        super()._create_envs()

        configured_arm_names = list(self.cfg.arm.joint_names)
        arm_name_to_index = {
            name: self.dof_names.index(name)
            for name in configured_arm_names
            if name in self.dof_names
        }
        if list(arm_name_to_index) != configured_arm_names:
            missing = [
                name for name in configured_arm_names if name not in arm_name_to_index
            ]
            raise ValueError("带臂资产缺少机械臂关节: " + ", ".join(missing))

        arm_indices = [arm_name_to_index[name] for name in configured_arm_names]
        policy_indices = [
            index for index in range(self.num_dof) if index not in set(arm_indices)
        ]
        if len(policy_indices) != self.num_actions:
            raise ValueError(
                f"带臂资产应包含 {self.num_actions} 个轮腿 DOF 和 6 个机械臂 DOF，"
                f"实际总 DOF={self.num_dof}，轮腿 DOF={len(policy_indices)}"
            )
        expected_policy_names = [
            name for name in self.dof_names if not name.startswith("ROBOT_ARM_JOINT")
        ]
        actual_policy_names = [self.dof_names[index] for index in policy_indices]
        if actual_policy_names != expected_policy_names:
            raise ValueError("轮腿关节顺序与原 16-D 策略接口不一致")

        self.arm_dof_indices = torch.tensor(
            arm_indices, dtype=torch.long, device=self.device
        )
        self.policy_dof_indices = torch.tensor(
            policy_indices, dtype=torch.long, device=self.device
        )
        policy_position = {
            dof_index: action_index
            for action_index, dof_index in enumerate(policy_indices)
        }
        self.policy_wheel_indices = torch.tensor(
            [policy_position[int(index)] for index in self.wheel_indices.tolist()],
            dtype=torch.long,
            device=self.device,
        )
        self.policy_abad_indices = torch.tensor(
            [
                policy_position[index]
                for index in policy_indices
                if "ABAD_JOINT" in self.dof_names[index]
            ],
            dtype=torch.long,
            device=self.device,
        )
        if self.policy_abad_indices.numel() != 4:
            raise ValueError("轮腿策略必须包含 4 个 ABAD 关节")
        self.arm_body_indices = torch.tensor(
            [
                handle
                for handle in (
                    self.gym.find_actor_rigid_body_handle(
                        self.envs[0],
                        self.actor_handles[0],
                        f"ROBOT_ARM_LINK{index}",
                    )
                    for index in range(1, 7)
                )
                if handle >= 0
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _init_buffers(self):
        super()._init_buffers()

        arm = self.cfg.arm
        if abs(float(arm.control_dt) - float(self.sim_params.dt)) > 1.0e-9:
            raise ValueError(
                f"机械臂控制周期 {arm.control_dt} 必须等于 PhysX dt "
                f"{self.sim_params.dt}；控制器在每个物理子步执行。"
            )
        if not (
            bool(arm.gravity_compensation)
            and bool(arm.coriolis_compensation)
        ):
            raise ValueError(
                "带臂训练必须启用重力和科氏/离心补偿，"
                "避免产生与 C++ 不一致的残缺控制器。"
            )
        self.arm_kp = torch.tensor(
            arm.kp, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_kd = torch.tensor(
            arm.kd, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_torque_limits = torch.tensor(
            arm.torque_limits, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_torque_rate_limits = torch.tensor(
            arm.torque_rate_limits, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_locked_velocity_limits = torch.tensor(
            arm.locked_velocity_limit, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_manual_velocity_limits = torch.tensor(
            arm.manual_velocity_limit, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_acceleration_limits = torch.tensor(
            arm.acceleration_limit, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.arm_gravity_world = torch.tensor(
            [0.0, 0.0, -9.81], dtype=torch.float, device=self.device
        )
        self.arm_home = torch.tensor(
            arm.home, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        self.arm_dynamics = ZgwtArmDynamics.from_urdf(
            asset_path,
            list(arm.joint_names),
            device=self.device,
            dtype=torch.float,
        )
        self.arm_command_targets = self.arm_home.repeat(self.num_envs, 1)
        self.arm_reference_position = self.arm_command_targets.clone()
        self.arm_reference_velocity = torch.zeros_like(self.arm_command_targets)
        self.arm_reference_acceleration = torch.zeros_like(
            self.arm_command_targets
        )
        self.arm_active_velocity_limits = self.arm_locked_velocity_limits.repeat(
            self.num_envs, 1
        )
        self.arm_previous_torque = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device
        )
        self.arm_pose_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.arm_pose_names = ["simulation_home"]
        self.arm_pose_categories = ["home_or_folded"]
        self.arm_pose_table = None
        self.arm_pose_weights = None

        pose_mode = str(arm.pose_mode).lower()
        if pose_mode not in {"home", "library"}:
            raise ValueError("arm.pose_mode 只能是 'home' 或 'library'")
        self.arm_pose_mode = pose_mode
        if pose_mode == "library":
            pose_file = arm.pose_library_file.format(
                LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
            )
            poses = load_simulation_verified_poses(
                pose_file,
                arm.position_lower,
                arm.position_upper,
                float(arm.minimum_joint_margin),
                float(arm.minimum_jacobian_sigma),
            )
            weights = normalized_pose_weights(
                poses, dict(arm.category_probabilities)
            )
            self.arm_pose_table = torch.tensor(
                [pose.q for pose in poses],
                dtype=torch.float,
                device=self.device,
            )
            self.arm_pose_weights = torch.tensor(
                weights, dtype=torch.float, device=self.device
            )
            self.arm_pose_names = [pose.pose_id for pose in poses]
            self.arm_pose_categories = [pose.category for pose in poses]
            print(
                f"### loaded {len(poses)} simulation-verified arm poses "
                f"from {os.path.abspath(pose_file)}"
            )

    @property
    def arm_targets(self):
        """兼容旧诊断名称；实际控制目标是受限轨迹参考位置。"""
        return self.arm_reference_position

    def _minimum_path_jacobian_sigma(self, start, target, base_quaternion):
        minimum_sigma = []
        step_size = float(self.cfg.arm.path_check_step)
        for row in range(start.shape[0]):
            steps = max(
                2,
                int(
                    torch.ceil(
                        torch.max(torch.abs(target[row] - start[row]))
                        / step_size
                    ).item()
                )
                + 1,
            )
            interpolation = torch.linspace(
                0.0, 1.0, steps, device=self.device
            ).unsqueeze(1)
            path = start[row : row + 1] + interpolation * (
                target[row : row + 1] - start[row : row + 1]
            )
            quaternion = base_quaternion[row : row + 1].repeat(steps, 1)
            _, _, jacobian = self.arm_dynamics.end_kinematics(
                path, base_quaternion=quaternion
            )
            minimum_sigma.append(
                torch.min(torch.linalg.svdvals(jacobian)[:, -1])
            )
        return torch.stack(minimum_sigma)

    def set_arm_joint_targets(
        self, targets, env_ids=None, manual=False, validate_path=True
    ):
        """设置机械臂最终关节目标，由传统轨迹限制器平滑执行。

        外部动态/遥操作目标默认限制在离硬限位 0.15 rad 的安全区。仿真 home
        位于 J2/J3 硬限位，只能通过 reset 内部路径使用，不能由本接口请求。
        """

        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
        if not torch.is_tensor(targets):
            targets = torch.tensor(
                targets, dtype=torch.float, device=self.device
            )
        targets = targets.to(device=self.device, dtype=torch.float)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0).repeat(len(env_ids), 1)
        if targets.shape != (len(env_ids), 6):
            raise ValueError(
                f"机械臂目标必须为 [{len(env_ids)}, 6]，实际 {list(targets.shape)}"
            )
        margin = float(self.cfg.arm.minimum_joint_margin)
        lower = torch.tensor(
            self.cfg.arm.position_lower, dtype=torch.float, device=self.device
        ) + margin
        upper = torch.tensor(
            self.cfg.arm.position_upper, dtype=torch.float, device=self.device
        ) - margin
        safe_targets = torch.minimum(
            torch.maximum(targets, lower), upper
        )
        if validate_path:
            minimum_sigma = self._minimum_path_jacobian_sigma(
                self.arm_reference_position[env_ids],
                safe_targets,
                self.root_states[env_ids, 3:7],
            )
            threshold = float(self.cfg.arm.minimum_jacobian_sigma)
            invalid = minimum_sigma < threshold
            if torch.any(invalid):
                raise ValueError(
                    "机械臂目标路径进入奇异软阈值，已拒绝。"
                    f" env_ids={env_ids[invalid].tolist()}, "
                    f"min_sigma={minimum_sigma[invalid].tolist()}, "
                    f"threshold={threshold}"
                )
        self.arm_command_targets[env_ids] = safe_targets
        velocity_limits = (
            self.arm_manual_velocity_limits
            if manual
            else self.arm_locked_velocity_limits
        )
        self.arm_active_velocity_limits[env_ids] = velocity_limits

    def _sample_arm_targets(self, env_ids):
        if self.arm_pose_mode == "home":
            self.arm_pose_index[env_ids] = -1
            return self.arm_home.repeat(len(env_ids), 1)

        sampled = torch.multinomial(
            self.arm_pose_weights, len(env_ids), replacement=True
        )
        self.arm_pose_index[env_ids] = sampled
        return self.arm_pose_table[sampled]

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        targets = self._sample_arm_targets(env_ids)
        self.arm_command_targets[env_ids] = targets
        self.arm_reference_position[env_ids] = targets
        self.arm_reference_velocity[env_ids] = 0.0
        self.arm_reference_acceleration[env_ids] = 0.0
        self.arm_active_velocity_limits[env_ids] = (
            self.arm_locked_velocity_limits
        )
        # 模式进入契约要求 target=measured q。这里在 reset 时直接把物理状态
        # 放到目标姿态，因此不会发生从零位跳向采样姿态的力矩冲击。
        self.dof_pos[env_ids[:, None], self.arm_dof_indices] = targets
        self.dof_vel[env_ids[:, None], self.arm_dof_indices] = 0.0
        self.arm_previous_torque[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        from isaacgym import gymtorch

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    @staticmethod
    def _select_env_dof_factors(factors, indices):
        if factors.shape[1] == 1:
            return factors
        return factors[:, indices]

    def _compute_torques(self, actions):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"轮腿策略动作必须为 [{self.num_envs}, {self.num_actions}]"
            )

        policy_pos = self.dof_pos[:, self.policy_dof_indices]
        policy_vel = self.dof_vel[:, self.policy_dof_indices]
        policy_default = self.default_dof_pos[:, self.policy_dof_indices]
        policy_p = self.p_gains[self.policy_dof_indices]
        policy_d = self.d_gains[self.policy_dof_indices]
        kp_factors = self._select_env_dof_factors(
            self.Kp_factors, self.policy_dof_indices
        )
        kd_factors = self._select_env_dof_factors(
            self.Kd_factors, self.policy_dof_indices
        )

        actions_scaled = actions * float(self.cfg.control.action_scale)
        position_actions = actions_scaled.clone()
        position_actions[:, self.policy_wheel_indices] = 0.0
        position_error = policy_default - policy_pos
        position_error[:, self.policy_wheel_indices] = 0.0
        policy_torque = (
            policy_p * kp_factors * (position_actions + position_error)
            - policy_d * kd_factors * policy_vel
        )

        wheel_error = (
            self.wheel_park_targets - self.dof_pos[:, self.wheel_indices]
        )
        policy_torque[:, self.policy_wheel_indices] = (
            float(self.cfg.control.wheel_park_stiffness) * wheel_error
            - float(self.cfg.control.wheel_park_damping)
            * policy_vel[:, self.policy_wheel_indices]
        )
        motor_factors = self._select_env_dof_factors(
            self.motor_strength_factors, self.policy_dof_indices
        )
        policy_torque *= motor_factors

        (
            self.arm_reference_position,
            self.arm_reference_velocity,
            self.arm_reference_acceleration,
        ) = advance_limited_joint_reference(
            self.arm_command_targets,
            self.arm_reference_position,
            self.arm_reference_velocity,
            self.arm_active_velocity_limits,
            self.arm_acceleration_limits,
            float(self.cfg.arm.control_dt),
            float(self.cfg.arm.velocity_filter_tau),
        )
        arm_pos = self.dof_pos[:, self.arm_dof_indices]
        arm_vel = self.dof_vel[:, self.arm_dof_indices]
        desired_acceleration = inertia_normalized_desired_acceleration(
            self.arm_reference_acceleration,
            self.arm_reference_position - arm_pos,
            self.arm_reference_velocity - arm_vel,
            self.arm_kp,
            self.arm_kd,
        )
        base_angular_velocity = (
            self.root_states[:, 10:13]
            if bool(self.cfg.arm.compensate_base_angular_velocity)
            else torch.zeros_like(self.root_states[:, 10:13])
        )
        desired_arm_torque = self.arm_dynamics.inverse_dynamics(
            arm_pos,
            arm_vel,
            desired_acceleration,
            base_quaternion=self.root_states[:, 3:7],
            base_angular_velocity=base_angular_velocity,
            gravity=self.arm_gravity_world,
        )
        desired_arm_torque = torch.clip(
            desired_arm_torque,
            min=-self.arm_torque_limits,
            max=self.arm_torque_limits,
        )
        arm_torque = torque_rate_limit(
            desired_arm_torque,
            self.arm_previous_torque,
            float(self.arm_torque_rate_limits[0, 0].item()),
            float(self.cfg.arm.control_dt),
        )
        arm_torque = torch.clip(
            arm_torque,
            min=-self.arm_torque_limits,
            max=self.arm_torque_limits,
        )
        self.arm_previous_torque[:] = arm_torque

        torques = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
        )
        torques[:, self.policy_dof_indices] = policy_torque
        torques[:, self.arm_dof_indices] = arm_torque
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_neutral_abduction_pose(self):
        """在静止站立时专门抑制四条腿向外劈开的外八姿态。"""

        policy_pos = self.dof_pos[:, self.policy_dof_indices]
        policy_default = self.default_dof_pos[:, self.policy_dof_indices]
        error = policy_pos[:, self.policy_abad_indices] - policy_default[
            :, self.policy_abad_indices
        ]
        return torch.sum(torch.abs(error), dim=1) * self._neutral_pose_weight()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        arm_error = torch.mean(
            torch.abs(
                self.dof_pos[env_ids[:, None], self.arm_dof_indices]
                - self.arm_targets[env_ids]
            )
        )
        arm_torque_ratio = torch.mean(
            torch.abs(self.arm_previous_torque[env_ids])
            / self.arm_torque_limits
        )
        super().reset_idx(env_ids)
        self.extras["episode"]["arm_mean_abs_position_error"] = arm_error
        self.extras["episode"]["arm_mean_torque_ratio"] = arm_torque_ratio
        self.extras["episode"]["arm_pose_index_mean"] = torch.mean(
            self.arm_pose_index[env_ids].float()
        )

    def check_termination(self):
        super().check_termination()
        if self.arm_body_indices.numel() == 0:
            return
        arm_contact = torch.norm(
            self.contact_forces[:, self.arm_body_indices, :], dim=2
        ).max(dim=1).values
        self.reset_buf |= arm_contact > float(
            self.cfg.arm.arm_contact_termination_force
        )

    # 机械臂由传统控制器负责，不能让其控制代价改变 16-D 腿部策略 reward。
    def _reward_torques(self):
        return torch.sum(
            torch.square(self.torques[:, self.policy_dof_indices]), dim=1
        )

    def _reward_dof_pos_limits(self):
        position = self.dof_pos[:, self.policy_dof_indices]
        limits = self.dof_pos_limits[self.policy_dof_indices]
        out_of_limits = -(position - limits[:, 0]).clip(max=0.0)
        out_of_limits += (position - limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        policy_torque = self.torques[:, self.policy_dof_indices]
        policy_limits = self.torque_limits[self.policy_dof_indices]
        return torch.sum(
            (torch.abs(policy_torque) >= 0.99 * policy_limits).float(), dim=1
        )
