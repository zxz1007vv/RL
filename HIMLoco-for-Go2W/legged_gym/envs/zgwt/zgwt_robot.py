from legged_gym.envs.base.legged_robot import LeggedRobot

from .zgwt_config import ZGWTRoughCfg
import torch


class Zgwt(LeggedRobot):
    """Zgwt wheel-legged robot environment."""

    cfg: ZGWTRoughCfg
    
    def _compute_torques(self, actions):
        """根据策略网络输出的 actions 计算最终发送给 Isaac Gym 的关节力矩。

        ZGWT 是轮腿机器人，这里把关节分成两类来处理：
        1. 普通腿部关节：ABAD / HIP / KNEE，使用位置 PD 控制。
           策略输出 action 先乘以 action_scale，作为默认关节角附近的目标角偏移。
        2. 轮足关节：FOOT_JOINT，也就是 self.wheel_indices 指向的 DOF。
           在 P 控制模式下，轮足不再按位置误差做 PD，而是单独覆盖为：
               torque = action * action_scale * kp - kd * kd_random_factor * dof_vel
           也就是动作直接映射为驱动力矩，同时保留一个速度阻尼项抑制轮子过快转动。

        注意：
        - 返回的 torques 维度必须等于机器人全部 DOF 数量，即 shape = [num_envs, num_dof]。
        - self.p_gains / self.d_gains 是每个 DOF 一组增益，shape 通常是 [num_dof]。
        - self.Kp_factors / self.Kd_factors 是域随机化系数，shape 通常是 [num_envs, 1]，
          会通过 PyTorch broadcasting 作用到每个环境里的所有关节。
        - 最后会用 self.torque_limits 对力矩做裁剪，避免超过 URDF / asset 里定义的关节力矩上限。

        Args:
            actions (torch.Tensor): 策略输出动作，shape = [num_envs, num_actions]。

        Returns:
            torch.Tensor: 裁剪后的关节力矩，shape = [num_envs, num_dof]。
        """
        # 关节位置误差：默认目标角 - 当前角。
        # 这里的 self.default_dof_pos 已经在 _init_buffers 里从 cfg.init_state.default_joint_angles 读入。
        # 对普通腿部关节来说，后面会使用：
        #     target_pos - current_pos = action_scaled + default_dof_pos - dof_pos
        # 所以这里先计算 default_dof_pos - dof_pos，再加上 action_scaled。
        dof_err = self.default_dof_pos - self.dof_pos 

        # 轮足关节不参与普通位置 PD 控制，因此把轮足位置误差清零。
        # 如果不清零，FOOT_JOINT 会被拉回 default_dof_pos，轮子就会表现得像普通转角关节。
        dof_err[:,self.wheel_indices] =  0 #

        # 将策略输出缩放到控制量尺度。
        # 对普通腿部关节：actions_scaled 表示目标关节角偏移，单位接近 rad。
        # 对轮足关节：actions_scaled 后面会被单独解释为力矩指令的比例量。
        actions_scaled = actions * self.cfg.control.action_scale 

        # pos_actions_scaled 专门给普通位置 PD 使用。
        # 必须 clone 一份，避免把 actions_scaled 里的轮足动作也清零；
        # 因为后面轮足单独控制时仍然需要原始的 actions_scaled[:, self.wheel_indices]。
        pos_actions_scaled = actions_scaled.clone()

        # 普通位置 PD 分支里，轮足目标位置偏移设为 0。
        # 这样先算出的 torques 中，轮足位置项不会起作用；随后会再覆盖轮足力矩。
        pos_actions_scaled[:, self.wheel_indices] = 0

        # 构造速度参考 vel_ref。
        # 普通腿部关节的速度参考保持 0，相当于 PD 控制中的 D 项希望关节速度趋近 0。
        # 轮足关节的速度参考来自 action * vel_scale，主要给 V 控制或基础公式保留兼容性。
        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale 
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices] 

        # 控制模式由 cfg.control.control_type 指定：
        # "P" 表示位置 PD；
        # "V" 表示速度控制；
        # "T" 表示策略直接输出力矩。
        control_type = self.cfg.control.control_type 

        if control_type=="P":
            # 普通腿部关节位置 PD：
            #   torque = kp * kp_factor * (target_pos - current_pos)
            #          + kd * kd_factor * (target_vel - current_vel)
            #
            # 其中：
            #   target_pos - current_pos = pos_actions_scaled + default_dof_pos - dof_pos
            #   target_vel = vel_ref
            #
            # 对轮足关节来说，上面已经把 dof_err 和 pos_actions_scaled 都清零；
            # 因此这个临时 torques 里的轮足位置项为 0，后面会立刻用轮足专用公式覆盖。
            torques = self.p_gains * self.Kp_factors * (pos_actions_scaled + dof_err) + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)

            # 轮足关节专用控制：
            #   torque = action_scaled * kp - kd * kd_factor * current_vel
            #
            # 这里 intentionally 不使用：
            #   default_dof_pos - dof_pos
            # 因为轮子/轮足不应该被控制到某个固定角度，而应该由 action 直接给出驱动力矩。
            #
            # self.wheel_indices 来自 cfg.asset.wheel_name = ["FOOT_JOINT"]，
            # 会匹配到 FBL/FAR/RBL/RAR 四个 FOOT_JOINT 的 DOF index。
            #
            # self.Kd_factors 的 shape 是 [num_envs, 1]，会广播到所有轮足关节列；
            # 这样每个环境可以有不同的阻尼随机化系数。
            torques[:, self.wheel_indices] = (
                actions_scaled[:, self.wheel_indices] * self.p_gains[self.wheel_indices]
                - self.d_gains[self.wheel_indices] * self.Kd_factors * self.dof_vel[:, self.wheel_indices]
            )
        elif control_type=="V":
            # 速度控制模式：
            #   torque = kp * (target_vel - current_vel)
            #          - kd * acceleration
            # 其中 acceleration 用当前速度和上一控制步速度差分近似。
            # 注意这里沿用原始实现，没有对轮足做额外覆盖。
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            # 力矩控制模式：
            # 策略输出乘以 action_scale 后直接作为力矩。
            # 最终仍会被 torque_limits 裁剪。
            torques = actions_scaled
        else:
            # 配置里出现未知控制类型时立即报错，避免静默使用错误控制逻辑。
            raise NameError(f"Unknown controller type: {control_type}")

        # 按每个 DOF 的力矩上限进行裁剪。
        # 这一步很重要，否则策略输出或 PD 误差过大时，可能给仿真施加不现实的力矩。
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
