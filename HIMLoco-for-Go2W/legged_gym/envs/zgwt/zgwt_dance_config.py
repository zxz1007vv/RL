from .zgwt_config import ZGWTRoughCfg, ZGWTRoughCfgPPO


class ZGWTDanceCfg(ZGWTRoughCfg):
    """Stationary body-pose tracking task for ZGWT dance motions."""

    class env(ZGWTRoughCfg.env):
        # ang_vel(3) + gravity(3) + commands(6) + q(16) + qd(16) + actions(16)
        num_one_step_observations = 3 + 3 + 6 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        # actor observation + base velocity + disturbance + four 3-D contacts.
        # This is a plane task; there is no 187-D terrain scan.
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 12
        num_privileged_obs = num_one_step_privileged_obs

    class terrain(ZGWTRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        max_init_terrain_level = 0

    class normalization(ZGWTRoughCfg.normalization):
        # action_scale=0.25, so joint-target offsets are capped at +/-0.50 rad.
        # This prevents one bad PPO update from immediately folding the legs.
        clip_actions = 2.0

    class control(ZGWTRoughCfg.control):
        # A damping-only brake still lets the wheel angle drift. Position-hold
        # each wheel at its reset angle so the four contact points cannot roll.
        wheel_park_stiffness = 25.0
        wheel_park_damping = 2.5

    class commands(ZGWTRoughCfg.commands):
        # [vx, vy, body_yaw_error, body_roll, body_pitch, body_height]
        # The sampled body_yaw target is stored separately. The actor receives
        # target_yaw - current_relative_yaw in command slot 2.
        num_commands = 6
        curriculum = False
        heading_command = False
        resampling_time = 8.0
        command_filter_time_constant = 0.5
        # Independent hard limit for body-yaw target changes. A full reversal
        # from +0.10 to -0.10 rad therefore takes at least 0.67 seconds.
        yaw_slew_rate = 0.30
        # 固定 observation normalization，人工阶段切换时不要修改。
        max_abs_yaw = 0.10
        max_abs_roll = 0.26
        max_abs_pitch = 0.26
        min_height = 0.40

        # ============================================================
        # MANUAL TRAINING STAGE / 人工训练阶段（唯一生效配置区）
        #
        # 每完成一个阶段：
        # 1. 保存 checkpoint
        # 2. 修改下面的 mode_probabilities 和 ranges
        # 3. 设置 runner.resume = True
        # 4. 加载上一个阶段 checkpoint
        #
        # 不要修改 observation、action、command 顺序和核心 reward。
        # 程序运行期间不会自动改变概率、范围、噪声或随机化。
        # ============================================================

        # 模式顺序固定为：
        # 0 neutral, 1 height, 2 roll, 3 pitch, 4 yaw,
        # 5 roll+pitch, 6 roll+yaw, 7 pitch+yaw, 8 roll+pitch+yaw,
        # 9 height+roll, 10 height+pitch, 11 height+yaw,
        # 12 height+roll+pitch, 13 height+roll+yaw,
        # 14 height+pitch+yaw, 15 height+roll+pitch+yaw。

        # 上一阶段：standv2（带窄范围动力学随机化的稳定站立）。
        # mode_probabilities = [
        #     1.0,
        #     0.0, 0.0, 0.0, 0.0,
        #     0.0, 0.0, 0.0, 0.0,
        #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # ]

        # 当前启用：stage1v2（小范围单轴姿态）。
        # 暂不加入 roll+pitch 组合；先确认每个单轴在随机化下都能稳定跟踪。
        mode_probabilities = [
            0.35, 0.10, 0.20, 0.20, 0.15, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        # 阶段 2 推荐值（完整姿态，height 不与姿态混合）：
        # mode_probabilities = [
        #     0.15, 0.10, 0.10, 0.10, 0.10, 0.10,
        #     0.10, 0.10, 0.15,
        #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # ]
        # ranges: yaw +/-0.10, roll/pitch +/-0.26, height 0.40~0.54。

        # 阶段 3/4 推荐值（完整 height 混合；阶段 4 只额外开启窄随机化）：
        # mode_probabilities = [
        #     0.08, 0.08, 0.04, 0.04, 0.04,
        #     0.05, 0.05, 0.05, 0.12,
        #     0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.12,
        # ]
        # ranges: yaw +/-0.10, roll/pitch +/-0.26, height 0.40~0.54。

        class ranges:
            # Linear velocity remains zero. body_yaw is a bounded angle relative
            # to the heading captured at episode reset, not a continuous turn rate.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            # 阶段 0：
            # body_yaw = [0.0, 0.0]
            # body_roll = [0.0, 0.0]
            # body_pitch = [0.0, 0.0]
            # body_height = [0.54, 0.54]

            # 当前启用：stage1v2（小范围单轴姿态）。
            body_yaw = [-0.025, 0.025]
            body_roll = [-0.08, 0.08]
            body_pitch = [-0.08, 0.08]
            body_height = [0.49, 0.54]

            # 阶段 2、3、4：
            # body_yaw = [-0.10, 0.10]
            # body_roll = [-0.26, 0.26]
            # body_pitch = [-0.26, 0.26]
            # body_height = [0.40, 0.54]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        # 当前启用：stage1v2，完整继承 standv2 的窄范围动力学随机化；
        # 暂不叠加 motor、noise、push、delay 或初始状态扰动。
        randomize_payload_mass = True
        payload_mass_range = [0.0, 6.0]
        randomize_com_displacement = True
        # 相对 URDF 名义 base COM 的三轴偏移，每个环境独立采样。
        com_displacement_range = [-0.02, 0.02]
        com_displacement_is_offset = True
        randomize_link_mass = False
        link_mass_range = [1.0, 1.0]
        randomize_friction = True
        # 覆盖当前 MuJoCo 地面 friction=0.8。
        friction_range = [0.75, 1.15]
        randomize_restitution = False
        restitution_range = [0.0, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.90, 1.10]
        randomize_kp = True
        kp_range = [0.90, 1.10]
        randomize_kd = True
        kd_range = [0.90, 1.10]
        randomize_initial_joint_pos = False
        # Keep recovery diversity without starting from visibly crooked legs.
        initial_joint_pos_range = [0.90, 1.10]
        randomize_initial_base_velocity = False
        disturbance = False
        disturbance_range = [-10.0, 10.0]
        disturbance_interval = 8
        push_robots = False
        push_interval_s = 12
        max_push_vel_xy = 1.2
        delay = False
        # 暂未实现 observation delay；阶段 4 也保持关闭。
        observation_delay = False

    class noise(ZGWTRoughCfg.noise):
        add_noise = False
        # 阶段 4：人工改为 add_noise=True，noise_level=0.20。
        noise_level = 0.20

    class rewards(ZGWTRoughCfg.rewards):
        class scales:
            # Roll and pitch share one target-gravity reward. This avoids a
            # second "stay level" objective fighting the commanded body pose.
            tracking_body_orientation = 11.0
            tracking_body_yaw = 6.0
            tracking_body_height = 6.0
            # Hold objectives are gated by command-tracking convergence. Support
            # centre has priority over exact individual wheel-centre anchoring.
            tracking_lin_vx = 3.0
            tracking_lin_vy = 3.0
            tracking_support_position = 3.0
            tracking_feet_position = 1.0
            tracking_max_foot_position = 0.5

            # Minimal stability, smoothness, and safety terms.
            yaw_rate = -0.05
            collision = -1.0
            feet_contact = -1.0
            feet_stumble = -0.1
            feet_vertical_motion = -0.1
            action_rate = -0.01
            action_smoothness = -0.005
            torques = -8.0e-6
            neutral_joint_pose = -3.0
            dof_pos_limits = -2.0
            torque_limits = -0.1
            severe_wheel_park = -4.0
            severe_support_loss = -4.0

            # 人工调参建议：动作抖动时，后期可小幅增加 action_rate 或
            # action_smoothness；每次只改 1～2 项，单次不超过 20%～30%。

        only_positive_rewards = False

        # First track the commanded body pose, then tighten stationary support
        # as tracking converges. Safety/smoothness costs use a floored multiplier.
        task_reward_scale = 28.0
        hold_gate_sigma = 0.50
        auxiliary_reward_sigma = 0.10
        auxiliary_reward_floor = 0.30

        # scales 是 tracking、hold、soft 和 hard reward 的唯一权重来源。
        # 父类虽然会统一乘以 dt，但 tracking/hold 加权平均的分子、分母
        # 使用同一组 dt-scaled weights，公共 dt 会在归一化时抵消。
        # 足端漂移仍明显时，可小幅提高上方 support/feet 对应的 scale；
        # 每次只改 1～2 项且不超过 20%～30%。核心 reward 结构或数值尺度
        # 若发生大改，只加载 actor/estimator，并重置 critic 和 optimizer。

        # reward = exp(-error / sigma). Orientation error is measured between
        # actual and commanded projected gravity, not against a level body.
        orientation_tracking_sigma = 0.012
        yaw_tracking_sigma = 0.0025
        height_tracking_sigma = 0.0016

        # Used only by diagnostics and the explicitly neutral-task gate.
        active_orientation_threshold = 0.03
        active_yaw_threshold = 0.02
        active_height_threshold = 0.02
        neutral_command_sigma = 0.20

        # Soft support deadzones retain contact-solver tolerance. Wheel axle
        # position-hold remains enabled independently in the controller.
        feet_position_tracking_sigma = 0.003
        support_position_tracking_sigma = 0.001
        max_foot_position_tracking_sigma = 0.001
        feet_anchor_deadzone = 0.015
        support_anchor_deadzone = 0.010

        default_body_height = 0.54
        termination_tilt = 0.55
        termination_min_height = 0.32
        severe_wheel_park_error = 0.35
        severe_support_loss_distance = 0.20


class ZGWTDanceCfgPPO(ZGWTRoughCfgPPO):
    class policy(ZGWTRoughCfgPPO.policy):
        init_noise_std = 0.15

    class algorithm(ZGWTRoughCfgPPO.algorithm):
        # 旧 stage1v1 在约 500～1000 次迭代后开始退化。任务切换时采用
        # 更保守的 PPO 更新，避免破坏 standv2 已学到的稳定站立策略。
        learning_rate = 5.0e-5
        # Adaptive scheduling raised 1e-4 to 6.67e-3 around iteration 500 in
        # the failed run, after which value loss jumped above 1e3.
        schedule = "fixed"
        desired_kl = 0.01
        entropy_coef = 0.0
        min_action_std = 0.05
        max_action_std = 0.25
        clip_param = 0.10
        max_grad_norm = 0.5
        num_learning_epochs = 2

    class runner(ZGWTRoughCfgPPO.runner):
        experiment_name = "ZGWT_DANCE"
        run_name = "stage1v2"
        # stage1v1 的最佳区间出现得较早，加密 checkpoint 便于人工选择。
        save_interval = 250
        max_iterations = 10000
        # 仅修改 mode probabilities、command ranges、noise 或窄范围随机化：
        # resume=True, load_actor_only=False, load_optimizer=True。
        # 仅小幅修改 auxiliary reward 权重：
        # resume=True, load_actor_only=False, load_optimizer=False。
        # 修改核心 reward、reward 尺度或 privileged observation：
        # resume=True, load_actor_only=True, load_optimizer=False。
        # 修改 actor observation 维度/顺序或 action 维度后，旧 actor 不兼容。
        resume = True
        # Use load_actor_only=True for checkpoints predating the 78-D critic
        # input or the refactored task reward. Actor/estimator shapes stay valid.
        load_actor_only = False
        # 保留 actor/critic/estimator 权重，但为新命令分布重置 Adam 状态。
        load_optimizer = False
        load_run = "Jul30_11-38-25_standv2"
        checkpoint = 4000
