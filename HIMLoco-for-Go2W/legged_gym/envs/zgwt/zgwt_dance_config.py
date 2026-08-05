from .zgwt_config import ZGWTRoughCfg, ZGWTRoughCfgPPO


class ZGWTDanceCfg(ZGWTRoughCfg):
    """Stationary body-pose tracking task for ZGWT dance motions."""

    class init_state(ZGWTRoughCfg.init_state):
        # 出生时保留 0.55 m 的安全离地高度，进入控制后再跟踪 0.48 m。
        pos = [0.0, 0.0, 0.55]

    class asset(ZGWTRoughCfg.asset):
        # 纯狗舞蹈任务使用物理等价的轻量显示资产：惯性、关节和 collision
        # 与原始 URDF 完全一致，只用 primitive visual 代替 123 MB STL。
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/zgwt/urdf/"
            "zgwt_lightweight.urdf"
        )

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
        # body_yaw 目标单独保存；actor 的第 2 个命令槽接收相对 yaw 误差。
        num_commands = 6
        # 关闭父类 Go2W 的线速度课程，避免它把 lin_vel_x 从 0 自动扩大。
        # pose_range_curriculum_enabled 是独立的姿态角范围课程开关。
        curriculum = False
        heading_command = False
        resampling_time = 8.0
        command_filter_time_constant = 0.5
        # yaw 目标变化的独立硬限速；从 +0.10 到 -0.10 rad 至少需要 0.67 s。
        yaw_slew_rate = 0.30
        # 下面三个范围仍供姿态协调 reward 计算命令相对幅度，不参与
        # actor command observation 的除法归一化。
        max_abs_yaw = 0.10
        max_abs_roll = 0.26
        max_abs_pitch = 0.26
        min_height = 0.40
        command_scales = [2.0, 2.0, 1.0, 1.0, 1.0, 2.0]
        # 模式顺序固定为：
        # 0 neutral, 1 height, 2 roll, 3 pitch, 4 yaw,
        # 5 roll+pitch, 6 roll+yaw, 7 pitch+yaw, 8 roll+pitch+yaw,
        # 9 height+roll, 10 height+pitch, 11 height+yaw,
        # 12 height+roll+pitch, 13 height+roll+yaw,
        # 14 height+pitch+yaw, 15 height+roll+pitch+yaw。

        mode_probabilities = [
            0.20, 0.20, 0.20, 0.20, 0.20,
            0.00, 0.00, 0.00, 0.00,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        # class ranges 是课程初始值；这里仅设置各命令的最终目标。
        # 某项初始值与最终值相同，就表示该项不随课程变化。
        # 中断后若从某一级 checkpoint 续训，可手动设置 start_level。
        pose_range_curriculum_enabled =False
        pose_range_curriculum_start_level = 0
        pose_range_final = {
            "body_yaw": 0.10,
            "body_roll": 0.26,
            "body_pitch": 0.26,
            "body_height": 0.48,
        }
        pose_range_num_levels = 5

        # 每一级至少训练 300 s（约 625 次 48-step PPO iteration），然后才
        # 根据完整 episode 的跟踪、失败、足端锚定和接触指标判断是否升级。
        # 固定时间只作为最低训练量，不会到点强制升级。
        pose_range_min_level_time_s = 300.0
        pose_curriculum_min_episodes = 1024
        pose_curriculum_tracking_threshold = 0.85
        pose_curriculum_max_failure_rate = 0.05
        pose_curriculum_max_anchor_cost = 0.20
        pose_curriculum_max_contact_cost = 0.05
        pose_curriculum_required_success_windows = 2

        class ranges:
            # 线速度始终为零。body_yaw 是相对 episode 初始朝向的有限角度，
            # 不是连续旋转的 yaw 角速度。
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            body_yaw = [-0.03, 0.03]
            body_roll = [-0.08, 0.08]
            body_pitch = [-0.08, 0.08]
            body_height = [0.48, 0.48]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        randomize_payload_mass = True
        # 第一阶段只训练接近完整机械臂的持续重载，不再让大量轻载环境稀释
        # 满载下的屈腿承压和防外八梯度。
        payload_mass_range = [-1, 10.0]
        randomize_com_displacement = True
        correlate_payload_and_com = True
        equivalent_payload_com_min = [-0.2, -0.05, -0.2]
        equivalent_payload_com_max = [0.2, 0.05, 0.2]
        com_displacement_range = [-1.0, 1.0]
        com_displacement_is_offset = True
        randomize_link_mass = False
        link_mass_range = [1.0, 1.0]
        randomize_friction = False
        friction_range = [0.8, 0.8]
        randomize_restitution = False
        restitution_range = [0.0, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.90, 1.10]
        randomize_kp = False
        kp_range = [1.0, 1.0]
        randomize_kd = False
        kd_range = [1.0, 1.0]
        randomize_initial_joint_pos = False
        # Keep recovery diversity without starting from visibly crooked legs.
        initial_joint_pos_range = [0.90, 1.10]
        randomize_initial_base_velocity = False
        disturbance = False
        disturbance_range = [-5.0, 5.0]
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
            #跟踪
            tracking_body_roll = 6.0
            tracking_body_pitch = 6.0
            tracking_body_yaw = 6.0
            tracking_body_height = 6.0
            tracking_lin_vx = 4.0
            tracking_lin_vy = 4.0

            # 稳定
            body_stability = 4.0
            stance_coordination = 4.0
            support_stability = 0.0
            base_height_deficit = -4.0   #惩罚机身低于目标高度
            knee_clearance = -3.0  #惩罚膝关节过度屈曲，避免低机身局部最优
            feet_anchor = -3.0    #惩罚轮子偏离驻足世界坐标系
            feet_contact = -0.5   #轮子接触
            feet_outward = -2.0   #惩罚轮足外移和 ABAD 造成的轮子外八

            # 平滑
            action_rate = -0.01  #惩罚action变化

            # 安全
            collision = -1.0
            dof_pos_limits = -2.0
            torque_limits = -0.05
            severe_wheel_park = -3.0    #轮子驻足，是否转动
            severe_support_loss = -3.0  #支撑丢失
            base_height_too_low = -3.0
            excessive_tilt = -4.0    #过度倾斜
            termination = -10.0

        only_positive_rewards = False

        # 四类 reward 直接加权求和；下列 sigma 均保持物理量单位。
        roll_tracking_sigma = 0.012
        pitch_tracking_sigma = 0.012
        yaw_tracking_sigma = 0.0025
        height_tracking_sigma = 0.0016
        height_deficit_deadzone = 0.020
        height_deficit_sigma = 0.0004
        height_deficit_weight = 1.0

        # 平面零速单轴跟踪和其余机身稳定量的允许尺度。
        lin_velocity_tracking_sigma = 0.010
        body_lin_vel_z_sigma = 0.010
        body_ang_vel_xy_sigma = 0.250
        body_yaw_rate_sigma = 0.090

        # URDF 中大腿和小腿名义长度分别为 0.26 m、0.28 m。
        thigh_length = 0.26
        shank_length = 0.28
        # residual 是单腿相对四腿均值的误差。5 mm 死区对应约 1 cm 的
        # 前后腿伸展差；原 15 mm 会放过日志中的约 3.4 cm 前后差。
        stance_coordination_deadzone = 0.005
        stance_command_deadzone_gain = 0.010
        stance_coordination_sigma = 0.0001
        # 顺序固定为 FBL、FAR、RBL、RAR，与代码中的膝关节索引一致。
        stance_hip_x = [0.272, 0.272, -0.272, -0.272]
        stance_hip_y = [0.104, -0.104, 0.104, -0.104]

        support_min_contact_force = 5.0

        # 高度目标以下 2 cm 保留接触柔顺空间，再用 Smooth-L1 连续惩罚。
        height_deficit_penalty_scale = 0.020
        # KNEE_LINK 原点就是膝关节中心。提前在碰撞发生前约束所有四膝，
        # 不针对后腿写特例，因而仍允许 roll/pitch 所需的协调屈伸。
        min_knee_clearance = 0.20
        knee_clearance_penalty_scale = 0.040

        # 3 deg 的旧死区会放过肉眼可见的轮子倾斜；收紧到约 1.7 deg。
        # sigma 暂时保持不变，先通过权重和死区温和加强，避免姿态抖动。
        feet_outward_deadzone = 0.005
        feet_outward_sigma = 0.0004
        abad_outward_deadzone = 0.030
        abad_outward_sigma = 0.010

        # 仅供 episode diagnostics 判断命令是否激活。
        active_orientation_threshold = 0.03
        active_yaw_threshold = 0.02
        active_height_threshold = 0.02
        neutral_command_sigma = 0.20

        # 四轮世界坐标 anchor 使用 Smooth-L1；最大单轮项权重大于平均项。
        feet_anchor_deadzone = 0.010
        feet_anchor_penalty_scale = 0.020
        feet_anchor_mean_weight = 0.35
        feet_anchor_max_weight = 0.65

        default_body_height = 0.48   # 带臂时允许四腿主动屈曲，以较低姿态承压
        hard_min_height = 0.36
        hard_tilt = 0.45
        termination_tilt = 0.55
        termination_min_height = 0.32
        severe_wheel_park_error = 0.35
        severe_support_loss_distance = 0.20
        severe_individual_foot_drift = 0.10

class ZGWTDanceCfgPPO(ZGWTRoughCfgPPO):
    class policy(ZGWTRoughCfgPPO.policy):
        init_noise_std = 0.15

    class algorithm(ZGWTRoughCfgPPO.algorithm):
        # 当前短阶段需要尽快跳出旧的高站姿和外八局部最优；固定学习率只提高
        # 一倍，并继续使用小 clip、梯度裁剪和 1000 次上限约束更新风险。
        learning_rate = 5.0e-5
        # Adaptive scheduling raised 1e-4 to 6.67e-3 around iteration 500 in
        # the failed run, after which value loss jumped above 1e3.
        schedule = "fixed"
        desired_kl = 0.01
        entropy_coef = 0.0
        min_action_std = 0.08
        max_action_std = 0.25
        clip_param = 0.08
        max_grad_norm = 0.5
        num_learning_epochs = 2

    class runner(ZGWTRoughCfgPPO.runner):
        experiment_name = "ZGWT_DANCE"
        run_name = "stage1_v3_8.5"
        # 小范围姿态阶段频繁保存，优先检查 400/800/1200。
        save_interval = 200
        max_iterations = 20000
        resume = True   #从 checkpoint 继续训练时必须改为 True
   
        reset_iteration_on_load = True  #是否重置迭代编号
        # reward 聚合与 critic 目标已改变，只继承接口仍兼容的 actor/estimator。
        load_actor_only = True
        # 同时重置 Adam，避免旧 reward 的优化器动量污染新阶段。
        load_optimizer = True
        load_run = "Aug05_17-03-17_stage1_v2_8.5"
        checkpoint = 400
