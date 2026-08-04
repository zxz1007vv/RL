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



        # 模式顺序固定为：
        # 0 neutral, 1 height, 2 roll, 3 pitch, 4 yaw,
        # 5 roll+pitch, 6 roll+yaw, 7 pitch+yaw, 8 roll+pitch+yaw,
        # 9 height+roll, 10 height+pitch, 11 height+yaw,
        # 12 height+roll+pitch, 13 height+roll+yaw,
        # 14 height+pitch+yaw, 15 height+roll+pitch+yaw。


        mode_probabilities = [
            0.40, 0.0, 0.20, 0.20, 0.20,
            0.00, 0.00, 0.00, 0.00,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]



        class ranges:
            # Linear velocity remains zero. body_yaw is a bounded angle relative
            # to the heading captured at episode reset, not a continuous turn rate.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]


            body_yaw = [-0.03, 0.03]
            body_roll = [-0.08, 0.08]
            body_pitch = [-0.08, 0.08]
            body_height = [0.48, 0.48]

            # 阶段 2、3、4：
            # body_yaw = [-0.10, 0.10]
            # body_roll = [-0.26, 0.26]
            # body_pitch = [-0.26, 0.26]
            # body_height = [0.40, 0.54]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        randomize_payload_mass = True
        # 第一阶段只训练接近完整机械臂的持续重载，不再让大量轻载环境稀释
        # 满载下的屈腿承压和防外八梯度。
        payload_mass_range = [6.0, 10.0]
        randomize_com_displacement = True
        correlate_payload_and_com = True
        equivalent_payload_com_min = [-0.4, -0.05, 0.2]  #质心偏移，考虑机械臂的位置主要在-x +z
        equivalent_payload_com_max = [-0.2, 0.05, 0.5]
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
            body_stability = 3.0
            stance_coordination = 4.0
            support_stability = 3.0  #约束轮足世界坐标和有效接触
           

            # 平滑
            action_rate = -0.01  #惩罚action变化

            # 安全
            collision = -1.0
            dof_pos_limits = -2.0
            torque_limits = -0.05
            severe_wheel_park = -4.0    #轮子驻足
            severe_support_loss = -4.0  #支撑丢失
            base_height_too_low = -4.0
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
        stance_coordination_deadzone = 0.015
        stance_command_deadzone_gain = 0.010
        stance_coordination_sigma = 0.0004
        # 顺序固定为 FBL、FAR、RBL、RAR，与代码中的膝关节索引一致。
        stance_hip_x = [0.272, 0.272, -0.272, -0.272]
        stance_hip_y = [0.104, -0.104, 0.104, -0.104]

        support_min_contact_force = 5.0
        support_contact_sigma = 0.25

        # 单独约束每只轮足相对出生锚点向身体外侧滑动，避免四脚对称外八时
        # 支撑中心保持不变，从而绕过 support_stability。
        feet_outward_deadzone = 0.005
        feet_outward_sigma = 0.0004
        abad_outward_deadzone = 0.050
        abad_outward_sigma = 0.010

        # 仅供 episode diagnostics 判断命令是否激活。
        active_orientation_threshold = 0.03
        active_yaw_threshold = 0.02
        active_height_threshold = 0.02
        neutral_command_sigma = 0.20

        # 四个轮足在 episode 初始世界坐标附近保持驻足。平均项约束整体滑动，
        # 最大项专门捕捉单轮异常；1 cm 死区过滤接触求解器的小幅抖动。
        feet_position_tracking_sigma = 0.0004
        max_foot_position_tracking_sigma = 0.000225
        feet_anchor_deadzone = 0.010

        default_body_height = 0.48   # 带臂时允许四腿主动屈曲，以较低姿态承压
        hard_min_height = 0.36
        hard_tilt = 0.45
        termination_tilt = 0.55
        termination_min_height = 0.32
        severe_wheel_park_error = 0.35
        severe_support_loss_distance = 0.20

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
        run_name = "stage1_v2_8.4"
        # 小范围姿态阶段频繁保存，优先检查 400/800/1200。
        save_interval = 100
        max_iterations = 10000
        # 仅修改 mode probabilities、command ranges、noise 或窄范围随机化：
        # 可保留 actor/critic/estimator，optimizer 是否保留应做短对照。
        # 仅小幅修改某个 reward 权重：
        # resume=True, load_actor_only=False, load_optimizer=False。
        # 修改核心 reward、reward 尺度或 privileged observation：
        # resume=True, load_actor_only=True, load_optimizer=False。
        # 修改 actor observation 维度/顺序或 action 维度后，旧 actor 不兼容。
        resume = True
        # 人工切换到新训练阶段时只继承模型权重，不继承上一阶段的迭代编号。
        # 同一阶段中断后续训时必须临时改为 False。
        reset_iteration_on_load = True
        # reward 聚合与 critic 目标已改变，只继承接口仍兼容的 actor/estimator。
        load_actor_only = True
        # 同时重置 Adam，避免旧 reward 的优化器动量污染新阶段。
        load_optimizer = False
        load_run = "Aug04_15-12-15_stage0a_h48"
        checkpoint = 1600
