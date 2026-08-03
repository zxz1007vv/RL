from ..zgwt.zgwt_dance_config import ZGWTDanceCfg, ZGWTDanceCfgPPO


class ZGWTDanceArmCfg(ZGWTDanceCfg):
    """Aligned ArmStand：完整机械臂物理模型下重新训练自然站立。"""

    class env(ZGWTDanceCfg.env):
        # 策略接口保持原样：360-D actor history、78-D critic、16-D action。
        num_actions = 16

    class init_state(ZGWTDanceCfg.init_state):
        default_joint_angles = dict(ZGWTDanceCfg.init_state.default_joint_angles)
        default_joint_angles.update(
            {
                "ROBOT_ARM_JOINT1": 0.0,
                "ROBOT_ARM_JOINT2": 0.0,
                "ROBOT_ARM_JOINT3": 0.0,
                "ROBOT_ARM_JOINT4": 0.0,
                "ROBOT_ARM_JOINT5": 0.0,
                "ROBOT_ARM_JOINT6": 0.0,
            }
        )

    class asset(ZGWTDanceCfg.asset):
        # 训练和 play 统一加载带完整 STL visual 的组合模型。
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/zgwsarm/zgwsarm.urdf"
        name = "zgwsarm"
        # Isaac Gym 中 0 表示启用资产自碰撞过滤，1 表示全部禁用。
        self_collisions = 0
        flip_visual_attachments = False

    class commands(ZGWTDanceCfg.commands):
        # 第一阶段只训练自然稳定站立，不同时增加舞蹈命令难度。
        mode_probabilities = [
            1.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        class ranges(ZGWTDanceCfg.commands.ranges):
            body_yaw = [0.0, 0.0]
            body_roll = [0.0, 0.0]
            body_pitch = [0.0, 0.0]
            body_height = [0.54, 0.54]

    # 带臂任务的 reward 在这里完整定义，不继承 ZGWTDanceCfg.rewards；
    # 本阶段针对带臂承载下的压弯和外八独立调参。
    class rewards:
        class scales:
            # 姿态命令跟踪。roll/pitch 共用目标重力方向误差。
            tracking_body_orientation = 13.0
            tracking_body_yaw = 6.0
            tracking_body_height = 9.0

            # 静止保持。先约束支撑中心，再约束各轮足位置。
            tracking_lin_vx = 3.0
            tracking_lin_vy = 3.0
            tracking_support_position = 4.0
            tracking_feet_position = 2.0
            tracking_max_foot_position = 1.5

            # 稳定、平滑和安全项。scales 是唯一 reward 权重来源。
            yaw_rate = -0.05
            collision = -1.0
            feet_contact = -1.0
            feet_stumble = -0.1
            feet_vertical_motion = -0.1
            action_rate = -0.01
            action_smoothness = -0.005
            # 允许策略为承载机械臂输出必要力矩，同时加强实际腿姿约束。
            torques = -5.0e-6
            neutral_joint_pose = -5.0
            neutral_abduction_pose = -4.0
            dof_pos_limits = -2.0
            torque_limits = -0.1
            severe_wheel_park = -4.0
            severe_support_loss = -4.0

        only_positive_rewards = False

        # 最终任务 reward 和软约束乘子。
        task_reward_scale = 28.0
        hold_gate_sigma = 0.50
        auxiliary_reward_sigma = 0.10
        auxiliary_reward_floor = 0.30

        # 各跟踪误差的指数尺度。
        orientation_tracking_sigma = 0.012
        yaw_tracking_sigma = 0.0025
        height_tracking_sigma = 0.0016
        tracking_sigma = 0.25

        # 中立姿态诊断及 neutral reward 门控。
        active_orientation_threshold = 0.03
        active_yaw_threshold = 0.02
        active_height_threshold = 0.02
        neutral_command_sigma = 0.20

        # 轮足/支撑位置保持尺度和接触求解容差。
        feet_position_tracking_sigma = 0.003
        support_position_tracking_sigma = 0.001
        max_foot_position_tracking_sigma = 0.001
        feet_anchor_deadzone = 0.015
        support_anchor_deadzone = 0.010

        # 通用关节、力矩和接触安全阈值。
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 100.0

        # 默认姿态及 episode 终止/严重失稳阈值。
        base_height_target = 0.54
        default_body_height = 0.54
        termination_tilt = 0.55
        termination_min_height = 0.32
        severe_wheel_park_error = 0.35
        severe_support_loss_distance = 0.20

    class arm:
        joint_names = [f"ROBOT_ARM_JOINT{i}" for i in range(1, 7)]
        home = [0.0] * 6
        position_lower = [-2.618, 0.0, -2.9671, -1.57, -1.57, -6.28]
        position_upper = [2.618, 3.14, 0.0, 1.57, 1.57, 6.28]

        # "home" 为 ArmStandV1；"library" 为 ArmStandV2-Sim。
        pose_mode = "home"
        pose_library_file = (
            "{LEGGED_GYM_ROOT_DIR}/config/robots/zgwsarm/"
            "arm_safe_static_poses.csv"
        )
        minimum_joint_margin = 0.15
        minimum_jacobian_sigma = 0.02
        category_probabilities = {
            "home_or_folded": 0.30,
            "workspace_midrange": 0.40,
            "directional_extension": 0.20,
            "near_allowed_boundary": 0.10,
        }

        # 唯一控制律为惯量归一化计算力矩；Kp/Kd 是加速度域闭环增益。
        kp = [500.0] * 6
        kd = [45.0] * 6
        torque_limits = [100.0] * 6
        torque_rate_limits = [1000.0] * 6
        control_dt = 0.002
        locked_velocity_limit = [0.8] * 6
        manual_velocity_limit = [0.5] * 6
        acceleration_limit = [2.0] * 6
        velocity_filter_tau = 0.005
        path_check_step = 0.02
        arm_contact_termination_force = 5.0
        gravity_compensation = True
        coriolis_compensation = True
        compensate_base_angular_velocity = True
        compensate_base_acceleration = False

    class domain_rand(ZGWTDanceCfg.domain_rand):
        # 完整机械臂保留名义动力学，在其周围加入低幅随机化，覆盖 URDF/MJCF、
        # 电机和接触差异；范围刻意小于旧无臂 payload 随机化，避免破坏自然腿姿。
        randomize_payload_mass = True
        payload_mass_range = [-0.5, 1.5]
        randomize_com_displacement = True
        com_displacement_range = [-0.015, 0.015]
        com_displacement_is_offset = True
        randomize_link_mass = True
        link_mass_range = [0.95, 1.05]
        randomize_friction = True
        friction_range = [0.70, 1.10]
        randomize_restitution = False
        restitution_range = [0.0, 0.05]
        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]
        randomize_kp = True
        kp_range = [0.95, 1.05]
        randomize_kd = True
        kd_range = [0.90, 1.10]
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.97, 1.03]
        randomize_initial_base_velocity = False
        disturbance = False
        push_robots = False
        delay = False
        observation_delay = False

    class noise(ZGWTDanceCfg.noise):
        add_noise = False


class ZGWTDanceArmCfgPPO(ZGWTDanceCfgPPO):
    class runner(ZGWTDanceCfgPPO.runner):
        experiment_name = "ZGWT_DANCE_ARM"
        run_name = "armstand_aligned_v1"
        save_interval = 250
        max_iterations = 20000
        resume = True
        # 只把旧 ArmStandV1 的 actor/estimator 当作初始化；新控制律、reward、
        # 随机化和 critic 均从新阶段重新适应，不继承迭代号或 Adam 状态。
        reset_iteration_on_load = True
        load_actor_only = True
        load_optimizer = False
        load_experiment_name = "ZGWT_DANCE_ARM"
        load_run = "Jul31_18-46-40_armstandv1"
        checkpoint = 20000


class ZGWTDanceArmStaticCfg(ZGWTDanceArmCfg):
    """ArmStand 静态姿态库阶段：每个 episode 固定一个安全姿态。"""

    class arm(ZGWTDanceArmCfg.arm):
        pose_mode = "library"


class ZGWTDanceArmStaticCfgPPO(ZGWTDanceArmCfgPPO):
    class runner(ZGWTDanceArmCfgPPO.runner):
        experiment_name = "ZGWT_DANCE_ARM"
        run_name = "armstandv2_aligned"
        # 后续开始姿态库阶段时，应改为加载新完成的 armstand_aligned_v1。
        load_experiment_name = "ZGWT_DANCE_ARM"
        load_run = "Jul31_18-46-40_armstandv1"
        checkpoint = 20000
        # 新阶段只继承网络权重，从第 0 代开始，并重置 Adam。
        reset_iteration_on_load = True
        load_optimizer = False
