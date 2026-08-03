from ..zgwt.zgwt_dance_config import ZGWTDanceCfg, ZGWTDanceCfgPPO


class ZGWTDanceArmCfg(ZGWTDanceCfg):
    """ArmStandV1：完整机械臂物理模型，机械臂固定在仿真 home。"""

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

    # 带臂任务的 reward 在这里完整定义，不继承 ZGWTDanceCfg.rewards。
    # 初始数值与父鸡头任务保持一致，后续 ArmStand 调参只修改本节。
    class rewards:
        class scales:
            # 姿态命令跟踪。roll/pitch 共用目标重力方向误差。
            tracking_body_orientation = 11.0
            tracking_body_yaw = 6.0
            tracking_body_height = 6.0

            # 静止保持。先约束支撑中心，再约束各轮足位置。
            tracking_lin_vx = 3.0
            tracking_lin_vy = 3.0
            tracking_support_position = 3.0
            tracking_feet_position = 1.0
            tracking_max_foot_position = 0.5

            # 稳定、平滑和安全项。scales 是唯一 reward 权重来源。
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

        # C++ MuJoCo simulation_controller 契约值；不代表真机标定值。
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
        mass_matrix_target_acceleration_feedforward = True
        joint_impedance_outside_mass_matrix = True
        compensate_base_angular_velocity = True
        compensate_base_acceleration = False

    class domain_rand(ZGWTDanceCfg.domain_rand):
        # 完整机械臂已经提供真实质量和质心，不再叠加等效 payload/COM。
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_link_mass = False
        randomize_friction = False
        randomize_restitution = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = False
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
        run_name = "armstandv1"
        save_interval = 250
        max_iterations = 20000
        resume = True
        # ArmStandV1 已完成首轮训练；当前默认加载本阶段 checkpoint，续训时
        # 保留原迭代号和 Adam 状态。首次从 stage1v4 创建本阶段时的设置已结束。
        reset_iteration_on_load = False
        load_actor_only = False
        load_optimizer = True
        load_experiment_name = "ZGWT_DANCE_ARM"
        load_run = "Jul31_18-46-40_armstandv1"
        checkpoint = 20000


class ZGWTDanceArmStaticCfg(ZGWTDanceArmCfg):
    """ArmStandV2-Sim：每个 episode 固定一个仿真验证安全姿态。"""

    class arm(ZGWTDanceArmCfg.arm):
        pose_mode = "library"


class ZGWTDanceArmStaticCfgPPO(ZGWTDanceArmCfgPPO):
    class runner(ZGWTDanceArmCfgPPO.runner):
        experiment_name = "ZGWT_DANCE_ARM"
        run_name = "armstandv2_static"
        # 人工完成 ArmStandV1 后，从该实验目录的最新 checkpoint 进入 V2。
        load_experiment_name = "ZGWT_DANCE_ARM"
        load_run = -1
        checkpoint = -1
        # 新阶段只继承网络权重，从第 0 代开始，并重置 Adam。
        reset_iteration_on_load = True
        load_optimizer = False
