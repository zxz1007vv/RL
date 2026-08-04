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

    # 完整机械臂沿用相同四类语义，只提高带臂阶段的姿态、高度和支撑权重。
    class rewards(ZGWTDanceCfg.rewards):
        class scales:
            tracking_body_roll = 6.5
            tracking_body_pitch = 6.5
            tracking_body_yaw = 6.0
            tracking_body_height = 9.0

            tracking_lin_vx = 4.0
            tracking_lin_vy = 4.0
            body_stability = 3.0
            stance_coordination = 4.0
            support_stability = 4.0

            action_rate = -0.01

            collision = -1.0
            dof_pos_limits = -2.0
            torque_limits = -0.05
            severe_wheel_park = -4.0
            severe_support_loss = -4.0
            base_height_too_low = -4.0
            excessive_tilt = -4.0
            termination = -10.0

    class arm:
        joint_names = [f"ROBOT_ARM_JOINT{i}" for i in range(1, 7)]
        home = [0.0] * 6
        position_lower = [-2.618, 0.0, -2.9671, -1.57, -1.57, -6.28]
        position_upper = [2.618, 3.14, 0.0, 1.57, 1.57, 6.28]

        # "home" 为固定 Home 站立阶段；"library" 为机械臂姿态库适应阶段。
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
        # 完整机械臂已经显式建模，不再把额外 payload 与 home 姿态等效 COM 绑定。
        correlate_payload_and_com = False
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


class ZGWTDanceArmPoseLibraryCfg(ZGWTDanceArmCfg):
    """机械臂姿态库适应阶段：每个 episode 固定抽取一个安全姿态。"""

    class arm(ZGWTDanceArmCfg.arm):
        pose_mode = "library"


class ZGWTDanceArmPoseLibraryCfgPPO(ZGWTDanceArmCfgPPO):
    class runner(ZGWTDanceArmCfgPPO.runner):
        experiment_name = "ZGWT_DANCE_ARM"
        run_name = "armstand_pose_library_aligned_v1"
        # 后续开始姿态库阶段时，应改为加载新完成的 armstand_aligned_v1。
        load_experiment_name = "ZGWT_DANCE_ARM"
        load_run = "Jul31_18-46-40_armstandv1"
        checkpoint = 20000
        # 新阶段只继承网络权重，从第 0 代开始，并重置 Adam。
        reset_iteration_on_load = True
        load_optimizer = False
