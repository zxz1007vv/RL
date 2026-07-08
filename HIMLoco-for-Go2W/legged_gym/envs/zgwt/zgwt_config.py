from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class ZGWTRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = (
            num_one_step_observations + 3 + 3 + 11 * 17 + 12
        )
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 16

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"   #地形网格类型 trimesh 三角网格；plane 平面；hightfield 高度场
        curriculum = True     #地形学习开关
        max_init_terrain_level = 0
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_proportions = [0.4, 0.2, 0.2, 0.1, 0.1] # 地形类型比例 [光滑坡, 粗糙坡, 上楼梯, 下楼梯, 随机离散地形]

    class commands(LeggedRobotCfg.commands):
        curriculum = True    #命令学习开关
        max_curriculum = 1.5    #命令学习最大课程值
        num_commands = 4        #命令维度
        resampling_time = 10.0    #命令重采样时间
        heading_command = False    #是否使用航向命令

        class ranges:  #初始命令范围，课程设置的最大值
            lin_vel_x = [-1, 1]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]   #初始位置，出生位置，比期望的基座高度0.54略高即可；
        default_joint_angles = {
            # "FAR_ABAD_JOINT": 0.0,
            # "FAR_HIP_JOINT": 0.8,
            # "FAR_KNEE_JOINT": -1.5,
            # "FAR_FOOT_JOINT": 0.0,
            # "FBL_ABAD_JOINT": 0.0,
            # "FBL_HIP_JOINT": 0.8,
            # "FBL_KNEE_JOINT": -1.5,
            # "FBL_FOOT_JOINT": 0.0,
            # "RAR_ABAD_JOINT": 0.0,
            # "RAR_HIP_JOINT": 0.8,
            # "RAR_KNEE_JOINT": -1.5,
            # "RAR_FOOT_JOINT": 0.0,
            # "RBL_ABAD_JOINT": 0.0,
            # "RBL_HIP_JOINT": 0.8,
            # "RBL_KNEE_JOINT": -1.5,
            # "RBL_FOOT_JOINT": 0.0,

            #对姿 狗
            'FBL_ABAD_JOINT': -0.0,
            'FAR_ABAD_JOINT': 0.0,
            'RBL_ABAD_JOINT': -0.0,
            'RAR_ABAD_JOINT': 0.0,

            'FBL_HIP_JOINT': 0.6,
            'FAR_HIP_JOINT': 0.6,
            'RBL_HIP_JOINT': -0.6,
            'RAR_HIP_JOINT': -0.6,

            'FBL_KNEE_JOINT': -1.2,
            'FAR_KNEE_JOINT': -1.2,
            'RBL_KNEE_JOINT': 1.2,
            'RAR_KNEE_JOINT': 1.2,

            'FBL_FOOT_JOINT': 0.0,
            'FAR_FOOT_JOINT': 0.0,
            'RBL_FOOT_JOINT': 0.0,
            'RAR_FOOT_JOINT': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # control_type = "P"
        # stiffness = {
        #     "ABAD_JOINT": 40.0,
        #     "HIP_JOINT": 40.0,
        #     "KNEE_JOINT": 40.0,
        #     "FOOT_JOINT": 0.0,
        # }
        # damping = {
        #     "ABAD_JOINT": 1.0,
        #     "HIP_JOINT": 1.0,
        #     "KNEE_JOINT": 1.0,
        #     "FOOT_JOINT": 0.5,
        # }

        control_type = 'P'
        stiffness = {
        'FBL_ABAD_JOINT': 90,
        'FAR_ABAD_JOINT': 90,
        'RBL_ABAD_JOINT': 90,
        'RAR_ABAD_JOINT': 90,

        'FBL_HIP_JOINT': 120,
        'FAR_HIP_JOINT': 120,
        'RBL_HIP_JOINT': 120,
        'RAR_HIP_JOINT': 120,

        'FBL_KNEE_JOINT': 120,
        'FAR_KNEE_JOINT': 120,
        'RBL_KNEE_JOINT': 120,
        'RAR_KNEE_JOINT': 120,

        'FBL_FOOT_JOINT': 60,
        'FAR_FOOT_JOINT': 60,
        'RBL_FOOT_JOINT': 60,
        'RAR_FOOT_JOINT': 60,
        }
        damping = {
        'FBL_ABAD_JOINT': 1,
        'FAR_ABAD_JOINT': 1,
        'RBL_ABAD_JOINT': 1,
        'RAR_ABAD_JOINT': 1,

        'FBL_HIP_JOINT': 1,
        'FAR_HIP_JOINT': 1,
        'RBL_HIP_JOINT': 1,
        'RAR_HIP_JOINT': 1,

        'FBL_KNEE_JOINT': 1,
        'FAR_KNEE_JOINT': 1,
        'RBL_KNEE_JOINT': 1,
        'RAR_KNEE_JOINT': 1,

        'FBL_FOOT_JOINT': 0.2,
        'FAR_FOOT_JOINT': 0.2,
        'RBL_FOOT_JOINT': 0.2,
        'RAR_FOOT_JOINT': 0.2,
    
        }


        action_scale = 0.25  #动作尺度
        vel_scale = 10.0     #速度尺度
        decimation = 5        #采样间隔
        wheel_speed = 1       #轮速

    class sim(LeggedRobotCfg.sim):  #对应真机控制频率
        dt = 0.002

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/zgwt/urdf/zgwt.urdf"
        name = "zgwt"
        foot_name = "FOOT_LINK"
        wheel_name = ["FOOT_JOINT"]
        penalize_contacts_on = ["ABAD_LINK", "HIP_LINK", "KNEE_LINK", "BASE_LINK"]
        terminate_after_contacts_on = ["BASE_LINK"]
        priviledge_contacts_on = [
            "ABAD_LINK",
            "HIP_LINK",
            "KNEE_LINK",
            "BASE_LINK",
        ]
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False #可视化坐标翻转

    class domain_rand(LeggedRobotCfg.domain_rand):   #域随机化配置
        randomize_payload_mass = True
        randomize_com_displacement = True
        randomize_friction = True
        randomize_motor_strength = True
        randomize_kp = True
        randomize_kd = True
        randomize_initial_joint_pos = True
        disturbance = True
        push_robots = True
        delay = True

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # 基本姿态：先让机身高度、姿态和默认关节姿态稳定下来。
            orientation = -1.0
            base_height = -3.0
            hip_default = -0.15

            # 稳定：抑制上下跳、roll/pitch 角速度、动作突变和过大力矩。
            lin_vel_z = -0.2
            ang_vel_xy = -0.1
            action_rate = -0.003
            dof_vel = -1.0e-7
            dof_vel_wheel = -1.0e-5
            dof_acc = -1.0e-8
            collision = -1.0

            # Low power：降低力矩、机械功率和控制突变，让步态更省力、更顺。
            torques = -1.0e-5
            mechanical_power = -5.0e-5
            diagonal_power_balance = -1.0e-4
            torque_rate = -5.0e-7
            action_smoothness = -0.0015

            # Tracking：vx/vy 分开调，避免 xy 合并项和单轴项重复计分。
            tracking_lin_vel = 0.0
            tracking_lin_vx = 1.5
            tracking_lin_vy = 2.0
            tracking_ang_vel = 1.0

            # 步态/接触：轮足车尽量保持轮足接地，减少启停时扬腿和重心晃动。
            stand_still = -0.2
            run_still = -0.02
            feet_contact = -0.15
            feet_stumble = -0.1

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.54  #大狗的基座高度目标
        max_contact_force = 100.0


class ZGWTRoughCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.6

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3.0e-4
        schedule = "fixed"
        desired_kl = 0.02

    class runner(LeggedRobotCfgPPO.runner):
        save_interval = 1000   #保存间隔
        num_steps_per_env = 48   #每个环境的步数
        max_iterations = 20000   #最大迭代次数
        experiment_name = "ZGWT"
        run_name = "V1"    #运行名称
        resume = None
        load_run = -1
        checkpoint = -1
        resume_path = None
