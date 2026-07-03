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
        mesh_type = "trimesh"   #地形网格类型 trimesh 三角网格；plane 平面；hightfield 高度场
        curriculum = False     #地形学习开关
        max_init_terrain_level = 0
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]

    class commands(LeggedRobotCfg.commands):
        curriculum = False    #命令学习开关
        max_curriculum = 1.0    #命令学习最大课程值
        num_commands = 4        #命令维度
        resampling_time = 10.0    #命令重采样时间
        heading_command = True    #是否使用航向命令

        class ranges:
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = {
            "FAR_ABAD_JOINT": 0.0,
            "FAR_HIP_JOINT": 0.8,
            "FAR_KNEE_JOINT": -1.5,
            "FAR_FOOT_JOINT": 0.0,
            "FBL_ABAD_JOINT": 0.0,
            "FBL_HIP_JOINT": 0.8,
            "FBL_KNEE_JOINT": -1.5,
            "FBL_FOOT_JOINT": 0.0,
            "RAR_ABAD_JOINT": 0.0,
            "RAR_HIP_JOINT": 0.8,
            "RAR_KNEE_JOINT": -1.5,
            "RAR_FOOT_JOINT": 0.0,
            "RBL_ABAD_JOINT": 0.0,
            "RBL_HIP_JOINT": 0.8,
            "RBL_KNEE_JOINT": -1.5,
            "RBL_FOOT_JOINT": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "ABAD_JOINT": 40.0,
            "HIP_JOINT": 40.0,
            "KNEE_JOINT": 40.0,
            "FOOT_JOINT": 0.0,
        }
        damping = {
            "ABAD_JOINT": 1.0,
            "HIP_JOINT": 1.0,
            "KNEE_JOINT": 1.0,
            "FOOT_JOINT": 0.5,
        }
        action_scale = 0.15
        vel_scale = 10.0
        decimation = 4
        wheel_speed = 1

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
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -10.0
            hip_default = -0.5
            stand_still = -0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.03
            torques = -5.0e-4
            dof_vel = -2.0e-7
            dof_acc = -2.0e-7
            run_still = -0.05

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.4
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
        save_interval = 100   #保存间隔
        num_steps_per_env = 48   #每个环境的步数
        max_iterations = 10000   #最大迭代次数
        experiment_name = "ZGWT"
        run_name = "stable_start"    #运行名称
        resume = None
        load_run = -1
        checkpoint = -1
        resume_path = None
