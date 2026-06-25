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
        mesh_type = "trimesh"
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_proportions = [0.1, 0.1, 0.35, 0.2, 0.25]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
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
        action_scale = 0.25
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
        flip_visual_attachments = True

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
            action_rate = -0.01
            torques = -5.0e-4
            dof_vel = -1.0e-7
            dof_acc = -1.0e-7
            run_still = -0.05

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.4
        max_contact_force = 100.0


class ZGWTRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        save_interval = 1000
        num_steps_per_env = 48
        max_iterations = 20000
        experiment_name = "ZGWT"
        run_name = ""
        resume = None
        load_run = -1
        checkpoint = -1
        resume_path = None
