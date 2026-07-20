from .zgwt_config import ZGWTRoughCfg, ZGWTRoughCfgPPO


class ZGWTDanceCfg(ZGWTRoughCfg):
    """Stationary body-pose tracking task for ZGWT dance motions."""

    class env(ZGWTRoughCfg.env):
        # ang_vel(3) + gravity(3) + commands(6) + q(16) + qd(16) + actions(16)
        num_one_step_observations = 3 + 3 + 6 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        # actor observation + base velocity + disturbance + height scan + contacts
        num_one_step_privileged_obs = (
            num_one_step_observations + 3 + 3 + 11 * 17 + 12
        )
        num_privileged_obs = num_one_step_privileged_obs

    class terrain(ZGWTRoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        max_init_terrain_level = 0

    class normalization(ZGWTRoughCfg.normalization):
        # action_scale=0.25, so this caps joint-target offsets at +/-0.75 rad.
        clip_actions = 3.0

    class control(ZGWTRoughCfg.control):
        wheel_park_damping = 1.5

    class commands(ZGWTRoughCfg.commands):
        # [vx, vy, yaw_rate, body_roll, body_pitch, body_height]
        num_commands = 6
        curriculum = False
        heading_command = False
        resampling_time = 10.0
        transition_time = 0.25
        curriculum_time = 1800.0
        neutral_pose_prob = 0.30
        command_scales = [2.0, 2.0, 0.25, 1.0, 1.0, 2.0]

        class ranges:
            # The velocity commands remain exactly zero in this task.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            body_roll = [-0.30, 0.30]
            body_pitch = [-0.28, 0.28]
            body_height = [0.42, 0.59]

        class initial_ranges:
            body_roll = [-0.08, 0.08]
            body_pitch = [-0.08, 0.08]
            body_height = [0.49, 0.55]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 10.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.08, 0.08]
        randomize_friction = True
        randomize_motor_strength = True
        randomize_kp = True
        randomize_kd = True
        randomize_initial_joint_pos = True
        disturbance = True
        push_robots = True
        delay = True

    class rewards(ZGWTRoughCfg.rewards):
        class scales:
            # Command tracking.
            tracking_body_orientation = 6.0   #roll pitch  4.0
            tracking_body_height = 3.0
            tracking_lin_vx = 1.5  #命令设置为0，跟踪奖励高反而不动
            tracking_lin_vy = 1.5
            tracking_ang_vel = 1.0

            # Keep the robot at its spawn point and keep the wheels parked.
            base_position_drift = 0.0
            support_center_drift = -10.0
            feet_horizontal_motion = -0.2
            base_stand_still = -2.0
            wheel_stand_still = -0.5
            wheel_vel_stand_still = -2.0e-3

            # Safety and smoothness. Fixed-level orientation, height, and joint
            # default rewards are deliberately absent because they oppose dance.
            collision = -1.0
            feet_contact = -0.4
            feet_stumble = -0.1
            action_rate = -0.003
            action_smoothness = -0.001
            torque_rate = -3.0e-7
            torques = -8.0e-6
            dof_vel = -1.0e-7
            dof_acc = -1.0e-8
            # Recover a symmetric nominal stance when pose commands are neutral.
            # The reward fades out for large dance commands in the robot class.
            neutral_joint_pose = -0.75
            lateral_leg_symmetry = -1.0
            lateral_foot_alignment = -1.0
            dof_pos_limits = -2.0
            torque_limits = -0.1

        only_positive_rewards = False
        orientation_tracking_sigma = 0.06
        height_tracking_sigma = 0.01
        neutral_orientation_sigma = 0.01
        neutral_height_sigma = 0.0025
        lateral_symmetry_roll_allowance = 2.0
        default_body_height = 0.54
        termination_tilt = 0.55
        termination_min_height = 0.32


class ZGWTDanceCfgPPO(ZGWTRoughCfgPPO):
    class policy(ZGWTRoughCfgPPO.policy):
        init_noise_std = 0.35

    class algorithm(ZGWTRoughCfgPPO.algorithm):
        learning_rate = 1.0e-4
        schedule = "adaptive"
        desired_kl = 0.01
        entropy_coef = 0.001
        clip_param = 0.15
        max_grad_norm = 0.5
        num_learning_epochs = 3

    class runner(ZGWTRoughCfgPPO.runner):
        experiment_name = "ZGWT_DANCE"
        run_name = "pose_tracking_fast_anchored_18.v1"
        resume = False
        load_run = -1
        checkpoint = -1
