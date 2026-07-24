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
        transition_time = 1.0
        # Independent hard limit for body-yaw target changes. A full reversal
        # from +0.10 to -0.10 rad therefore takes at least 1.33 seconds.
        yaw_slew_rate = 0.15
        # Give the wider/faster command set enough time to unfold smoothly.
        curriculum_time = 2400.0
        neutral_pose_prob = 0.30
        command_scales = [2.0, 2.0, 1.0, 1.0, 1.0, 2.0]

        class ranges:
            # Linear velocity remains zero. body_yaw is a bounded angle relative
            # to the heading captured at episode reset, not a continuous turn rate.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            body_yaw = [-0.10, 0.10]
            body_roll = [-0.34, 0.34]
            body_pitch = [-0.32, 0.32]
            body_height = [0.40, 0.55]

        class initial_ranges:
            body_yaw = [-0.02, 0.02]
            body_roll = [-0.08, 0.08]
            body_pitch = [-0.08, 0.08]
            body_height = [0.49, 0.55]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 12.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_link_mass = True
        link_mass_range = [0.95, 1.05]
        randomize_friction = True
        # Fixed-support body poses require enough grip. Very low friction taught
        # the old policy to satisfy yaw by translating the wheel contacts.
        friction_range = [0.70, 1.30]
        randomize_restitution = True
        restitution_range = [0.0, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]
        randomize_kp = True
        kp_range = [0.85, 1.15]
        randomize_kd = True
        kd_range = [0.80, 1.20]
        randomize_initial_joint_pos = False
        # Keep recovery diversity without starting from visibly crooked legs.
        initial_joint_pos_range = [0.90, 1.10]
        randomize_initial_base_velocity = False
        disturbance = True
        disturbance_range = [-10.0, 10.0]
        disturbance_interval = 8
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 1.2
        delay = True

    class noise(ZGWTRoughCfg.noise):
        add_noise = True
        noise_level = 1.15

    class rewards(ZGWTRoughCfg.rewards):
        class scales:
            # 跟踪
            tracking_body_orientation = 6.0   #roll pitch  4.0
            tracking_body_yaw = 3.0
            tracking_body_height = 3.0
            tracking_lin_vx = 1.5  #命令设置为0，跟踪奖励高反而不动
            tracking_lin_vy = 1.5
            tracking_ang_vel = 0.0

            # 固定支撑点位置
            base_position_drift = 0.0
            tracking_feet_position = 2.0
            support_center_drift = -15.0
            feet_position_drift = -20.0
            max_foot_position_drift = -25.0
            base_linear_motion = -2.5
            body_yaw_in_place = -10.0
            yaw_rate = -0.20
            feet_horizontal_motion = -0.5
            feet_air_horizontal_motion = -0.25
            base_stand_still = -2.0
            wheel_stand_still = -0.5
            wheel_vel_stand_still = -2.0e-3

            # 稳定性
            collision = -1.0
            feet_contact = -0.6
            feet_stumble = -0.1
            action_rate = -0.005
            action_smoothness = -0.0025
            torque_rate = -5.0e-7
            torques = -8.0e-6
            dof_vel = -1.0e-7
            dof_acc = -1.0e-8

            # 姿态对称性和关节限制
            neutral_joint_pose = -1.0
            lateral_leg_symmetry = -1.5
            lateral_foot_alignment = -1.5
            dof_pos_limits = -2.0
            torque_limits = -0.1

        only_positive_rewards = False
        orientation_tracking_sigma = 0.06
        yaw_tracking_sigma = 0.015
        feet_position_tracking_sigma = 0.003
        height_tracking_sigma = 0.01
        neutral_orientation_sigma = 0.01
        neutral_height_sigma = 0.0025
        lateral_symmetry_roll_allowance = 2.0
        lateral_symmetry_yaw_allowance = 0.75
        yaw_symmetry_gate_sigma = 0.01
        body_yaw_in_place_full_scale = 0.05
        default_body_height = 0.54
        termination_tilt = 0.55
        termination_min_height = 0.32


class ZGWTDanceCfgPPO(ZGWTRoughCfgPPO):
    class policy(ZGWTRoughCfgPPO.policy):
        init_noise_std = 0.25

    class algorithm(ZGWTRoughCfgPPO.algorithm):
        learning_rate = 1.0e-4
        # Adaptive scheduling raised 1e-4 to 6.67e-3 around iteration 500 in
        # the failed run, after which value loss jumped above 1e3.
        schedule = "fixed"
        desired_kl = 0.01
        entropy_coef = 0.001
        clip_param = 0.15
        max_grad_norm = 0.5
        num_learning_epochs = 3

    class runner(ZGWTRoughCfgPPO.runner):
        experiment_name = "ZGWT_DANCE"
        run_name = "724v4_fixed_lr_body_yaw"
        resume = False
        load_run = -1
        checkpoint = -1
