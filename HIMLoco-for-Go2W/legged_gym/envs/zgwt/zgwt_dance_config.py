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
        transition_time = 0.5
        # Independent hard limit for body-yaw target changes. A full reversal
        # from +0.10 to -0.10 rad therefore takes at least 0.67 seconds.
        yaw_slew_rate = 0.30
        # Give the wider/faster command set enough time to unfold smoothly.
        curriculum_time = 2400.0
        # Stage 1 matches the usual joystick operation: height is isolated while
        # roll/pitch/yaw can mix. Stage 2 adds the short mixed states produced by
        # joystick filtering and command hand-over. Both lists use this order:
        # neutral, H, R, P, Y, RP, RY, PY, RPY,
        # HR, HP, HY, HRP, HRY, HPY, HRPY.
        initial_mode_probabilities = [
            0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.075, 0.075, 0.05,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        ]
        final_mode_probabilities = [
            0.10, 0.10, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.10,
            0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.04,
        ]
        mixed_height_curriculum_start = 0.35
        mixed_height_curriculum_end = 0.75
        # Height is normalized by max(|0.40-0.54|, |0.55-0.54|)=0.14 m.
        command_scales = [2.0, 2.0, 1.0, 1.0, 1.0, 7.142857]

        class ranges:
            # Linear velocity remains zero. body_yaw is a bounded angle relative
            # to the heading captured at episode reset, not a continuous turn rate.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            body_yaw = [-0.10, 0.10]
            body_roll = [-0.26, 0.26]
            body_pitch = [-0.26, 0.26]
            body_height = [0.40, 0.55]

        class initial_ranges:
            body_yaw = [-0.02, 0.02]
            body_roll = [-0.08, 0.08]
            body_pitch = [-0.08, 0.08]
            body_height = [0.49, 0.55]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        # Stage 1 learns the nominal fixed-support pose task first. Re-enable
        # narrow randomization only after active command errors have converged.
        randomize_payload_mass = False
        payload_mass_range = [0.0, 12.0]
        randomize_com_displacement = False
        com_displacement_range = [-0.05, 0.05]
        randomize_link_mass = False
        link_mass_range = [0.95, 1.05]
        randomize_friction = False
        friction_range = [0.70, 1.30]
        randomize_restitution = False
        restitution_range = [0.0, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.85, 1.15]
        randomize_kp = False
        kp_range = [0.85, 1.15]
        randomize_kd = False
        kd_range = [0.80, 1.20]
        randomize_initial_joint_pos = False
        # Keep recovery diversity without starting from visibly crooked legs.
        initial_joint_pos_range = [0.90, 1.10]
        randomize_initial_base_velocity = False
        disturbance = False
        disturbance_range = [-10.0, 10.0]
        disturbance_interval = 8
        push_robots = False
        push_interval_s = 12
        max_push_vel_xy = 1.2
        delay = False

    class noise(ZGWTRoughCfg.noise):
        add_noise = False
        noise_level = 0.0

    class rewards(ZGWTRoughCfg.rewards):
        class scales:
            # Roll and pitch share one target-gravity reward. This avoids a
            # second "stay level" objective fighting the commanded body pose.
            tracking_body_orientation = 11.0
            tracking_body_yaw = 6.0
            tracking_body_height = 6.0
            # Hold objectives are gated by command-tracking convergence. Support
            # centre has priority over exact individual wheel-centre anchoring.
            tracking_lin_vx = 3.0
            tracking_lin_vy = 3.0
            tracking_support_position = 3.0
            tracking_feet_position = 1.0
            tracking_max_foot_position = 0.5

            # Minimal stability, smoothness, and safety terms.
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

        only_positive_rewards = False

        # First track the commanded body pose, then tighten stationary support
        # as tracking converges. Safety/smoothness costs use a floored multiplier.
        task_reward_scale = 28.0
        hold_gate_sigma = 0.50
        auxiliary_reward_sigma = 0.10
        auxiliary_reward_floor = 0.30

        # reward = exp(-error / sigma). Orientation error is measured between
        # actual and commanded projected gravity, not against a level body.
        orientation_tracking_sigma = 0.012
        yaw_tracking_sigma = 0.0025
        height_tracking_sigma = 0.0008

        # Used only by diagnostics and the explicitly neutral-task gate.
        active_orientation_threshold = 0.03
        active_yaw_threshold = 0.02
        active_height_threshold = 0.02

        # Soft support deadzones retain contact-solver tolerance. Wheel axle
        # position-hold remains enabled independently in the controller.
        feet_position_tracking_sigma = 0.003
        support_position_tracking_sigma = 0.001
        max_foot_position_tracking_sigma = 0.001
        feet_anchor_deadzone = 0.015
        support_anchor_deadzone = 0.010

        default_body_height = 0.54
        termination_tilt = 0.55
        termination_min_height = 0.32


class ZGWTDanceCfgPPO(ZGWTRoughCfgPPO):
    class policy(ZGWTRoughCfgPPO.policy):
        init_noise_std = 0.15

    class algorithm(ZGWTRoughCfgPPO.algorithm):
        learning_rate = 1.0e-4
        # Adaptive scheduling raised 1e-4 to 6.67e-3 around iteration 500 in
        # the failed run, after which value loss jumped above 1e3.
        schedule = "fixed"
        desired_kl = 0.01
        entropy_coef = 0.0
        min_action_std = 0.05
        max_action_std = 0.25
        clip_param = 0.15
        max_grad_norm = 0.5
        num_learning_epochs = 3

    class runner(ZGWTRoughCfgPPO.runner):
        experiment_name = "ZGWT_DANCE"
        run_name = "729v3_two_stage_height_mix_gated_hold"
        save_interval = 500
        resume = False
        load_run = -1
        checkpoint = -1
