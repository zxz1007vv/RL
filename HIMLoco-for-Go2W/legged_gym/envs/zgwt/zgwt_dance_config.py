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

    class commands(ZGWTRoughCfg.commands):
        # [vx, vy, yaw_rate, body_roll, body_pitch, body_height]
        num_commands = 6
        curriculum = False
        heading_command = False
        resampling_time = 3.0
        transition_time = 0.8
        curriculum_time = 600.0
        command_scales = [2.0, 2.0, 0.25, 1.0, 1.0, 2.0]

        class ranges:
            # The velocity commands remain exactly zero in this task.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            body_roll = [-0.25, 0.25]
            body_pitch = [-0.25, 0.25]
            body_height = [0.40, 0.58]

        class initial_ranges:
            body_roll = [-0.10, 0.10]
            body_pitch = [-0.10, 0.10]
            body_height = [0.48, 0.56]

    class domain_rand(ZGWTRoughCfg.domain_rand):
        # First learn the pose task cleanly. These can be restored gradually for
        # the robustness/fine-tuning stage.
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

    class rewards(ZGWTRoughCfg.rewards):
        class scales:
            # Command tracking.
            tracking_body_orientation = 4.0   #roll pitch
            tracking_body_height = 3.0
            tracking_lin_vx = 1.5  #命令设置为0，跟踪奖励高反而不动
            tracking_lin_vy = 1.5
            tracking_ang_vel = 1.0

            # Keep the robot at its spawn point and keep the wheels parked.
            base_position_drift = -3.0
            base_stand_still = -1.0
            wheel_stand_still = -0.5
            wheel_vel_stand_still = -2.0e-4

            # Safety and smoothness. Fixed-level orientation, height, and joint
            # default rewards are deliberately absent because they oppose dance.
            collision = -1.0
            feet_contact = -0.15
            feet_stumble = -0.1
            action_rate = -0.003
            action_smoothness = -0.001
            torque_rate = -3.0e-7
            torques = -8.0e-6
            dof_vel = -1.0e-7
            dof_acc = -1.0e-8
            dof_pos_limits = -2.0
            torque_limits = -0.1

        only_positive_rewards = False
        orientation_tracking_sigma = 0.08
        height_tracking_sigma = 0.01
        default_body_height = 0.54
        termination_tilt = 0.80
        termination_min_height = 0.25


class ZGWTDanceCfgPPO(ZGWTRoughCfgPPO):
    class runner(ZGWTRoughCfgPPO.runner):
        experiment_name = "ZGWT_DANCE"
        run_name = "pose_tracking_v1"
        resume = False
        load_run = -1
        checkpoint = -1

