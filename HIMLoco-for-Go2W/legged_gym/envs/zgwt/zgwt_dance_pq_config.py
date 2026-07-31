"""Isolated load-stand configuration for the ZGWT P/Q experiment."""

from .zgwt_dance_config import ZGWTDanceCfg, ZGWTDanceCfgPPO


class ZGWTDancePQCfg(ZGWTDanceCfg):
    """Phase-A load stand task; actor/critic interfaces remain unchanged."""

    class commands(ZGWTDanceCfg.commands):
        mode_probabilities = [
            1.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            body_yaw = [0.0, 0.0]
            body_roll = [0.0, 0.0]
            body_pitch = [0.0, 0.0]
            body_height = [0.54, 0.54]

    class domain_rand(ZGWTDanceCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 8.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        com_displacement_is_offset = True

        randomize_link_mass = False
        randomize_friction = False
        randomize_restitution = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = False
        randomize_initial_base_velocity = False
        push_robots = False
        disturbance = False
        delay = False
        observation_delay = False

    class noise(ZGWTDanceCfg.noise):
        add_noise = False

    class pq:
        enabled = True
        shadow_mode = True
        compensate_wheels = False

        input_dim = 36
        hidden_dims = [64]
        feature_dim = 16
        activation = "tanh"
        feature_seed = 20260731
        append_constant_bias_feature = True

        joint_position_scale = 1.0
        joint_velocity_scale = 0.05
        joint_reference_scale = 1.0

        filter_time_constant = 0.05
        forgetting_rate = 5.0
        adaptation_gain = 1.0e-3
        weight_leak = 1.0e-4
        m0_diag = [1.0] * 12

        compensation_torque_ratio = 0.05
        compensation_ramp_time = 1.5
        max_weight_abs = 3.0
        max_weight_update_abs = 0.5

        freeze_during_ramp = True
        freeze_on_reset = True
        freeze_on_contact_loss = True
        freeze_on_torque_saturation = True
        freeze_on_excessive_tilt = True
        excessive_tilt_threshold = 0.35
        reset_weights_on_episode_reset = True

        use_spectral_regularization = False
        use_projection = False

        # Evaluation-only exact overrides. None keeps the training randomization.
        fixed_payload_mass = None
        fixed_com_offset = None


class ZGWTDancePQCfgPPO(ZGWTDanceCfgPPO):
    """Low-rate joint fine-tuning after Phase-A PQ validation succeeds."""

    class algorithm(ZGWTDanceCfgPPO.algorithm):
        learning_rate = 2.5e-5
        schedule = "fixed"
        clip_param = 0.08
        num_learning_epochs = 2
        max_grad_norm = 0.5
        entropy_coef = 0.0

    class runner(ZGWTDanceCfgPPO.runner):
        experiment_name = "ZGWT_DANCE_PQ"
        run_name = "pq_load_stand_v1"
        save_interval = 100
        max_iterations = 1000

        # Phase A does not run PPO updates: use play_zgwt_dance_pq.py first.
        # For Phase B, the unchanged 360-D observation and 16-D action make the
        # old actor/estimator shape-compatible. PQ changes the closed-loop
        # dynamics, so retain actor/estimator while resetting critic and both
        # optimizers. P/Q/W and filter states are runtime state and are not part
        # of PPO checkpoints.
        resume = True
        reset_iteration_on_load = True
        load_actor_only = True
        load_optimizer = False

        # Checkpoints are loaded from the original experiment while all new logs
        # and checkpoints remain isolated under logs/ZGWT_DANCE_PQ.
        load_experiment_name = "ZGWT_DANCE"
        load_run = "Jul30_14-54-33_stage1v2"
        checkpoint = 4750

