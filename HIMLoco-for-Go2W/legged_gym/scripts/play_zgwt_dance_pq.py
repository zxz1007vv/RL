"""Phase-A baseline/shadow/active evaluation for ``zgwt_dance_pq``.

Examples:
  python legged_gym/scripts/play_zgwt_dance_pq.py --pq_mode off --payload_mass 4
  python legged_gym/scripts/play_zgwt_dance_pq.py --pq_mode shadow --payload_mass 4
  python legged_gym/scripts/play_zgwt_dance_pq.py --pq_mode active --payload_mass 4
"""

import argparse
import os
import sys

import isaacgym  # noqa: F401 - must be imported before torch for Isaac Gym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - performs task registration
from legged_gym.utils import export_policy_as_jit, get_args, task_registry


def _parse_pq_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pq_mode", choices=("off", "shadow", "active"), default="shadow"
    )
    parser.add_argument("--payload_mass", type=float, default=0.0)
    parser.add_argument("--com_x", type=float, default=0.0)
    parser.add_argument("--com_y", type=float, default=0.0)
    parser.add_argument("--com_z", type=float, default=0.0)
    parser.add_argument(
        "--test", choices=("neutral", "stage1"), default="neutral"
    )
    parser.add_argument("--body_yaw", type=float, default=0.0)
    parser.add_argument("--body_roll", type=float, default=0.0)
    parser.add_argument("--body_pitch", type=float, default=0.0)
    parser.add_argument("--body_height", type=float, default=0.54)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--load_step_time", type=float)
    parser.add_argument("--load_step_payload", type=float)
    parser.add_argument("--export_pq_features", type=str)
    parser.add_argument(
        "--export_actor",
        action="store_true",
        help="Export the unchanged estimator+actor JIT beside the PQ weights.",
    )
    parser.add_argument(
        "--asset_file",
        type=str,
        help="Optional URDF override, useful for headless no-visual smoke tests.",
    )
    pq_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return pq_args


def _mean(value):
    return float(torch.mean(value).item())


def play_pq(args, pq_args):
    if args.task != "zgwt_dance_pq":
        raise ValueError("play_zgwt_dance_pq.py requires --task zgwt_dance_pq")
    if pq_args.payload_mass < 0.0:
        raise ValueError("payload_mass must be non-negative")
    if pq_args.duration <= 0.0 or pq_args.warmup < 0.0:
        raise ValueError("duration must be positive and warmup non-negative")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = min(
        env_cfg.env.num_envs, args.num_envs if args.num_envs is not None else 50
    )
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_initial_base_velocity = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.noise.add_noise = False
    env_cfg.pq.enabled = pq_args.pq_mode != "off"
    env_cfg.pq.shadow_mode = pq_args.pq_mode == "shadow"
    env_cfg.pq.fixed_payload_mass = pq_args.payload_mass
    env_cfg.pq.fixed_com_offset = [
        pq_args.com_x,
        pq_args.com_y,
        pq_args.com_z,
    ]
    if pq_args.asset_file:
        env_cfg.asset.file = os.path.abspath(pq_args.asset_file)

    if pq_args.test == "stage1":
        env_cfg.commands.ranges.body_yaw = [-0.025, 0.025]
        env_cfg.commands.ranges.body_roll = [-0.08, 0.08]
        env_cfg.commands.ranges.body_pitch = [-0.08, 0.08]
        env_cfg.commands.ranges.body_height = [0.49, 0.54]

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.set_pq_mode(pq_args.pq_mode)

    train_cfg.runner.resume = True
    train_cfg.runner.reset_iteration_on_load = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    policy = runner.get_inference_policy(device=env.device)

    if pq_args.export_actor:
        actor_export_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "ZGWT_DANCE_PQ",
            "exported",
            "policies",
        )
        actor_path = export_policy_as_jit(
            runner.alg.actor_critic,
            actor_export_dir,
            filename="actor.pt",
        )
        print("Exported unchanged estimator+actor JIT:", actor_path)

    if pq_args.export_pq_features:
        feature_path = os.path.abspath(pq_args.export_pq_features)
    else:
        feature_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "ZGWT_DANCE_PQ",
            "exported",
            "pq_feature_weights.pt",
        )
    print("Exported fixed PQ features:", env.export_pq_features(feature_path))

    if pq_args.test == "neutral":
        command = (0.0, 0.0, 0.0, 0.54)
    else:
        command = (
            pq_args.body_yaw,
            pq_args.body_roll,
            pq_args.body_pitch,
            pq_args.body_height,
        )

    def set_command():
        yaw, roll, pitch, height = command
        env.commands[:, :2] = 0.0
        env.yaw_command_targets[:] = yaw
        env.pose_command_targets[:, 0] = roll
        env.pose_command_targets[:, 1] = pitch
        env.pose_command_targets[:, 2] = height

    set_command()
    env.compute_observations()
    obs = env.get_observations()
    initial_base_xy = env.root_states[:, :2].clone()
    initial_support_xy = torch.mean(env.feet_pos[:, :, :2], dim=1).clone()
    initial_feet_xy = env.feet_pos[:, :, :2].clone()

    names = (
        "roll_error",
        "pitch_error",
        "yaw_error",
        "height_error",
        "base_xy_drift",
        "support_xy_drift",
        "feet_xy_drift",
        "max_wheel_drift",
        "torque_saturation",
        "contact_loss",
        "fall",
    )
    sums = {name: 0.0 for name in names}
    samples = 0
    total_steps = int(round((pq_args.warmup + pq_args.duration) / env.dt))
    warmup_steps = int(round(pq_args.warmup / env.dt))
    load_step_done = False

    for step in range(total_steps):
        set_command()
        if (
            not load_step_done
            and pq_args.load_step_time is not None
            and step * env.dt >= pq_args.load_step_time
        ):
            stepped_payload = (
                pq_args.load_step_payload
                if pq_args.load_step_payload is not None
                else pq_args.payload_mass
            )
            env.set_payload_mass_and_com(
                stepped_payload,
                [pq_args.com_x, pq_args.com_y, pq_args.com_z],
            )
            load_step_done = True
            print(
                f"Applied true rigid-body payload step at {step * env.dt:.3f}s: "
                f"{stepped_payload:.3f}kg"
            )

        with torch.inference_mode():
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
        if step < warmup_steps:
            continue

        roll, pitch = env._current_roll_pitch()
        height = env._current_base_height()
        wheel_drift = torch.norm(
            env.feet_pos[:, :, :2] - initial_feet_xy, dim=2
        )
        values = {
            "roll_error": torch.abs(env.commands[:, 3] - roll),
            "pitch_error": torch.abs(env.commands[:, 4] - pitch),
            "yaw_error": torch.abs(env._body_yaw_error()),
            "height_error": torch.abs(env.commands[:, 5] - height),
            "base_xy_drift": torch.norm(
                env.root_states[:, :2] - initial_base_xy, dim=1
            ),
            "support_xy_drift": torch.norm(
                torch.mean(env.feet_pos[:, :, :2], dim=1)
                - initial_support_xy,
                dim=1,
            ),
            "feet_xy_drift": torch.mean(wheel_drift, dim=1),
            "max_wheel_drift": torch.max(wheel_drift, dim=1).values,
            "torque_saturation": torch.mean(
                (
                    torch.abs(env.torques) >= 0.99 * env.torque_limits
                ).float(),
                dim=1,
            ),
            "contact_loss": torch.mean(
                (
                    env.contact_forces[:, env.feet_indices, 2] < 5.0
                ).float(),
                dim=1,
            ),
            "fall": dones.float(),
        }
        for name, value in values.items():
            sums[name] += _mean(value)
        samples += 1

    divisor = max(samples, 1)
    print("\n================ ZGWT DANCE PQ EVALUATION ================")
    print(
        f"mode={pq_args.pq_mode}, payload={pq_args.payload_mass:.3f}kg, "
        f"COM=({pq_args.com_x:+.3f},{pq_args.com_y:+.3f},{pq_args.com_z:+.3f})m"
    )
    for name in names:
        print(f"{name}: {sums[name] / divisor:.8f}")

    all_envs = torch.arange(env.num_envs, device=env.device)
    pq_metrics = env.pq_compensator.episode_metrics(all_envs)
    print("\nPQ diagnostics")
    for name in sorted(pq_metrics):
        value = pq_metrics[name]
        print(f"{name}: {float(value.item()):.8f}")
    print("==========================================================")


if __name__ == "__main__":
    pq_cli_args = _parse_pq_args()
    core_args = get_args()
    if core_args.task == "aliengo":
        core_args.task = "zgwt_dance_pq"
    play_pq(core_args, pq_cli_args)
