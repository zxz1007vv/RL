# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import itertools
import os
import re
import signal
import shutil

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import (
    Logger,
    export_policy_as_jit,
    get_args,
    get_load_path,
    task_registry,
)

import numpy as np
import torch


def play(
    args,
    x_vel=1.0,
    y_vel=0.0,
    body_yaw=0.0,
    body_roll=0.0,
    body_pitch=0.0,
    body_height=0.54,
    dance_trajectory=True,
    dance_ramp_time=3.0,
    dance_frequency=0.24,
    yaw_amplitude=None,
    roll_amplitude=None,
    pitch_amplitude=None,
    height_center=None,
    height_amplitude=None,
    evaluation_mode="neutral",
    evaluation_warmup_time=2.0,
    evaluation_duration=600.0,
    infinite_report_interval=10.0,
    verbose_pose_print=False,
):
    if evaluation_mode not in ("neutral", "command"):
        raise ValueError(
            "evaluation_mode must be either 'neutral' or 'command', "
            f"got {evaluation_mode!r}"
        )
    if evaluation_warmup_time < 0.0:
        raise ValueError("evaluation_warmup_time must be non-negative")
    if evaluation_duration is not None and evaluation_duration <= 0.0:
        raise ValueError("evaluation_duration must be positive")
    if infinite_report_interval <= 0.0:
        raise ValueError("infinite_report_interval must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)     #环境数量上限
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 0
    # Evaluation starts from a clean, symmetric stance. Training robustness
    # randomization should not obscure pose-tracking quality in play.
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_initial_base_velocity = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.noise.add_noise = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if evaluation_mode == "neutral":
        x_vel = 0.0
        y_vel = 0.0
        body_yaw = 0.0
        body_roll = 0.0
        body_pitch = 0.0
        body_height = float(env.cfg.rewards.default_body_height)
        dance_trajectory = False

    def command_center_and_amplitude(name):
        lower, upper = env.command_ranges[name]
        return 0.5 * (lower + upper), 0.5 * (upper - lower)

    if env.cfg.commands.num_commands >= 6:
        _, train_yaw_amplitude = command_center_and_amplitude("body_yaw")
        _, train_roll_amplitude = command_center_and_amplitude("body_roll")
        _, train_pitch_amplitude = command_center_and_amplitude("body_pitch")
        train_height_center, train_height_amplitude = command_center_and_amplitude(
            "body_height"
        )
        yaw_amplitude = (
            train_yaw_amplitude if yaw_amplitude is None else yaw_amplitude
        )
        roll_amplitude = (
            train_roll_amplitude if roll_amplitude is None else roll_amplitude
        )
        pitch_amplitude = (
            train_pitch_amplitude if pitch_amplitude is None else pitch_amplitude
        )
        height_center = (
            train_height_center if height_center is None else height_center
        )
        height_amplitude = (
            train_height_amplitude if height_amplitude is None else height_amplitude
        )

    def clip_command(value, name):
        lower, upper = env.command_ranges[name]
        return np.clip(value, lower, upper)

    def set_test_commands(time_s=0.0):
        if env.cfg.commands.num_commands >= 6:
            # The dedicated dance policy is stationary by construction.
            env.commands[:, :2] = 0.0
            pose_targets = torch.empty_like(env.commands[:, 3:6])
            if dance_trajectory:
                phase = 2.0 * np.pi * dance_frequency * time_s
                ramp = min(time_s / max(dance_ramp_time, env.dt), 1.0)
                yaw_target = ramp * yaw_amplitude * np.sin(phase)
                pose_targets[:, 0] = ramp * roll_amplitude * np.sin(phase)
                pose_targets[:, 1] = ramp * pitch_amplitude * np.sin(
                    phase + np.pi / 2.0
                )
                dance_height = height_center + height_amplitude * np.sin(
                    phase * 0.5
                )
                pose_targets[:, 2] = body_height + ramp * (
                    dance_height - body_height
                )
            else:
                yaw_target = body_yaw
                pose_targets[:, 0] = body_roll
                pose_targets[:, 1] = body_pitch
                pose_targets[:, 2] = body_height
            yaw_target = clip_command(yaw_target, "body_yaw")
            pose_targets[:, 0].clamp_(*env.command_ranges["body_roll"])
            pose_targets[:, 1].clamp_(*env.command_ranges["body_pitch"])
            pose_targets[:, 2].clamp_(*env.command_ranges["body_height"])
            if hasattr(env, "yaw_command_targets"):
                # Let the environment apply the same transition filter as training.
                env.yaw_command_targets[:] = yaw_target
            else:
                env.commands[:, 2] = yaw_target
            if hasattr(env, "pose_command_targets"):
                # Let the environment apply the same transition filter as training.
                env.pose_command_targets[:] = pose_targets
            else:
                env.commands[:, 3:6] = pose_targets
        else:
            env.commands[:, 0] = x_vel
            env.commands[:, 1] = y_vel
            env.commands[:, 2] = body_yaw

    set_test_commands(0.0)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # Evaluation/export does not create a new training stage. Preserve the
    # checkpoint iteration for truthful diagnostics; train.py still follows the
    # stage-specific reset_iteration_on_load setting from the task config.
    train_cfg.runner.reset_iteration_on_load = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # The runner resets the environment while it is created, so restore the
    # requested test command and refresh the first policy observation.
    set_test_commands(0.0)
    env.compute_observations()
    obs = env.get_observations()

    # Evaluation references are captured after the runner reset and after the
    # requested command has been restored, exactly once for the whole run.
    initial_base_xy = env.root_states[:, :2].clone()
    initial_support_xy = torch.mean(env.feet_pos[:, :, :2], dim=1).clone()
    initial_feet_xy = env.feet_pos[:, :, :2].clone()

    leg_joint_mask = torch.ones(
        env.num_dof, dtype=torch.bool, device=env.device
    )
    leg_joint_mask[env.wheel_indices] = False
    leg_joint_indices = leg_joint_mask.nonzero(as_tuple=False).flatten()

    metric_sums = {
        name: torch.zeros((), dtype=torch.float, device=env.device)
        for name in (
            "roll_error",
            "pitch_error",
            "yaw_error",
            "height_error",
            "base_xy_drift",
            "support_xy_drift",
            "feet_xy_drift",
            "leg_joint_error",
            "contact_loss",
            "torque_saturation",
            "action_saturation",
            "reset_count",
        )
    }
    metric_maxima = {
        name: torch.zeros((), dtype=torch.float, device=env.device)
        for name in (
            "roll_error",
            "pitch_error",
            "yaw_error",
            "height_error",
            "base_xy_drift",
            "support_xy_drift",
            "wheel_xy_drift",
            "leg_joint_error",
        )
    }
    evaluated_steps = 0


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        checkpoint_root = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name
        )
        checkpoint_path = get_load_path(
            checkpoint_root,
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        checkpoint_label = checkpoint_name.split('model_', 1)[-1]
        run_label = re.sub(
            r'[^A-Za-z0-9_.-]+', '_', str(train_cfg.runner.run_name)
        ).strip('_.-') or 'unnamed_run'
        export_filename = f'policy_{run_label}_ckpt{checkpoint_label}.pt'
        exported_path = export_policy_as_jit(
            ppo_runner.alg.actor_critic, path, filename=export_filename
        )
        latest_path = os.path.join(path, 'policy.pt')
        if os.path.abspath(exported_path) != os.path.abspath(latest_path):
            shutil.copy2(exported_path, latest_path)
        print('Loaded checkpoint: ', checkpoint_path)
        print('Exported versioned policy: ', exported_path)
        print('Updated deployment policy: ', latest_path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    warmup_steps = max(0, int(round(evaluation_warmup_time / env.dt)))
    stop_requested = False
    previous_sigint_handler = None
    report_steps = None

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\n收到 Ctrl+C，正在结束当前仿真步并输出完整误差汇总……")

    if evaluation_duration is None:
        step_indices = itertools.count()
        report_steps = max(1, int(round(infinite_report_interval / env.dt)))
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_stop)
        print(
            "Play 将持续运行；每 "
            f"{infinite_report_interval:.1f} 秒输出一次累计误差，"
            "按 Ctrl+C 停止并输出完整汇总。"
        )
    else:
        evaluation_steps = max(1, int(round(evaluation_duration / env.dt)))
        total_steps = warmup_steps + evaluation_steps
        step_indices = range(total_steps)

    for i in step_indices:
        set_test_commands(i * env.dt)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        if (
            verbose_pose_print
            and env.cfg.commands.num_commands >= 6
            and i % 50 == 0
        ):
            actual_roll, actual_pitch = env._current_roll_pitch()
            actual_height = env._current_base_height()

            cmd_roll = env.commands[robot_index, 3].item()
            cmd_pitch = env.commands[robot_index, 4].item()
            cmd_height = env.commands[robot_index, 5].item()
            cmd_body_yaw = env.commands[robot_index, 2].item()

            roll_now = actual_roll[robot_index].item()
            pitch_now = actual_pitch[robot_index].item()
            height_now = actual_height[robot_index].item()
            body_yaw_now = env._current_relative_yaw()[robot_index].item()
            wrapped_yaw_error = env._body_yaw_error()[robot_index].item()

            print(
                "\n"
                f"body yaw: cmd={cmd_body_yaw:+.3f}, actual={body_yaw_now:+.3f}, "
                f"error={wrapped_yaw_error:+.3f}\n"
                f"roll:   cmd={cmd_roll:+.3f}, actual={roll_now:+.3f}, "
                f"error={cmd_roll-roll_now:+.3f}\n"
                f"pitch:  cmd={cmd_pitch:+.3f}, actual={pitch_now:+.3f}, "
                f"error={cmd_pitch-pitch_now:+.3f}\n"
                f"height: cmd={cmd_height:+.3f}, actual={height_now:+.3f}, "
                f"error={cmd_height-height_now:+.3f}"
            )

        if i >= warmup_steps:
            evaluated_steps += 1
            actual_roll, actual_pitch = env._current_roll_pitch()
            actual_height = env._current_base_height()
            roll_error = torch.abs(env.commands[:, 3] - actual_roll)
            pitch_error = torch.abs(env.commands[:, 4] - actual_pitch)
            yaw_error = torch.abs(env._body_yaw_error())
            height_error = torch.abs(env.commands[:, 5] - actual_height)

            support_xy = torch.mean(env.feet_pos[:, :, :2], dim=1)
            base_xy_drift = torch.norm(
                env.root_states[:, :2] - initial_base_xy, dim=1
            )
            support_xy_drift = torch.norm(
                support_xy - initial_support_xy, dim=1
            )
            wheel_xy_drift = torch.norm(
                env.feet_pos[:, :, :2] - initial_feet_xy, dim=2
            )

            for name, values in (
                ("roll_error", roll_error),
                ("pitch_error", pitch_error),
                ("yaw_error", yaw_error),
                ("height_error", height_error),
                ("base_xy_drift", base_xy_drift),
                ("support_xy_drift", support_xy_drift),
            ):
                metric_sums[name] += torch.sum(values)
                metric_maxima[name] = torch.maximum(
                    metric_maxima[name], torch.max(values)
                )

            metric_sums["feet_xy_drift"] += torch.sum(wheel_xy_drift)
            metric_maxima["wheel_xy_drift"] = torch.maximum(
                metric_maxima["wheel_xy_drift"], torch.max(wheel_xy_drift)
            )

            if evaluation_mode == "neutral":
                leg_joint_error = torch.abs(
                    env.dof_pos[:, leg_joint_indices]
                    - env.default_dof_pos[:, leg_joint_indices]
                )
                metric_sums["leg_joint_error"] += torch.sum(leg_joint_error)
                metric_maxima["leg_joint_error"] = torch.maximum(
                    metric_maxima["leg_joint_error"],
                    torch.max(leg_joint_error),
                )

            contact_loss = (
                env.contact_forces[:, env.feet_indices, 2] < 5.0
            )
            torque_saturation = (
                torch.abs(env.torques) >= 0.99 * env.torque_limits
            )
            action_saturation = (
                torch.abs(env.actions[:, leg_joint_indices])
                >= 0.99 * float(env.cfg.normalization.clip_actions)
            )
            metric_sums["contact_loss"] += torch.sum(contact_loss)
            metric_sums["torque_saturation"] += torch.sum(
                torque_saturation
            )
            metric_sums["action_saturation"] += torch.sum(
                action_saturation
            )
            metric_sums["reset_count"] += torch.sum(dones)

            if (
                evaluation_duration is None
                and evaluated_steps % report_steps == 0
            ):
                env_samples = evaluated_steps * env.num_envs
                elapsed_time = evaluated_steps * env.dt
                print(
                    "\n"
                    f"[ZGWT 实时累计误差 | {elapsed_time:.1f} s]\n"
                    f"  roll:   mean={metric_sums['roll_error'].item() / env_samples:.6f}, "
                    f"max={metric_maxima['roll_error'].item():.6f} rad\n"
                    f"  pitch:  mean={metric_sums['pitch_error'].item() / env_samples:.6f}, "
                    f"max={metric_maxima['pitch_error'].item():.6f} rad\n"
                    f"  yaw:    mean={metric_sums['yaw_error'].item() / env_samples:.6f}, "
                    f"max={metric_maxima['yaw_error'].item():.6f} rad\n"
                    f"  height: mean={metric_sums['height_error'].item() / env_samples:.6f}, "
                    f"max={metric_maxima['height_error'].item():.6f} m\n"
                    f"  base XY drift: mean={metric_sums['base_xy_drift'].item() / env_samples:.6f}, "
                    f"max={metric_maxima['base_xy_drift'].item():.6f} m\n"
                    f"  reset count: {int(metric_sums['reset_count'].item())}"
                )

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'body_yaw': env._current_relative_yaw()[robot_index].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

        if stop_requested:
            break

    if previous_sigint_handler is not None:
        signal.signal(signal.SIGINT, previous_sigint_handler)
    if evaluated_steps == 0:
        print("没有完成有效评估采样，因此无法计算误差汇总。")
        return

    scalar_sums = {name: value.item() for name, value in metric_sums.items()}
    scalar_maxima = {
        name: value.item() for name, value in metric_maxima.items()
    }
    env_samples = evaluated_steps * env.num_envs
    wheel_samples = env_samples * len(env.feet_indices)
    leg_samples = env_samples * len(leg_joint_indices)
    torque_samples = env_samples * env.num_dof

    means = {
        name: scalar_sums[name] / env_samples
        for name in (
            "roll_error",
            "pitch_error",
            "yaw_error",
            "height_error",
            "base_xy_drift",
            "support_xy_drift",
        )
    }
    means["feet_xy_drift"] = scalar_sums["feet_xy_drift"] / wheel_samples
    means["contact_loss"] = scalar_sums["contact_loss"] / wheel_samples
    means["torque_saturation"] = (
        scalar_sums["torque_saturation"] / torque_samples
    )
    means["action_saturation"] = (
        scalar_sums["action_saturation"] / leg_samples
    )
    if evaluation_mode == "neutral":
        means["leg_joint_error"] = (
            scalar_sums["leg_joint_error"] / leg_samples
        )
    # Over the fixed evaluation window, this is the number of resets per
    # evaluated environment (one reset among 50 envs is therefore 0.02).
    reset_rate = scalar_sums["reset_count"] / env.num_envs

    print("\n================ ZGWT PLAY EVALUATION ================")
    print(f"mode: {evaluation_mode}")
    print(f"environments: {env.num_envs}")
    print(f"warmup: {evaluation_warmup_time:.1f} s")
    print(f"evaluation: {evaluated_steps * env.dt:.1f} s")
    print("\nPose tracking")
    print(
        f"  roll error:    mean={means['roll_error']:.6f}, "
        f"max={scalar_maxima['roll_error']:.6f} rad"
    )
    print(
        f"  pitch error:   mean={means['pitch_error']:.6f}, "
        f"max={scalar_maxima['pitch_error']:.6f} rad"
    )
    print(
        f"  yaw error:     mean={means['yaw_error']:.6f}, "
        f"max={scalar_maxima['yaw_error']:.6f} rad"
    )
    print(
        f"  height error:  mean={means['height_error']:.6f}, "
        f"max={scalar_maxima['height_error']:.6f} m"
    )
    print("\nStationary stability")
    print(
        f"  base XY drift:       mean={means['base_xy_drift']:.6f}, "
        f"max={scalar_maxima['base_xy_drift']:.6f} m"
    )
    print(
        f"  support XY drift:    mean={means['support_xy_drift']:.6f}, "
        f"max={scalar_maxima['support_xy_drift']:.6f} m"
    )
    print(f"  feet XY drift:       mean={means['feet_xy_drift']:.6f} m")
    print(
        f"  max wheel drift:     {scalar_maxima['wheel_xy_drift']:.6f} m"
    )
    if evaluation_mode == "neutral":
        print("\nNeutral stance")
        print(
            f"  leg joint error: mean={means['leg_joint_error']:.6f}, "
            f"max={scalar_maxima['leg_joint_error']:.6f} rad"
        )
    print("\nSafety")
    print(f"  episode reset count:    {int(scalar_sums['reset_count'])}")
    print(f"  fall/reset rate:        {reset_rate:.6f}")
    print(f"  contact loss ratio:     {means['contact_loss']:.6f}")
    print(f"  torque saturation:      {means['torque_saturation']:.6f}")
    print(f"  action saturation:      {means['action_saturation']:.6f}")

    if evaluation_mode == "neutral":
        review_reasons = []
        checks = (
            (means["roll_error"] > 0.03, "roll error exceeds limit"),
            (means["pitch_error"] > 0.03, "pitch error exceeds limit"),
            (means["yaw_error"] > 0.02, "yaw error exceeds limit"),
            (means["height_error"] > 0.015, "height error exceeds limit"),
            (
                scalar_maxima["base_xy_drift"] > 0.020,
                "base drift exceeds limit",
            ),
            (
                scalar_maxima["support_xy_drift"] > 0.015,
                "support drift exceeds limit",
            ),
            (
                scalar_maxima["wheel_xy_drift"] > 0.035,
                "wheel drift exceeds limit",
            ),
            (reset_rate > 0.01, "reset rate exceeds limit"),
            (
                means["torque_saturation"] > 0.01,
                "torque saturation exceeds limit",
            ),
            (
                means["action_saturation"] > 0.01,
                "action saturation exceeds limit",
            ),
        )
        review_reasons.extend(reason for failed, reason in checks if failed)
        print(f"\nResult: {'REVIEW' if review_reasons else 'PASS'}")
        if review_reasons:
            print("Reasons:")
            for reason in review_reasons:
                print(f"- {reason}")
    print("======================================================")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(
        args,
        evaluation_mode="command",
        x_vel=0.0,
        y_vel=0.0,
        body_yaw=0.05,
        body_roll=0.0,
        body_pitch=0.0,
        body_height=0.54,
        dance_trajectory=False,
        evaluation_duration=None,
        # 设置为 None 时将一直运行，直到 Ctrl+C 或关闭 viewer。
        # 姿态命令测试时改为 evaluation_mode="command"，并设置相应的
        # body_yaw、body_roll、body_pitch 和 body_height。
    )
