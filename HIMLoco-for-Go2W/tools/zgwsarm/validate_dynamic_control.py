"""不启动 PPO，验证 ZGWSARM 动态传统控制链路。

从 ArmStandV2 安全姿态库选择一个近邻目标，先做完整插值路径 Jacobian 检查，
再以手动速度限制执行。过程中检查参考速度、参考加速度、力矩、力矩变化率、
机械臂碰撞导致的 reset 和最终跟踪误差。
"""

import argparse
import sys

import isaacgym
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def validate(args, duration):
    env, _ = task_registry.make_env(name=args.task, args=args)
    env.reset()
    if not hasattr(env, "arm_pose_table") or env.arm_pose_table is None:
        raise ValueError("动态验证必须使用带安全姿态库的任务")
    if env.num_envs != 1:
        raise ValueError("动态验证固定使用 --num_envs 1")

    start = env.arm_reference_position[0].clone()
    distances = torch.max(torch.abs(env.arm_pose_table - start), dim=1).values
    candidates = torch.argsort(distances)
    chosen = None
    for index in candidates.tolist():
        if distances[index] < 0.02:
            continue
        try:
            env.set_arm_joint_targets(
                env.arm_pose_table[index],
                manual=True,
                validate_path=True,
            )
            chosen = index
            break
        except ValueError:
            continue
    if chosen is None:
        raise RuntimeError("没有找到通过完整 Jacobian 路径检查的近邻姿态")

    actions = torch.zeros(
        env.num_envs, env.num_actions, dtype=torch.float, device=env.device
    )
    steps = max(1, int(round(duration / env.dt)))
    max_reference_velocity = 0.0
    max_reference_acceleration = 0.0
    max_torque_ratio = 0.0
    max_torque_rate = 0.0
    reset_count = 0
    previous_torque = env.arm_previous_torque.clone()
    for _ in range(steps):
        _, _, _, dones, _, _, _ = env.step(actions)
        max_reference_velocity = max(
            max_reference_velocity,
            float(torch.max(torch.abs(env.arm_reference_velocity)).item()),
        )
        max_reference_acceleration = max(
            max_reference_acceleration,
            float(torch.max(torch.abs(env.arm_reference_acceleration)).item()),
        )
        max_torque_ratio = max(
            max_torque_ratio,
            float(
                torch.max(
                    torch.abs(env.arm_previous_torque)
                    / env.arm_torque_limits
                ).item()
            ),
        )
        max_torque_rate = max(
            max_torque_rate,
            float(
                torch.max(
                    torch.abs(env.arm_previous_torque - previous_torque)
                    / env.dt
                ).item()
            ),
        )
        previous_torque[:] = env.arm_previous_torque
        reset_count += int(torch.sum(dones).item())

    final_error = float(
        torch.max(
            torch.abs(
                env.dof_pos[:, env.arm_dof_indices]
                - env.arm_reference_position
            )
        ).item()
    )
    command_error = float(
        torch.max(
            torch.abs(env.arm_command_targets - env.arm_reference_position)
        ).item()
    )
    print("\n================ ZGWSARM DYNAMIC VALIDATION ================")
    print(f"source pose index: {int(env.arm_pose_index[0].item())}")
    print(f"target pose index: {chosen}")
    print(f"maximum reference velocity:     {max_reference_velocity:.6f} rad/s")
    print(
        f"maximum reference acceleration: {max_reference_acceleration:.6f} rad/s^2"
    )
    print(f"maximum torque ratio:           {max_torque_ratio:.6f}")
    print(f"maximum observed torque rate:   {max_torque_rate:.6f} N·m/s")
    print(f"final tracking error:           {final_error:.6f} rad")
    print(f"remaining command error:        {command_error:.6f} rad")
    print(f"reset count:                    {reset_count}")
    print("============================================================")

    failures = []
    if max_reference_velocity > 0.5001:
        failures.append("手动目标速度超过 0.5 rad/s")
    if max_reference_acceleration > 2.0001:
        failures.append("目标加速度超过 2.0 rad/s²")
    # 每个 RL step 包含五个 2 ms 子步，因此观测到的跨 RL step 差分仍应
    # 不超过同一 1000 N·m/s 限制。
    if max_torque_rate > 1000.1:
        failures.append("力矩变化率超过 1000 N·m/s")
    if max_torque_ratio > 1.0001:
        failures.append("力矩超过 100 N·m 限幅")
    if final_error > 0.05:
        failures.append("最终关节跟踪误差超过 0.05 rad")
    if reset_count:
        failures.append("动态路径触发碰撞或机器人 reset")
    if failures:
        raise RuntimeError("；".join(failures))


def parse_duration(argv):
    parser = argparse.ArgumentParser(add_help=False)
    # 惯量归一化控制配合 1000 N·m/s 力矩斜率限制时，需要给最终目标
    # 留出独立收敛时间；5 秒验证曾出现 0.070416 rad 的剩余误差。
    parser.add_argument("--duration", type=float, default=10.0)
    known, remaining = parser.parse_known_args(argv[1:])
    sys.argv = [argv[0]] + remaining
    return known.duration


if __name__ == "__main__":
    duration_value = parse_duration(sys.argv)
    core_args = get_args()
    if core_args.task != "zgwt_dance_arm_static":
        raise ValueError("--task 必须是 zgwt_dance_arm_static")
    validate(core_args, duration_value)
