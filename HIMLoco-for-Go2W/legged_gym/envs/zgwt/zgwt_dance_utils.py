"""Pure tensor helpers shared by the ZGWT dance task and its unit tests."""

import math

import torch


def wrap_to_pi(angle):
    """Wrap an angle tensor to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def target_projected_gravity(roll, pitch):
    """Gravity expected in body coordinates at the requested roll/pitch."""
    cos_pitch = torch.cos(pitch)
    return torch.stack(
        (
            torch.sin(pitch),
            -torch.sin(roll) * cos_pitch,
            -torch.cos(roll) * cos_pitch,
        ),
        dim=-1,
    )


def effective_actions(actions, wheel_indices):
    """Return the action actually consumed by the controller and observations."""
    if actions.ndim != 2:
        raise ValueError("actions must have shape [num_envs, num_actions]")
    masked = actions.clone()
    masked[:, wheel_indices] = 0.0
    return masked


def select_wheel_factors(factors, wheel_indices, num_dof):
    """Select per-wheel damping factors without accidental broadcasting."""
    if factors.ndim != 2:
        raise ValueError("factors must have shape [num_envs, 1|num_dof]")
    if factors.shape[1] == 1:
        return factors
    if factors.shape[1] == num_dof:
        return factors[:, wheel_indices]
    raise ValueError("factors must have shape [num_envs, 1|num_dof]")


def neutral_command_weight(normalized_commands, sigma):
    """Continuous neutral-pose gate for normalized yaw/roll/pitch/height."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    cost = torch.sum(torch.square(normalized_commands[..., 2:]), dim=-1) / sigma
    return torch.exp(-cost)


def low_pass_alpha(dt, time_constant):
    """Exact discrete alpha for a first-order low-pass filter."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if time_constant <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / time_constant)


def floored_soft_multiplier(soft_penalty, sigma, floor):
    """Map a non-positive soft reward to [floor, 1]."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    exponent = torch.clamp(soft_penalty / sigma, min=-20.0, max=0.0)
    return floor + (1.0 - floor) * torch.exp(exponent)


def leg_extension_from_knee(knee_angle, thigh_length, shank_length):
    """根据膝关节夹角计算髋到轮心的等效直线伸展量。"""
    if thigh_length <= 0.0 or shank_length <= 0.0:
        raise ValueError("腿段长度必须为正数")
    extension_sq = (
        thigh_length * thigh_length
        + shank_length * shank_length
        + 2.0 * thigh_length * shank_length * torch.cos(knee_angle)
    )
    return torch.sqrt(torch.clamp(extension_sq, min=1.0e-12))


def height_tracking_score(
    current_height,
    target_height,
    symmetric_sigma,
    deficit_deadzone,
    deficit_sigma,
    deficit_weight,
):
    """组合对称高度误差与超过死区后的单侧下沉代价。"""
    if symmetric_sigma <= 0.0 or deficit_sigma <= 0.0:
        raise ValueError("高度 reward 的 sigma 必须为正数")
    if deficit_deadzone < 0.0 or deficit_weight < 0.0:
        raise ValueError("高度下沉死区和权重不能为负数")
    error = current_height - target_height
    deficit = torch.relu(-error - deficit_deadzone)
    cost = torch.square(error) / symmetric_sigma
    cost += deficit_weight * torch.square(deficit) / deficit_sigma
    return torch.exp(-cost)


def smooth_l1_excess_cost(excess, penalty_scale):
    """Convert a non-negative physical-limit violation into a smooth cost."""
    if penalty_scale <= 0.0:
        raise ValueError("penalty_scale must be positive")
    normalized = torch.relu(excess) / penalty_scale
    return torch.where(
        normalized < 1.0,
        0.5 * torch.square(normalized),
        normalized - 0.5,
    )


def stance_coordination_score(
    leg_extension,
    expected_differential,
    deadzone,
    sigma,
):
    """奖励扣除命令所需差分后仍保持一致的四腿伸展量。"""
    if leg_extension.ndim != 2 or leg_extension.shape[1] != 4:
        raise ValueError("leg_extension 必须为 [num_envs, 4]")
    if expected_differential.shape != leg_extension.shape:
        raise ValueError("expected_differential 必须与 leg_extension 同形")
    if sigma <= 0.0:
        raise ValueError("协调 reward 的 sigma 必须为正数")
    if torch.any(deadzone < 0.0):
        raise ValueError("协调死区不能为负数")
    common_mode = torch.mean(leg_extension, dim=1, keepdim=True)
    residual = leg_extension - common_mode - expected_differential
    excess = torch.relu(torch.abs(residual) - deadzone)
    cost = torch.mean(torch.square(excess), dim=1)
    return torch.exp(-cost / sigma), residual
