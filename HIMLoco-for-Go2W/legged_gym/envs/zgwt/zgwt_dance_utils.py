"""Pure tensor helpers shared by the ZGWT dance task and its unit tests."""

import math

import torch


def wrap_to_pi(angle):
    """Wrap an angle tensor to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def normalize_dance_commands(
    commands,
    default_height,
    min_height,
    max_abs_yaw,
    max_abs_roll,
    max_abs_pitch,
):
    """Normalize [vx, vy, yaw_error, roll, pitch, absolute_height].

    The dance task deliberately supports crouching only: ``default_height`` is
    both the neutral and maximum height. Heights above it are clipped to neutral
    instead of creating an untrainable upward command.
    """
    if commands.shape[-1] != 6:
        raise ValueError(f"expected six dance commands, got {commands.shape[-1]}")
    if min_height >= default_height:
        raise ValueError("min_height must be below default_height")
    for name, limit in (
        ("max_abs_yaw", max_abs_yaw),
        ("max_abs_roll", max_abs_roll),
        ("max_abs_pitch", max_abs_pitch),
    ):
        if limit <= 0.0:
            raise ValueError(f"{name} must be positive")

    normalized = torch.zeros_like(commands)
    normalized[..., 2] = torch.clamp(
        commands[..., 2] / max_abs_yaw, -1.0, 1.0
    )
    normalized[..., 3] = torch.clamp(
        commands[..., 3] / max_abs_roll, -1.0, 1.0
    )
    normalized[..., 4] = torch.clamp(
        commands[..., 4] / max_abs_pitch, -1.0, 1.0
    )
    crouch_span = default_height - min_height
    normalized[..., 5] = torch.clamp(
        (commands[..., 5] - default_height) / crouch_span, -1.0, 0.0
    )
    return normalized


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
