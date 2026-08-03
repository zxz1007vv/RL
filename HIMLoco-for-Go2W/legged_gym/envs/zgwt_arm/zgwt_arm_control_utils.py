"""ZGWT 带臂任务使用的纯 Python 配置、姿态库和控制辅助函数。

本文件不依赖 Isaac Gym，便于单元测试，也方便 C++ 端复核 CSV 过滤规则。
机械臂姿态只作为环境参数和诊断标签，不进入任何策略观测。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence


ARM_JOINT_COLUMNS = tuple(f"arm_q{i}" for i in range(1, 7))
REQUIRED_POSE_COLUMNS = (
    "timestamp_s",
    "pose_id",
    "category",
    "source",
    "simulation_verified",
    "hardware_verified",
    "collision_checked",
    "singularity_checked",
    "min_joint_margin_rad",
    "min_jacobian_sigma",
) + ARM_JOINT_COLUMNS


class ArmStaticPose(NamedTuple):
    pose_id: str
    category: str
    source: str
    q: tuple
    simulation_verified: bool
    hardware_verified: bool
    collision_checked: bool
    singularity_checked: bool
    min_joint_margin_rad: float
    min_jacobian_sigma: float


def inertia_normalized_desired_acceleration(
    reference_acceleration,
    position_error,
    velocity_error,
    kp,
    kd,
):
    """唯一控制律：把位置/速度反馈解释为加速度并统一送入逆动力学。"""

    return reference_acceleration + kp * position_error + kd * velocity_error


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def joint_limit_margin(
    q: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> float:
    if not (len(q) == len(lower) == len(upper)):
        raise ValueError("q/lower/upper 维度必须一致")
    return min(
        min(float(value) - float(lo), float(hi) - float(value))
        for value, lo, hi in zip(q, lower, upper)
    )


def _require_finite(values: Iterable[float], label: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} 中包含非有限数值")


def load_simulation_verified_poses(
    csv_path: str,
    lower: Sequence[float],
    upper: Sequence[float],
    minimum_joint_margin: float,
    minimum_jacobian_sigma: float,
) -> List[ArmStaticPose]:
    """读取可供 ArmStandV2 仿真训练使用的姿态。

    ``hardware_verified`` 不参与仿真筛选，也绝不会由本函数自动提升。
    姿态必须同时通过仿真、碰撞和奇异性检查，并再次用契约关节限位复核。
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"机械臂姿态库不存在: {path}")

    accepted: List[ArmStaticPose] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_POSE_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("机械臂姿态库缺少列: " + ", ".join(missing))

        for line_number, row in enumerate(reader, start=2):
            try:
                q = tuple(float(row[name]) for name in ARM_JOINT_COLUMNS)
                stored_margin = float(row["min_joint_margin_rad"])
                sigma = float(row["min_jacobian_sigma"])
                _require_finite(q + (stored_margin, sigma), f"第 {line_number} 行")
                pose = ArmStaticPose(
                    pose_id=row["pose_id"].strip(),
                    category=row["category"].strip(),
                    source=row["source"].strip(),
                    q=q,
                    simulation_verified=parse_bool(row["simulation_verified"]),
                    hardware_verified=parse_bool(row["hardware_verified"]),
                    collision_checked=parse_bool(row["collision_checked"]),
                    singularity_checked=parse_bool(row["singularity_checked"]),
                    min_joint_margin_rad=stored_margin,
                    min_jacobian_sigma=sigma,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"机械臂姿态库第 {line_number} 行无效: {exc}") from exc

            actual_margin = joint_limit_margin(q, lower, upper)
            if abs(actual_margin - stored_margin) > 1.0e-4:
                raise ValueError(
                    f"姿态 {pose.pose_id} 的关节余量记录不一致: "
                    f"CSV={stored_margin:.6f}, 实算={actual_margin:.6f}"
                )
            if (
                pose.simulation_verified
                and pose.collision_checked
                and pose.singularity_checked
                and actual_margin >= minimum_joint_margin
                and sigma >= minimum_jacobian_sigma
            ):
                accepted.append(pose)

    if not accepted:
        raise ValueError(
            "姿态库中没有可用于 ArmStandV2 的姿态；必须同时满足 "
            "simulation_verified=true、collision_checked=true、"
            "singularity_checked=true、关节余量和 Jacobian 阈值。"
        )
    return accepted


def normalized_pose_weights(
    poses: Sequence[ArmStaticPose], category_probabilities: Dict[str, float]
) -> List[float]:
    """把类别概率均分给类别内的姿态，空类别不会偷走概率质量。"""

    if not poses:
        raise ValueError("poses 不能为空")
    counts: Dict[str, int] = {}
    for pose in poses:
        counts[pose.category] = counts.get(pose.category, 0) + 1

    raw = []
    for pose in poses:
        probability = float(category_probabilities.get(pose.category, 0.0))
        if probability < 0.0:
            raise ValueError("姿态类别概率不能为负数")
        raw.append(probability / counts[pose.category])

    total = sum(raw)
    if total <= 0.0:
        raise ValueError("可用姿态类别的总概率必须大于 0")
    return [value / total for value in raw]


def torque_rate_limit(
    desired, previous, maximum_rate: float, control_dt: float
):
    """Tensor/array 通用的逐控制步力矩变化率限制。"""

    if maximum_rate <= 0.0 or control_dt <= 0.0:
        raise ValueError("maximum_rate 和 control_dt 必须为正数")
    maximum_step = maximum_rate * control_dt
    delta = (desired - previous).clip(min=-maximum_step, max=maximum_step)
    return previous + delta


def advance_limited_joint_reference(
    command,
    position,
    velocity,
    velocity_limit,
    acceleration_limit,
    control_dt: float,
    velocity_filter_tau: float,
):
    """推进一阶速度滤波、速度/加速度受限的关节参考轨迹。

    使用制动距离限制目标速度，避免接近目标时仍以最大速度前进。返回
    ``(position_next, velocity_next, acceleration_next)``。
    """

    if control_dt <= 0.0:
        raise ValueError("control_dt 必须为正数")
    if velocity_filter_tau < 0.0:
        raise ValueError("velocity_filter_tau 不能为负数")
    if (velocity_limit <= 0.0).any() or (acceleration_limit <= 0.0).any():
        raise ValueError("速度和加速度限制必须为正数")

    error = command - position
    braking_speed = (2.0 * acceleration_limit * error.abs()).sqrt()
    requested_velocity = error.sign() * velocity_limit.minimum(braking_speed)
    if velocity_filter_tau == 0.0:
        alpha = 1.0
    else:
        alpha = 1.0 - math.exp(-control_dt / velocity_filter_tau)
    filtered_velocity = velocity + alpha * (requested_velocity - velocity)
    acceleration = ((filtered_velocity - velocity) / control_dt).clip(
        min=-acceleration_limit, max=acceleration_limit
    )
    velocity_next = (velocity + acceleration * control_dt).clip(
        min=-velocity_limit, max=velocity_limit
    )
    position_next = position + velocity_next * control_dt
    # 只有速度已经能在一个受限加速度步内归零时才吸附到终点，避免原先
    # “越过目标后瞬间把速度清零”造成几十 rad/s² 的伪加速度尖峰。
    settled = (error.abs() < 1.0e-5) & (
        velocity.abs() <= acceleration_limit * control_dt
    )
    position_next = position_next.where(~settled, command)
    velocity_next = velocity_next.where(~settled, velocity_next * 0.0)
    acceleration = acceleration.where(
        ~settled, (velocity_next - velocity) / control_dt
    )
    return position_next, velocity_next, acceleration
