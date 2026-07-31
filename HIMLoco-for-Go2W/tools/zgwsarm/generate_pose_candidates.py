"""离线生成远离限位和奇异区的 ZGWSARM 静态姿态候选。

本脚本只完成 URDF 正运动学、几何 Jacobian 和关节余量检查。它不会把
候选标记为 ``simulation_verified``，因为碰撞和静态保持仍必须在 PhysX
中验证。这样可以在没有遥操作记录时生成候选，又不会伪造安全认证。

示例：
  python tools/zgwsarm/generate_pose_candidates.py --count 500
"""

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "resources/robots/zgwsarm/zgwsarm.urdf"
DEFAULT_OUTPUT = ROOT / "config/robots/zgwsarm/arm_pose_candidates.csv"
ARM_NAMES = [f"ROBOT_ARM_JOINT{i}" for i in range(1, 7)]
LOWER = np.array([-2.618, 0.0, -2.9671, -1.57, -1.57, -6.28])
UPPER = np.array([2.618, 3.14, 0.0, 1.57, 1.57, 6.28])
CSV_COLUMNS = [
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
] + [f"arm_q{i}" for i in range(1, 7)]


def _vector(text):
    return np.array([float(value) for value in text.split()], dtype=np.float64)


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def load_arm_chain(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    chain = []
    for name in ARM_NAMES:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = _vector(origin.attrib.get("xyz", "0 0 0"))
        rpy = _vector(origin.attrib.get("rpy", "0 0 0"))
        axis = _vector(joint.find("axis").attrib["xyz"])
        chain.append((_transform(_rpy_matrix(rpy), xyz), axis))
    return chain


def geometric_jacobian(chain, q):
    transform = np.eye(4)
    joint_positions = []
    joint_axes = []
    for (origin, local_axis), angle in zip(chain, q):
        transform = transform @ origin
        joint_positions.append(transform[:3, 3].copy())
        joint_axes.append(transform[:3, :3] @ local_axis)
        transform = transform @ _transform(_axis_angle(local_axis, angle))

    end_position = transform[:3, 3]
    jacobian = np.zeros((6, 6))
    for index, (position, axis) in enumerate(zip(joint_positions, joint_axes)):
        jacobian[:3, index] = np.cross(axis, end_position - position)
        jacobian[3:, index] = axis
    return transform, jacobian


def classify_pose(q, end_position):
    center = 0.5 * (LOWER + UPPER)
    half_range = 0.5 * (UPPER - LOWER)
    normalized = np.abs((q - center) / half_range)
    if np.max(normalized) > 0.72:
        return "near_allowed_boundary"
    horizontal_extension = np.linalg.norm(end_position[:2])
    if horizontal_extension > 0.40:
        return "directional_extension"
    return "workspace_midrange"


def generate(args):
    chain = load_arm_chain(args.urdf)
    rng = np.random.default_rng(args.seed)
    low = LOWER + args.joint_margin
    high = UPPER - args.joint_margin
    if np.any(low >= high):
        raise ValueError("joint_margin 过大，导致可采样区间为空")

    accepted = []
    attempts = 0
    maximum_attempts = max(args.count * args.max_attempt_multiplier, args.count)
    while len(accepted) < args.count and attempts < maximum_attempts:
        attempts += 1
        q = rng.uniform(low, high)
        end_transform, jacobian = geometric_jacobian(chain, q)
        sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
        if sigma < args.minimum_sigma:
            continue
        margin = float(np.min(np.minimum(q - LOWER, UPPER - q)))
        accepted.append(
            {
                "q": q,
                "sigma": sigma,
                "margin": margin,
                "category": classify_pose(q, end_transform[:3, 3]),
            }
        )

    if len(accepted) < args.count:
        raise RuntimeError(
            f"只生成 {len(accepted)}/{args.count} 个候选；"
            "请增加 --max-attempt-multiplier 或检查奇异性阈值。"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, item in enumerate(accepted):
            row = {
                "timestamp_s": "0.0",
                "pose_id": f"synthetic_{index:04d}",
                "category": item["category"],
                "source": "synthetic_urdf_jacobian",
                "simulation_verified": "false",
                "hardware_verified": "false",
                "collision_checked": "false",
                "singularity_checked": "true",
                "min_joint_margin_rad": f"{item['margin']:.9f}",
                "min_jacobian_sigma": f"{item['sigma']:.9f}",
            }
            row.update(
                {
                    f"arm_q{joint + 1}": f"{value:.9f}"
                    for joint, value in enumerate(item["q"])
                }
            )
            writer.writerow(row)

    categories = {}
    for item in accepted:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    print(f"已生成 {len(accepted)} 个候选: {output}")
    print(f"采样尝试次数: {attempts}")
    print(f"类别统计: {categories}")
    print("注意：候选尚未做 PhysX 碰撞/保持验证，不能直接用于 ArmStandV2。")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--joint-margin", type=float, default=0.15)
    parser.add_argument("--minimum-sigma", type=float, default=0.02)
    parser.add_argument("--max-attempt-multiplier", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
