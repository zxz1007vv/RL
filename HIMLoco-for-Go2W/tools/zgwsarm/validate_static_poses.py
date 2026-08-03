"""在固定基座 PhysX 中验证机械臂候选姿态并生成 ArmStandV2 姿态库。

验证内容包括：机械臂/机身/地面碰撞、aligned 计算力矩静态保持误差、力矩占比。
输入候选必须已经完成关节限位和 Jacobian 检查。通过的行才会写成
``simulation_verified=true``；``hardware_verified`` 永远保持 false。
"""

import argparse
import csv
import math
from pathlib import Path

from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs.zgwt_arm.zgwt_arm_dynamics import ZgwtArmDynamics
from legged_gym.envs.zgwt_arm.zgwt_arm_control_utils import (
    inertia_normalized_desired_acceleration,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET = ROOT / "resources/robots/zgwsarm/zgwsarm.urdf"
DEFAULT_INPUT = ROOT / "config/robots/zgwsarm/arm_pose_candidates.csv"
DEFAULT_OUTPUT = ROOT / "config/robots/zgwsarm/arm_safe_static_poses.csv"
ARM_NAMES = [f"ROBOT_ARM_JOINT{i}" for i in range(1, 7)]


DOG_DEFAULT = {
    "FBL_ABAD_JOINT": 0.0,
    "FAR_ABAD_JOINT": 0.0,
    "RBL_ABAD_JOINT": 0.0,
    "RAR_ABAD_JOINT": 0.0,
    "FBL_HIP_JOINT": 0.6,
    "FAR_HIP_JOINT": 0.6,
    "RBL_HIP_JOINT": -0.6,
    "RAR_HIP_JOINT": -0.6,
    "FBL_KNEE_JOINT": -1.2,
    "FAR_KNEE_JOINT": -1.2,
    "RBL_KNEE_JOINT": 1.2,
    "RAR_KNEE_JOINT": 1.2,
    "FBL_FOOT_JOINT": 0.0,
    "FAR_FOOT_JOINT": 0.0,
    "RBL_FOOT_JOINT": 0.0,
    "RAR_FOOT_JOINT": 0.0,
}


def _device_id(device):
    if device == "cpu":
        return 0
    if not device.startswith("cuda:"):
        raise ValueError("--device 必须是 cpu 或 cuda:N")
    return int(device.split(":", 1)[1])


def _load_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not rows:
            raise ValueError("候选姿态文件为空，请先运行生成脚本")
        return reader.fieldnames, rows


def validate(args):
    fieldnames, rows = _load_rows(args.input)
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.device.startswith("cuda")
    sim_params.physx.use_gpu = args.device.startswith("cuda")
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1

    device_id = _device_id(args.device)
    sim = gym.create_sim(device_id, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("无法创建 Isaac Gym 仿真")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane.static_friction = 0.8
    plane.dynamic_friction = 0.8
    gym.add_ground(sim, plane)

    asset_path = Path(args.asset).resolve()
    options = gymapi.AssetOptions()
    options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    options.collapse_fixed_joints = True
    options.fix_base_link = True
    options.replace_cylinder_with_capsule = False
    options.flip_visual_attachments = False
    asset = gym.load_asset(
        sim, str(asset_path.parent), asset_path.name, options
    )
    if asset is None:
        gym.destroy_sim(sim)
        raise RuntimeError(f"无法加载资产: {asset_path}")

    dof_names = list(gym.get_asset_dof_names(asset))
    body_names = list(gym.get_asset_rigid_body_names(asset))
    if len(dof_names) != 22:
        gym.destroy_sim(sim)
        raise ValueError(f"训练资产必须是 22 DOF，实际为 {len(dof_names)}")
    arm_dof_indices = [dof_names.index(name) for name in ARM_NAMES]
    arm_body_indices = [
        index
        for index, name in enumerate(body_names)
        if name.startswith("ROBOT_ARM_LINK")
    ]

    count = len(rows)
    arm_dynamics = ZgwtArmDynamics.from_urdf(
        str(asset_path), ARM_NAMES, device=args.device, dtype=torch.float
    )
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    columns = int(math.ceil(math.sqrt(count)))
    envs = []
    for index in range(count):
        env = gym.create_env(sim, lower, upper, columns)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.55)
        gym.create_actor(env, asset, pose, "zgwsarm", index, 0, 0)
        envs.append(env)

    gym.prepare_sim(sim)
    state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(
        count, len(dof_names), 2
    )
    contact = gymtorch.wrap_tensor(
        gym.acquire_net_contact_force_tensor(sim)
    ).view(count, len(body_names), 3)

    target = torch.zeros(
        count, len(dof_names), dtype=torch.float, device=args.device
    )
    for index, name in enumerate(dof_names):
        if name in DOG_DEFAULT:
            target[:, index] = DOG_DEFAULT[name]
    for row_index, row in enumerate(rows):
        target[row_index, arm_dof_indices] = torch.tensor(
            [float(row[f"arm_q{i}"]) for i in range(1, 7)],
            dtype=torch.float,
            device=args.device,
        )
    state[..., 0] = target
    state[..., 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(state))

    kp = torch.zeros(len(dof_names), dtype=torch.float, device=args.device)
    kd = torch.zeros_like(kp)
    for index, name in enumerate(dof_names):
        if "ABAD_JOINT" in name:
            kp[index], kd[index] = 90.0, 1.0
        elif "HIP_JOINT" in name or "KNEE_JOINT" in name:
            kp[index], kd[index] = 120.0, 1.0
        elif "FOOT_JOINT" in name:
            kp[index], kd[index] = 25.0, 2.5
    kp[arm_dof_indices] = 500.0
    kd[arm_dof_indices] = 45.0
    previous_arm_torque = torch.zeros(
        count, 6, dtype=torch.float, device=args.device
    )
    maximum_arm_contact = torch.zeros(
        count, dtype=torch.float, device=args.device
    )
    maximum_arm_torque_ratio = torch.zeros_like(maximum_arm_contact)
    fixed_base_quaternion = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=torch.float,
        device=args.device,
    ).repeat(count, 1)
    zero_base_angular_velocity = torch.zeros(
        count, 3, dtype=torch.float, device=args.device
    )
    gravity_world = torch.tensor(
        [0.0, 0.0, -9.81],
        dtype=torch.float,
        device=args.device,
    )

    steps = max(1, int(round(args.duration / args.dt)))
    for _ in range(steps):
        gym.refresh_dof_state_tensor(sim)
        desired = kp * (target - state[..., 0]) - kd * state[..., 1]
        arm_position = state[:, arm_dof_indices, 0]
        arm_velocity = state[:, arm_dof_indices, 1]
        desired_acceleration = inertia_normalized_desired_acceleration(
            torch.zeros_like(arm_position),
            target[:, arm_dof_indices] - arm_position,
            -arm_velocity,
            500.0,
            45.0,
        )
        desired_arm = arm_dynamics.inverse_dynamics(
            arm_position,
            arm_velocity,
            desired_acceleration,
            base_quaternion=fixed_base_quaternion,
            base_angular_velocity=zero_base_angular_velocity,
            gravity=gravity_world,
        )
        desired_arm = desired_arm.clamp(-100.0, 100.0)
        arm_delta = (desired_arm - previous_arm_torque).clamp(
            -1000.0 * args.dt, 1000.0 * args.dt
        )
        previous_arm_torque += arm_delta
        desired[:, arm_dof_indices] = previous_arm_torque
        desired = torch.minimum(
            torch.maximum(desired, -torch.tensor(180.0, device=args.device)),
            torch.tensor(180.0, device=args.device),
        )
        gym.set_dof_actuation_force_tensor(
            sim, gymtorch.unwrap_tensor(desired)
        )
        gym.simulate(sim)
        if args.device == "cpu":
            gym.fetch_results(sim, True)
        gym.refresh_net_contact_force_tensor(sim)
        arm_force = torch.norm(contact[:, arm_body_indices], dim=2).max(dim=1).values
        maximum_arm_contact = torch.maximum(maximum_arm_contact, arm_force)
        maximum_arm_torque_ratio = torch.maximum(
            maximum_arm_torque_ratio,
            torch.abs(previous_arm_torque).max(dim=1).values / 100.0,
        )

    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    final_error = torch.abs(
        state[:, arm_dof_indices, 0] - target[:, arm_dof_indices]
    ).max(dim=1).values

    passed = (
        (maximum_arm_contact <= args.maximum_contact_force)
        & (final_error <= args.maximum_hold_error)
        & (maximum_arm_torque_ratio <= args.maximum_torque_ratio)
    ).cpu()
    contact_cpu = maximum_arm_contact.cpu()
    error_cpu = final_error.cpu()
    torque_cpu = maximum_arm_torque_ratio.cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = [
        name
        for name in ("max_arm_contact_force", "max_hold_error", "max_hold_torque_ratio")
        if name not in fieldnames
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames) + extra_columns
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            ok = bool(passed[index].item())
            row["collision_checked"] = "true"
            row["simulation_verified"] = "true" if ok else "false"
            row["hardware_verified"] = "false"
            row["max_arm_contact_force"] = f"{contact_cpu[index].item():.9f}"
            row["max_hold_error"] = f"{error_cpu[index].item():.9f}"
            row["max_hold_torque_ratio"] = f"{torque_cpu[index].item():.9f}"
            writer.writerow(row)

    gym.destroy_sim(sim)
    print(f"PhysX 验证通过 {int(passed.sum().item())}/{count}: {output}")
    print("所有合成姿态仍保持 hardware_verified=false。")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=str(DEFAULT_ASSET))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.002)
    # 1000 N·m/s 的力矩变化率限制会让大负载姿态需要约 1～2 秒收敛。
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--maximum-contact-force", type=float, default=1.0)
    parser.add_argument("--maximum-hold-error", type=float, default=0.03)
    parser.add_argument("--maximum-torque-ratio", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
