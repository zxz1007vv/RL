"""基于 URDF 的六轴机械臂批量刚体动力学。

实现世界坐标系递归牛顿－欧拉逆动力学，覆盖：

- 随机机身 roll/pitch 下的重力补偿；
- 机械臂关节速度产生的科氏/离心偏置力；
- 机身角速度与偏置安装位置产生的惯性项；
- ``M(q) @ qdd_ref`` 计算参考加速度力矩前馈；
- 固定末端工具的质量和惯量。

机身线加速度和角加速度前馈按 C++ 契约关闭；机身当前角速度仍参与 bias。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import torch


def _parse_vector(text, default="0 0 0"):
    return [float(value) for value in (text or default).split()]


def _rpy_matrix(values):
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _compose_transform(first, second):
    first_r, first_t = first
    second_r, second_t = second
    first_r = torch.tensor(first_r, dtype=torch.double)
    second_r = torch.tensor(second_r, dtype=torch.double)
    first_t = torch.tensor(first_t, dtype=torch.double)
    second_t = torch.tensor(second_t, dtype=torch.double)
    rotation = first_r @ second_r
    translation = first_t + first_r @ second_t
    return rotation.tolist(), translation.tolist()


def _joint_origin(joint):
    origin = joint.find("origin")
    if origin is None:
        return _rpy_matrix([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]
    return (
        _rpy_matrix(_parse_vector(origin.attrib.get("rpy"))),
        _parse_vector(origin.attrib.get("xyz")),
    )


def _link_inertial(link):
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"link {link.attrib['name']} 缺少 inertial")
    origin = inertial.find("origin")
    mass = float(inertial.find("mass").attrib["value"])
    inertia = inertial.find("inertia").attrib
    matrix = [
        [float(inertia["ixx"]), float(inertia["ixy"]), float(inertia["ixz"])],
        [float(inertia["ixy"]), float(inertia["iyy"]), float(inertia["iyz"])],
        [float(inertia["ixz"]), float(inertia["iyz"]), float(inertia["izz"])],
    ]
    return {
        "mass": mass,
        "com": _parse_vector(origin.attrib.get("xyz")) if origin is not None else [0.0] * 3,
        "inertial_rotation": (
            _rpy_matrix(_parse_vector(origin.attrib.get("rpy")))
            if origin is not None
            else _rpy_matrix([0.0, 0.0, 0.0])
        ),
        "inertia": matrix,
    }


def load_arm_dynamics_description(urdf_path, arm_joint_names):
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    child_to_joint = {
        joint.find("child").attrib["link"]: joint for joint in root.findall("joint")
    }

    first_arm_joint = joints[arm_joint_names[0]]
    arm_base_link = first_arm_joint.find("parent").attrib["link"]
    fixed_path = []
    cursor = arm_base_link
    while cursor != "BASE_LINK":
        fixed_joint = child_to_joint.get(cursor)
        if fixed_joint is None or fixed_joint.attrib.get("type") != "fixed":
            raise ValueError("无法从 BASE_LINK 找到机械臂固定安装链")
        fixed_path.append(fixed_joint)
        cursor = fixed_joint.find("parent").attrib["link"]
    fixed_path.reverse()

    base_to_arm = (_rpy_matrix([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])
    for fixed_joint in fixed_path:
        base_to_arm = _compose_transform(base_to_arm, _joint_origin(fixed_joint))

    origins = []
    axes = []
    bodies = []
    for index, name in enumerate(arm_joint_names):
        joint = joints[name]
        origin = _joint_origin(joint)
        if index == 0:
            origin = _compose_transform(base_to_arm, origin)
        origins.append(origin)
        axis_node = joint.find("axis")
        axes.append(_parse_vector(axis_node.attrib.get("xyz")))
        child_name = joint.find("child").attrib["link"]
        bodies.append(_link_inertial(links[child_name]))

    last_child = joints[arm_joint_names[-1]].find("child").attrib["link"]
    fixed_tools = []
    parent_name = last_child
    while True:
        candidates = [
            joint
            for joint in joints.values()
            if joint.find("parent").attrib["link"] == parent_name
            and joint.attrib.get("type") == "fixed"
        ]
        if not candidates:
            break
        if len(candidates) > 1:
            raise ValueError("末端固定链存在分支，无法确定工具质量链")
        fixed_joint = candidates[0]
        child_name = fixed_joint.find("child").attrib["link"]
        if child_name not in links or links[child_name].find("inertial") is None:
            break
        fixed_tools.append(
            {
                "origin": _joint_origin(fixed_joint),
                "body": _link_inertial(links[child_name]),
            }
        )
        parent_name = child_name

    return {
        "origins": origins,
        "axes": axes,
        "bodies": bodies,
        "fixed_tools": fixed_tools,
    }


def quaternion_xyzw_to_matrix(quaternion):
    x, y, z, w = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _axis_angle_matrix(axis, angle):
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(axis.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=angle.dtype, device=angle.device)
    identity = identity.expand(angle.shape + (3, 3))
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]
    return identity + sin * skew + (1.0 - cos) * (skew @ skew)


def _matvec(matrix, vector):
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


class ZgwtArmDynamics:
    def __init__(self, description, device, dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.num_joints = len(description["origins"])
        if self.num_joints != 6:
            raise ValueError("ZGWSARM 动力学必须包含六个关节")

        self.origin_rotations = torch.tensor(
            [item[0] for item in description["origins"]],
            dtype=dtype,
            device=device,
        )
        self.origin_translations = torch.tensor(
            [item[1] for item in description["origins"]],
            dtype=dtype,
            device=device,
        )
        self.axes = torch.tensor(
            description["axes"], dtype=dtype, device=device
        )
        all_bodies = list(description["bodies"])
        self.fixed_tool_origins = [
            (
                torch.tensor(item["origin"][0], dtype=dtype, device=device),
                torch.tensor(item["origin"][1], dtype=dtype, device=device),
            )
            for item in description["fixed_tools"]
        ]
        all_bodies.extend(item["body"] for item in description["fixed_tools"])
        self.body_joint_owners = list(range(6)) + [5] * len(
            description["fixed_tools"]
        )
        owners = torch.tensor(
            self.body_joint_owners, dtype=torch.long, device=device
        )
        joints = torch.arange(self.num_joints, dtype=torch.long, device=device)
        self.downstream_mask = (owners[:, None] >= joints[None, :]).to(dtype)
        self.masses = torch.tensor(
            [body["mass"] for body in all_bodies], dtype=dtype, device=device
        )
        self.com_offsets = torch.tensor(
            [body["com"] for body in all_bodies], dtype=dtype, device=device
        )
        self.inertial_rotations = torch.tensor(
            [body["inertial_rotation"] for body in all_bodies],
            dtype=dtype,
            device=device,
        )
        self.inertias = torch.tensor(
            [body["inertia"] for body in all_bodies],
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_urdf(cls, urdf_path, arm_joint_names, device, dtype=torch.float):
        return cls(
            load_arm_dynamics_description(urdf_path, arm_joint_names),
            device=device,
            dtype=dtype,
        )

    def _forward_kinematics(
        self,
        q,
        dq=None,
        qdd=None,
        base_quaternion=None,
        base_angular_velocity=None,
        gravity=None,
    ):
        batch = q.shape[0]
        zeros = torch.zeros(batch, 3, dtype=q.dtype, device=q.device)
        if dq is None:
            dq = torch.zeros_like(q)
        if qdd is None:
            qdd = torch.zeros_like(q)
        if base_quaternion is None:
            base_rotation = torch.eye(
                3, dtype=q.dtype, device=q.device
            ).expand(batch, 3, 3)
        else:
            base_rotation = quaternion_xyzw_to_matrix(base_quaternion)
        if base_angular_velocity is None:
            base_angular_velocity = zeros
        if gravity is None:
            gravity = zeros
        if gravity.ndim == 1:
            gravity = gravity.unsqueeze(0).expand(batch, 3)

        parent_position = zeros
        parent_rotation = base_rotation
        parent_velocity = zeros
        parent_omega = base_angular_velocity
        # 不使用机身线/角加速度前馈；-gravity 是 RNEA 的虚拟基座加速度。
        parent_acceleration = -gravity
        parent_alpha = zeros

        joint_positions = []
        joint_axes = []
        body_states = []
        link_states = []

        for index in range(self.num_joints):
            origin_r = self.origin_rotations[index].expand(batch, 3, 3)
            origin_t = self.origin_translations[index].expand(batch, 3)
            joint_position = parent_position + _matvec(
                parent_rotation, origin_t
            )
            pre_rotation = parent_rotation @ origin_r
            axis = _matvec(pre_rotation, self.axes[index].expand(batch, 3))
            joint_rotation = _axis_angle_matrix(
                self.axes[index].expand(batch, 3), q[:, index]
            )
            child_rotation = pre_rotation @ joint_rotation

            offset = joint_position - parent_position
            joint_velocity = parent_velocity + torch.cross(
                parent_omega, offset, dim=1
            )
            joint_acceleration = (
                parent_acceleration
                + torch.cross(parent_alpha, offset, dim=1)
                + torch.cross(
                    parent_omega,
                    torch.cross(parent_omega, offset, dim=1),
                    dim=1,
                )
            )
            child_omega = parent_omega + axis * dq[:, index : index + 1]
            child_alpha = (
                parent_alpha
                + axis * qdd[:, index : index + 1]
                + torch.cross(
                    parent_omega,
                    axis * dq[:, index : index + 1],
                    dim=1,
                )
            )

            com_offset = _matvec(
                child_rotation, self.com_offsets[index].expand(batch, 3)
            )
            com_position = joint_position + com_offset
            com_acceleration = (
                joint_acceleration
                + torch.cross(child_alpha, com_offset, dim=1)
                + torch.cross(
                    child_omega,
                    torch.cross(child_omega, com_offset, dim=1),
                    dim=1,
                )
            )
            inertial_rotation = (
                child_rotation
                @ self.inertial_rotations[index].expand(batch, 3, 3)
            )
            body_states.append(
                (
                    com_position,
                    child_omega,
                    child_alpha,
                    com_acceleration,
                    inertial_rotation,
                )
            )
            link_states.append(
                (
                    joint_position,
                    child_rotation,
                    joint_velocity,
                    child_omega,
                    joint_acceleration,
                    child_alpha,
                )
            )
            joint_positions.append(joint_position)
            joint_axes.append(axis)

            parent_position = joint_position
            parent_rotation = child_rotation
            parent_velocity = joint_velocity
            parent_omega = child_omega
            parent_acceleration = joint_acceleration
            parent_alpha = child_alpha

        tool_position, tool_rotation, tool_velocity, tool_omega, tool_accel, tool_alpha = (
            link_states[-1]
        )
        for tool_index, (fixed_r, fixed_t) in enumerate(self.fixed_tool_origins):
            fixed_r = fixed_r.expand(batch, 3, 3)
            fixed_t = fixed_t.expand(batch, 3)
            offset = _matvec(tool_rotation, fixed_t)
            tool_position = tool_position + offset
            tool_velocity = tool_velocity + torch.cross(
                tool_omega, offset, dim=1
            )
            tool_accel = (
                tool_accel
                + torch.cross(tool_alpha, offset, dim=1)
                + torch.cross(
                    tool_omega,
                    torch.cross(tool_omega, offset, dim=1),
                    dim=1,
                )
            )
            tool_rotation = tool_rotation @ fixed_r
            body_index = 6 + tool_index
            com_offset = _matvec(
                tool_rotation, self.com_offsets[body_index].expand(batch, 3)
            )
            body_states.append(
                (
                    tool_position + com_offset,
                    tool_omega,
                    tool_alpha,
                    tool_accel
                    + torch.cross(tool_alpha, com_offset, dim=1)
                    + torch.cross(
                        tool_omega,
                        torch.cross(tool_omega, com_offset, dim=1),
                        dim=1,
                    ),
                    tool_rotation
                    @ self.inertial_rotations[body_index].expand(
                        batch, 3, 3
                    ),
                )
            )

        return joint_positions, joint_axes, body_states, link_states[-1][:2]

    def inverse_dynamics(
        self,
        q,
        dq,
        qdd,
        base_quaternion=None,
        base_angular_velocity=None,
        gravity=None,
    ):
        joint_positions, joint_axes, body_states, _ = self._forward_kinematics(
            q,
            dq=dq,
            qdd=qdd,
            base_quaternion=base_quaternion,
            base_angular_velocity=base_angular_velocity,
            gravity=gravity,
        )
        batch = q.shape[0]
        body_positions = []
        body_forces = []
        body_moments = []
        for body_index, state in enumerate(body_states):
            com_position, omega, alpha, acceleration, inertial_rotation = state
            mass = self.masses[body_index]
            inertia_world = (
                inertial_rotation
                @ self.inertias[body_index].expand(batch, 3, 3)
                @ inertial_rotation.transpose(1, 2)
            )
            force = mass * acceleration
            angular_momentum = _matvec(inertia_world, omega)
            moment_at_com = _matvec(inertia_world, alpha) + torch.cross(
                omega, angular_momentum, dim=1
            )
            body_positions.append(com_position)
            body_forces.append(force)
            body_moments.append(moment_at_com)

        body_positions = torch.stack(body_positions, dim=1)
        body_forces = torch.stack(body_forces, dim=1)
        body_moments = torch.stack(body_moments, dim=1)
        joint_positions = torch.stack(joint_positions, dim=1)
        joint_axes = torch.stack(joint_axes, dim=1)
        offsets = body_positions[:, :, None, :] - joint_positions[:, None, :, :]
        moments_at_joints = body_moments[:, :, None, :] + torch.cross(
            offsets, body_forces[:, :, None, :].expand_as(offsets), dim=3
        )
        projected = torch.sum(
            moments_at_joints * joint_axes[:, None, :, :], dim=3
        )
        return torch.sum(projected * self.downstream_mask, dim=1)

    def end_kinematics(self, q, base_quaternion=None):
        joint_positions, joint_axes, _, end_state = self._forward_kinematics(
            q, base_quaternion=base_quaternion
        )
        end_position, end_rotation = end_state
        jacobian = torch.zeros(
            q.shape[0], 6, 6, dtype=q.dtype, device=q.device
        )
        for index in range(6):
            jacobian[:, :3, index] = torch.cross(
                joint_axes[index],
                end_position - joint_positions[index],
                dim=1,
            )
            jacobian[:, 3:, index] = joint_axes[index]
        return end_position, end_rotation, jacobian
