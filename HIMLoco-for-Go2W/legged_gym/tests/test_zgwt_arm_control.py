import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


UTILS_PATH = (
    Path(__file__).parents[1]
    / "envs"
    / "zgwt_arm"
    / "zgwt_arm_control_utils.py"
)
SPEC = importlib.util.spec_from_file_location("zgwt_arm_control_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)

DYNAMICS_PATH = (
    Path(__file__).parents[1]
    / "envs"
    / "zgwt_arm"
    / "zgwt_arm_dynamics.py"
)
DYNAMICS_SPEC = importlib.util.spec_from_file_location(
    "zgwt_arm_dynamics", DYNAMICS_PATH
)
DYNAMICS = importlib.util.module_from_spec(DYNAMICS_SPEC)
DYNAMICS_SPEC.loader.exec_module(DYNAMICS)


class TestZGWTArmControl(unittest.TestCase):
    LOWER = [-2.618, 0.0, -2.9671, -1.57, -1.57, -6.28]
    UPPER = [2.618, 3.14, 0.0, 1.57, 1.57, 6.28]

    def test_torque_rate_limit_matches_two_nm_per_physics_step(self):
        desired = torch.full((2, 6), 100.0)
        previous = torch.zeros_like(desired)
        limited = UTILS.torque_rate_limit(desired, previous, 1000.0, 0.002)
        torch.testing.assert_close(limited, torch.full_like(desired, 2.0))

    def test_joint_limit_margin_detects_home_boundary(self):
        margin = UTILS.joint_limit_margin(
            [0.0] * 6, self.LOWER, self.UPPER
        )
        self.assertEqual(margin, 0.0)

    def test_limited_reference_respects_velocity_and_acceleration(self):
        command = torch.full((2, 6), 1.0)
        position = torch.zeros_like(command)
        velocity = torch.zeros_like(command)
        velocity_limit = torch.full((1, 6), 0.8)
        acceleration_limit = torch.full((1, 6), 2.0)
        for _ in range(100):
            previous_velocity = velocity
            position, velocity, acceleration = (
                UTILS.advance_limited_joint_reference(
                    command,
                    position,
                    velocity,
                    velocity_limit,
                    acceleration_limit,
                    0.002,
                    0.005,
                )
            )
            self.assertLessEqual(velocity.abs().max().item(), 0.8 + 1.0e-6)
            self.assertLessEqual(
                acceleration.abs().max().item(), 2.0 + 1.0e-6
            )
            torch.testing.assert_close(
                acceleration, (velocity - previous_velocity) / 0.002
            )
        for _ in range(3000):
            position, velocity, acceleration = (
                UTILS.advance_limited_joint_reference(
                    command,
                    position,
                    velocity,
                    velocity_limit,
                    acceleration_limit,
                    0.002,
                    0.005,
                )
            )
        self.assertLess(torch.max(torch.abs(command - position)).item(), 1.0e-4)
        self.assertLess(torch.max(torch.abs(velocity)).item(), 1.0e-3)

    def test_pose_loader_rechecks_flags_margin_and_sigma(self):
        q = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        margin = UTILS.joint_limit_margin(q, self.LOWER, self.UPPER)
        row = {
            "timestamp_s": "0.0",
            "pose_id": "verified_pose",
            "category": "workspace_midrange",
            "source": "unit_test",
            "simulation_verified": "true",
            "hardware_verified": "false",
            "collision_checked": "true",
            "singularity_checked": "true",
            "min_joint_margin_rad": f"{margin:.9f}",
            "min_jacobian_sigma": "0.08",
        }
        row.update({f"arm_q{i + 1}": value for i, value in enumerate(q)})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=UTILS.REQUIRED_POSE_COLUMNS
                )
                writer.writeheader()
                writer.writerow(row)
            poses = UTILS.load_simulation_verified_poses(
                str(path), self.LOWER, self.UPPER, 0.15, 0.02
            )

        self.assertEqual(len(poses), 1)
        self.assertFalse(poses[0].hardware_verified)

    def test_empty_verified_pose_set_is_rejected(self):
        row = {
            "timestamp_s": "0.0",
            "pose_id": "unsafe_home",
            "category": "home_or_folded",
            "source": "unit_test",
            "simulation_verified": "false",
            "hardware_verified": "false",
            "collision_checked": "false",
            "singularity_checked": "false",
            "min_joint_margin_rad": "0.0",
            "min_jacobian_sigma": "0.0",
        }
        row.update({name: "0.0" for name in UTILS.ARM_JOINT_COLUMNS})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=UTILS.REQUIRED_POSE_COLUMNS
                )
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "没有可用于 ArmStandV2"):
                UTILS.load_simulation_verified_poses(
                    str(path), self.LOWER, self.UPPER, 0.15, 0.02
                )


class TestZGWTArmDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        urdf = (
            Path(__file__).parents[2]
            / "resources"
            / "robots"
            / "zgwsarm"
            / "zgwsarm.urdf"
        )
        cls.model = DYNAMICS.ZgwtArmDynamics.from_urdf(
            str(urdf),
            [f"ROBOT_ARM_JOINT{i}" for i in range(1, 7)],
            device="cpu",
            dtype=torch.double,
        )

    def test_gravity_torque_matches_potential_gradient(self):
        q = torch.tensor(
            [[0.2, 1.0, -1.2, 0.3, -0.4, 0.5]], dtype=torch.double
        )
        zero = torch.zeros_like(q)
        quaternion = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.double
        )
        torque = self.model.inverse_dynamics(
            q,
            zero,
            zero,
            base_quaternion=quaternion,
            base_angular_velocity=torch.zeros(1, 3, dtype=torch.double),
            gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.double),
        )

        def potential(position):
            _, _, bodies, _ = self.model._forward_kinematics(
                position, base_quaternion=quaternion
            )
            return sum(
                self.model.masses[index] * 9.81 * body[0][:, 2]
                for index, body in enumerate(bodies)
            )

        epsilon = 1.0e-6
        gradient = []
        for index in range(6):
            upper = q.clone()
            lower = q.clone()
            upper[:, index] += epsilon
            lower[:, index] -= epsilon
            gradient.append(
                ((potential(upper) - potential(lower)) / (2 * epsilon)).item()
            )
        torch.testing.assert_close(
            torque[0], torch.tensor(gradient, dtype=torch.double)
        )

    def test_mass_matrix_is_symmetric_positive_definite(self):
        q = torch.tensor(
            [[0.2, 1.0, -1.2, 0.3, -0.4, 0.5]], dtype=torch.double
        )
        zero = torch.zeros_like(q)
        columns = []
        for index in range(6):
            acceleration = zero.clone()
            acceleration[:, index] = 1.0
            columns.append(
                self.model.inverse_dynamics(
                    q,
                    zero,
                    acceleration,
                    gravity=torch.zeros(3, dtype=torch.double),
                )[0]
            )
        mass_matrix = torch.stack(columns, dim=1)
        torch.testing.assert_close(mass_matrix, mass_matrix.T)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(mass_matrix) > 0.0))

    def test_batch_result_is_independent_of_other_environments(self):
        first = torch.tensor(
            [0.2, 1.0, -1.2, 0.3, -0.4, 0.5], dtype=torch.double
        )
        batch = torch.stack(
            (
                first,
                torch.tensor(
                    [-1.0, 0.7, -2.0, -0.2, 0.6, -1.5],
                    dtype=torch.double,
                ),
            )
        )
        zero_batch = torch.zeros_like(batch)
        batch_torque = self.model.inverse_dynamics(
            batch,
            zero_batch,
            zero_batch,
            gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.double),
        )
        single_torque = self.model.inverse_dynamics(
            batch[:1],
            zero_batch[:1],
            zero_batch[:1],
            gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.double),
        )
        torch.testing.assert_close(batch_torque[0], single_torque[0])


if __name__ == "__main__":
    unittest.main()
