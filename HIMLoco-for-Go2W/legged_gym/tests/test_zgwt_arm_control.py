import ast
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

    @staticmethod
    def _formula_inputs():
        q = torch.tensor(
            [0.2, 0.7, -1.1, 0.3, -0.4, 0.6], dtype=torch.double
        )
        dq = torch.tensor(
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=torch.double
        )
        q_ref = torch.tensor(
            [-0.1, 1.0, -1.4, 0.5, -0.2, 0.1], dtype=torch.double
        )
        dq_ref = torch.tensor(
            [-0.2, 0.1, -0.1, 0.2, -0.3, 0.4], dtype=torch.double
        )
        qdd_ref = torch.tensor(
            [0.3, -0.1, 0.2, -0.4, 0.6, -0.5], dtype=torch.double
        )
        factor = torch.tensor(
            [
                [1.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 1.2, 0.0, 0.0, 0.0, 0.0],
                [0.1, -0.2, 1.1, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.2, 1.3, 0.0, 0.0],
                [-0.1, 0.0, 0.1, 0.2, 1.0, 0.0],
                [0.2, -0.1, 0.0, 0.1, 0.2, 1.2],
            ],
            dtype=torch.double,
        )
        mass_matrix = factor @ factor.T
        bias = torch.tensor(
            [1.0, -2.0, 0.5, 1.5, -0.7, 0.9], dtype=torch.double
        )
        kp = torch.full((6,), 500.0, dtype=torch.double)
        kd = torch.full((6,), 45.0, dtype=torch.double)
        return q, dq, q_ref, dq_ref, qdd_ref, mass_matrix, bias, kp, kd

    def test_only_control_law_places_feedback_inside_mass_matrix(self):
        q, dq, q_ref, dq_ref, qdd_ref, mass, bias, kp, kd = (
            self._formula_inputs()
        )
        acceleration = UTILS.inertia_normalized_desired_acceleration(
            qdd_ref,
            q_ref - q,
            dq_ref - dq,
            kp,
            kd,
        )
        actual = bias + mass @ acceleration
        feedback = kp * (q_ref - q) + kd * (dq_ref - dq)
        expected = bias + mass @ (qdd_ref + feedback)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, expected + feedback))

    def test_legacy_control_law_switches_are_absent(self):
        root = Path(__file__).parents[2]
        paths = (
            root / "legged_gym/envs/zgwt_arm/zgwt_arm_control_utils.py",
            root / "legged_gym/envs/zgwt_arm/zgwt_arm_config.py",
            root / "legged_gym/envs/zgwt_arm/zgwt_arm_robot.py",
            root / "config/robots/zgwsarm/arm_control_contract.yaml",
        )
        forbidden = (
            "physical_joint_impedance",
            "joint_impedance_outside_mass_matrix",
            "arm.control_law",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} 仍包含 {token}")

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
            with self.assertRaisesRegex(ValueError, "没有可用于机械臂姿态库"):
                UTILS.load_simulation_verified_poses(
                    str(path), self.LOWER, self.UPPER, 0.15, 0.02
                )

    def test_pose_library_task_samples_only_on_reset(self):
        env_dir = Path(__file__).parents[1] / "envs" / "zgwt_arm"
        config_tree = ast.parse(
            (env_dir / "zgwt_arm_config.py").read_text(encoding="utf-8")
        )
        pose_library_config = next(
            node
            for node in config_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ZGWTDanceArmPoseLibraryCfg"
        )
        arm_config = next(
            node
            for node in pose_library_config.body
            if isinstance(node, ast.ClassDef) and node.name == "arm"
        )
        pose_mode = next(
            node
            for node in arm_config.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pose_mode"
                for target in node.targets
            )
        )
        self.assertEqual(ast.literal_eval(pose_mode.value), "library")

        robot_tree = ast.parse(
            (env_dir / "zgwt_arm_robot.py").read_text(encoding="utf-8")
        )
        callers = []
        for function in (
            node
            for node in ast.walk(robot_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_sample_arm_targets"
                for node in ast.walk(function)
            ):
                callers.append(function.name)
        self.assertEqual(callers, ["_reset_dofs"])

    def test_armstand_reward_randomization_and_policy_configuration_contract(self):
        config_path = (
            Path(__file__).parents[1]
            / "envs"
            / "zgwt_arm"
            / "zgwt_arm_config.py"
        )
        tree = ast.parse(config_path.read_text(encoding="utf-8"))

        def class_node(parent, name):
            return next(
                node
                for node in parent.body
                if isinstance(node, ast.ClassDef) and node.name == name
            )

        def assignment_value(parent, name):
            assignment = next(
                node
                for node in parent.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in node.targets
                )
            )
            value = assignment.value
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
                return ast.literal_eval(value.left) * ast.literal_eval(value.right)
            return ast.literal_eval(value)

        armstand_v1 = class_node(tree, "ZGWTDanceArmCfg")
        arm_config = class_node(armstand_v1, "arm")
        self.assertEqual(assignment_value(arm_config, "kp"), [500.0] * 6)
        self.assertEqual(assignment_value(arm_config, "kd"), [45.0] * 6)

        commands = class_node(armstand_v1, "commands")
        ranges = class_node(commands, "ranges")
        self.assertEqual(
            assignment_value(ranges, "body_height"), [0.48, 0.48]
        )

        pose_library_stage = class_node(tree, "ZGWTDanceArmPoseLibraryCfg")
        aligned_arm = class_node(pose_library_stage, "arm")
        self.assertEqual(assignment_value(aligned_arm, "pose_mode"), "library")

        ppo = class_node(tree, "ZGWTDanceArmCfgPPO")
        runner = class_node(ppo, "runner")
        self.assertEqual(
            assignment_value(runner, "run_name"), "armstand_aligned_v1"
        )
        self.assertTrue(assignment_value(runner, "reset_iteration_on_load"))
        self.assertTrue(assignment_value(runner, "load_actor_only"))
        self.assertFalse(assignment_value(runner, "load_optimizer"))

        pose_library_ppo = class_node(tree, "ZGWTDanceArmPoseLibraryCfgPPO")
        aligned_runner = class_node(pose_library_ppo, "runner")
        self.assertEqual(
            assignment_value(aligned_runner, "run_name"),
            "armstand_pose_library_aligned_v1",
        )
        self.assertEqual(assignment_value(aligned_runner, "checkpoint"), 20000)

        rewards = class_node(armstand_v1, "rewards")
        scales = class_node(rewards, "scales")
        self.assertEqual(
            assignment_value(scales, "tracking_body_roll"), 6.5
        )
        self.assertEqual(
            assignment_value(scales, "tracking_body_pitch"), 6.5
        )
        self.assertEqual(assignment_value(scales, "tracking_body_height"), 9.0)
        self.assertEqual(assignment_value(scales, "stance_coordination"), 4.0)
        self.assertEqual(assignment_value(scales, "feet_outward"), -2.0)

        domain_rand = class_node(armstand_v1, "domain_rand")
        self.assertTrue(assignment_value(domain_rand, "randomize_link_mass"))
        self.assertEqual(
            assignment_value(domain_rand, "com_displacement_range"),
            [-0.015, 0.015],
        )
        self.assertFalse(
            assignment_value(domain_rand, "correlate_payload_and_com")
        )
        self.assertEqual(
            assignment_value(domain_rand, "motor_strength_range"),
            [0.95, 1.05],
        )

        env = class_node(armstand_v1, "env")
        self.assertEqual(assignment_value(env, "num_actions"), 16)
        one_step_observation = 3 + 3 + 6 + 16 + 16 + 16
        critic_observation = one_step_observation + 3 + 3 + 12
        self.assertEqual(one_step_observation * 6, 360)
        self.assertEqual(critic_observation, 78)

    def test_actual_pose_library_large_sampling_stays_simulation_safe(self):
        pose_path = (
            Path(__file__).parents[2]
            / "config"
            / "robots"
            / "zgwsarm"
            / "arm_safe_static_poses.csv"
        )
        poses = UTILS.load_simulation_verified_poses(
            str(pose_path), self.LOWER, self.UPPER, 0.15, 0.02
        )
        self.assertEqual(len(poses), 455)
        weights = UTILS.normalized_pose_weights(
            poses,
            {
                "home_or_folded": 0.30,
                "workspace_midrange": 0.40,
                "directional_extension": 0.20,
                "near_allowed_boundary": 0.10,
            },
        )
        generator = torch.Generator().manual_seed(20260803)
        sampled_indices = torch.multinomial(
            torch.tensor(weights, dtype=torch.double),
            100_000,
            replacement=True,
            generator=generator,
        )
        self.assertEqual(sampled_indices.numel(), 100_000)
        self.assertGreater(len(poses), 0)
        for pose in poses:
            self.assertTrue(pose.simulation_verified)
            self.assertTrue(pose.collision_checked)
            self.assertTrue(pose.singularity_checked)
            self.assertGreaterEqual(pose.min_joint_margin_rad, 0.15)
            self.assertGreaterEqual(pose.min_jacobian_sigma, 0.02)
            self.assertGreater(
                UTILS.joint_limit_margin(pose.q, self.LOWER, self.UPPER),
                0.15 - 1.0e-9,
            )
            self.assertNotEqual(tuple(pose.q), (0.0,) * 6)
            self.assertFalse(pose.hardware_verified)


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
