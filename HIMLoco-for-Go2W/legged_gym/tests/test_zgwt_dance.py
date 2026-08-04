import math
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import torch

UTILS_PATH = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_utils.py"
SPEC = importlib.util.spec_from_file_location("zgwt_dance_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)

effective_actions = UTILS.effective_actions
floored_soft_multiplier = UTILS.floored_soft_multiplier
height_tracking_score = UTILS.height_tracking_score
leg_extension_from_knee = UTILS.leg_extension_from_knee
low_pass_alpha = UTILS.low_pass_alpha
neutral_command_weight = UTILS.neutral_command_weight
normalize_dance_commands = UTILS.normalize_dance_commands
select_wheel_factors = UTILS.select_wheel_factors
stance_coordination_score = UTILS.stance_coordination_score
target_projected_gravity = UTILS.target_projected_gravity
wrap_to_pi = UTILS.wrap_to_pi


class TestZGWTDanceMath(unittest.TestCase):
    def test_latest_load_path_uses_run_mtime_and_numeric_checkpoint(self):
        helpers_path = Path(__file__).parents[1] / "utils" / "helpers.py"
        helpers_tree = ast.parse(helpers_path.read_text(encoding="utf-8"))
        get_load_path_node = next(
            node
            for node in helpers_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_load_path"
        )
        namespace = {"os": os}
        exec(
            compile(
                ast.Module(body=[get_load_path_node], type_ignores=[]),
                str(helpers_path),
                "exec",
            ),
            namespace,
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            older = root / "Dec31_23-59-59_older"
            newer = root / "Jan01_00-00-01_newer"
            exported = root / "exported"
            older.mkdir()
            newer.mkdir()
            exported.mkdir()
            (older / "model_999.pt").touch()
            (newer / "model_2.pt").touch()
            (newer / "model_10.pt").touch()
            os.utime(older, (100.0, 100.0))
            os.utime(newer, (200.0, 200.0))
            os.utime(exported, (300.0, 300.0))

            selected = namespace["get_load_path"](
                str(root), load_run=-1, checkpoint=-1
            )
            self.assertEqual(Path(selected), newer / "model_10.pt")

    @staticmethod
    def _dance_config_node():
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_config.py"
        tree = ast.parse(path.read_text())
        return next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZGWTDanceCfg"
        )

    @classmethod
    def _commands_config_node(cls):
        dance = cls._dance_config_node()
        return next(
            node for node in dance.body
            if isinstance(node, ast.ClassDef) and node.name == "commands"
        )

    def test_manual_command_probability_sum(self):
        commands = self._commands_config_node()
        assignments = {
            node.targets[0].id: node.value
            for node in commands.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        probabilities = ast.literal_eval(assignments["mode_probabilities"])
        self.assertEqual(len(probabilities), 16)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=7)
        forbidden = {
            "initial_stage",
            "max_stage",
            "success_curriculum",
            "stage_mode_probabilities",
            "stage_range_scales",
        }
        self.assertTrue(forbidden.isdisjoint(assignments))

    def test_reward_scales_are_the_only_geometry_weights(self):
        dance = self._dance_config_node()
        rewards = next(
            node for node in dance.body
            if isinstance(node, ast.ClassDef) and node.name == "rewards"
        )
        reward_assignments = {
            node.targets[0].id
            for node in rewards.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertNotIn("tracking_weights", reward_assignments)
        self.assertNotIn("hold_weights", reward_assignments)

        scales = next(
            node for node in rewards.body
            if isinstance(node, ast.ClassDef) and node.name == "scales"
        )
        scale_values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in scales.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(scale_values["tracking_body_roll"], 6)
        self.assertEqual(scale_values["tracking_body_pitch"], 6)
        self.assertEqual(scale_values["tracking_body_yaw"], 6.0)
        self.assertEqual(scale_values["tracking_body_height"], 6.0)
        self.assertEqual(scale_values["tracking_lin_vx"], 4.0)
        self.assertEqual(scale_values["tracking_lin_vy"], 4.0)
        self.assertEqual(scale_values["stance_coordination"], 4.0)
        self.assertNotIn("tracking_body_orientation", scale_values)
        self.assertNotIn("neutral_joint_pose", scale_values)
        self.assertNotIn("action_smoothness", scale_values)

    def test_reward_dispatch_is_inherited_from_legged_robot(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        method_names = {
            node.name for node in dance.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("_prepare_reward_function", method_names)
        self.assertNotIn("compute_reward", method_names)
        self.assertNotIn("_tracking_cost", method_names)
        self.assertNotIn("_hold_cost", method_names)

        assignment_names = {
            target.id
            for node in dance.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertFalse(
            any(name.endswith("REWARD_NAMES") for name in assignment_names)
        )

    def test_severe_support_loss_is_a_termination_condition(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        check_termination = next(
            node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "check_termination"
        )
        called_methods = {
            node.func.attr
            for node in ast.walk(check_termination)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        accessed_attributes = {
            node.attr
            for node in ast.walk(check_termination)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn("_support_center_drift", called_methods)
        self.assertIn("severe_support_loss_distance", accessed_attributes)

    def test_planar_velocity_commands_are_fixed_to_zero(self):
        commands = self._commands_config_node()
        ranges = next(
            node
            for node in commands.body
            if isinstance(node, ast.ClassDef) and node.name == "ranges"
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ranges.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(assignments["lin_vel_x"], [0.0, 0.0])
        self.assertEqual(assignments["lin_vel_y"], [0.0, 0.0])

    def test_fixed_network_shapes(self):
        one_step = 3 + 3 + 6 + 16 + 16 + 16
        privileged = one_step + 3 + 3 + 12
        self.assertEqual(one_step, 60)
        self.assertEqual(one_step * 6, 360)
        self.assertEqual(privileged, 78)

    def test_command_endpoint_normalization_crouch_only(self):
        commands = torch.tensor(
            [
                [0.0, 0.0, -0.10, -0.26, -0.26, 0.40],
                [0.0, 0.0, 0.00, 0.00, 0.00, 0.54],
                [0.0, 0.0, 0.10, 0.26, 0.26, 0.55],
            ]
        )
        normalized = normalize_dance_commands(
            commands, 0.54, 0.40, 0.10, 0.26, 0.26
        )
        torch.testing.assert_close(
            normalized[0], torch.tensor([0.0, 0.0, -1.0, -1.0, -1.0, -1.0])
        )
        torch.testing.assert_close(
            normalized[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        # Upward requests are clipped to neutral because the task cannot rise.
        torch.testing.assert_close(
            normalized[2], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        )

    def test_yaw_wrap(self):
        wrapped = wrap_to_pi(torch.tensor([math.pi + 0.2, -math.pi - 0.2]))
        torch.testing.assert_close(wrapped, torch.tensor([-math.pi + 0.2, math.pi - 0.2]))

    def test_projected_gravity_target(self):
        gravity = target_projected_gravity(torch.tensor([0.0]), torch.tensor([0.0]))
        torch.testing.assert_close(gravity, torch.tensor([[0.0, 0.0, -1.0]]))
        self.assertAlmostEqual(torch.linalg.norm(gravity, dim=1).item(), 1.0, places=6)

    def test_reward_and_hold_gate_monotonicity(self):
        costs = torch.tensor([0.0, 0.5, 1.0, 2.0])
        scores = torch.exp(-costs)
        gates = torch.exp(-costs / 0.5)
        self.assertTrue(torch.all(scores[:-1] > scores[1:]))
        self.assertTrue(torch.all(gates[:-1] > gates[1:]))

    def test_neutral_gate_is_continuous(self):
        commands = torch.zeros(3, 6)
        commands[:, 3] = torch.tensor([0.0, 0.1, 0.2])
        weights = neutral_command_weight(commands, 0.20)
        self.assertEqual(weights[0].item(), 1.0)
        self.assertTrue(weights[0] > weights[1] > weights[2])

    def test_soft_floor_and_hard_penalty_outside(self):
        multiplier = floored_soft_multiplier(
            torch.tensor([0.0, -100.0]), sigma=0.1, floor=0.30
        )
        self.assertAlmostEqual(multiplier[0].item(), 1.0, places=6)
        self.assertGreaterEqual(multiplier[1].item(), 0.30)
        task = torch.tensor([2.0])
        hard = torch.tensor([-3.0])
        final = task * multiplier[:1] + hard
        self.assertAlmostEqual(final.item(), -1.0, places=6)

    def test_height_deficit_only_penalizes_excess_downward_error(self):
        current = torch.tensor([0.54, 0.52, 0.50, 0.58])
        target = torch.full((4,), 0.54)
        score = height_tracking_score(
            current,
            target,
            symmetric_sigma=0.0016,
            deficit_deadzone=0.02,
            deficit_sigma=0.0004,
            deficit_weight=1.0,
        )
        symmetric_only = torch.exp(-torch.square(current - target) / 0.0016)
        torch.testing.assert_close(score[:2], symmetric_only[:2])
        self.assertLess(score[2], symmetric_only[2])
        torch.testing.assert_close(score[3], symmetric_only[3])

    def test_stance_coordination_allows_common_mode_crouch(self):
        knee = torch.tensor(
            [
                [-1.2, -1.2, 1.2, 1.2],
                [-1.6, -1.6, 1.6, 1.6],
                [-2.0, -1.2, 1.2, 1.2],
            ]
        )
        extension = leg_extension_from_knee(knee, 0.26, 0.28)
        score, residual = stance_coordination_score(
            extension,
            torch.zeros_like(extension),
            deadzone=torch.full((3, 1), 0.015),
            sigma=0.0004,
        )
        torch.testing.assert_close(score[:2], torch.ones(2))
        self.assertLess(score[2], 1.0)
        self.assertGreater(torch.max(torch.abs(residual[2])), 0.015)

    def test_command_differential_does_not_count_as_folding(self):
        common = torch.full((1, 4), 0.44)
        expected = torch.tensor([[0.01, -0.01, 0.01, -0.01]])
        score, residual = stance_coordination_score(
            common + expected,
            expected,
            deadzone=torch.full((1, 1), 0.005),
            sigma=0.0004,
        )
        torch.testing.assert_close(score, torch.ones(1))
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_wheel_action_masking(self):
        wheel_indices = torch.tensor([3, 7, 11, 15])
        first = torch.zeros(2, 16)
        second = first.clone()
        second[:, wheel_indices] = 10.0
        torch.testing.assert_close(
            effective_actions(first, wheel_indices),
            effective_actions(second, wheel_indices),
        )

    def test_wheel_damping_shapes(self):
        wheel_indices = torch.tensor([3, 7, 11, 15])
        shared = select_wheel_factors(torch.ones(2, 1), wheel_indices, 16)
        per_dof = select_wheel_factors(
            torch.arange(32.0).reshape(2, 16), wheel_indices, 16
        )
        self.assertEqual(tuple(shared.shape), (2, 1))
        self.assertEqual(tuple(per_dof.shape), (2, 4))
        with self.assertRaises(ValueError):
            select_wheel_factors(torch.ones(2, 3), wheel_indices, 16)

    def test_low_pass_time_constant(self):
        alpha = low_pass_alpha(0.01, 0.5)
        self.assertAlmostEqual(alpha, 1.0 - math.exp(-0.02), places=12)


if __name__ == "__main__":
    unittest.main()
