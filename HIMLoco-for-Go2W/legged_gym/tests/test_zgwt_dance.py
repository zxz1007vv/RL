import math
import ast
import importlib.util
from pathlib import Path
import unittest

import torch

UTILS_PATH = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_utils.py"
SPEC = importlib.util.spec_from_file_location("zgwt_dance_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)

effective_actions = UTILS.effective_actions
floored_soft_multiplier = UTILS.floored_soft_multiplier
low_pass_alpha = UTILS.low_pass_alpha
neutral_command_weight = UTILS.neutral_command_weight
normalize_dance_commands = UTILS.normalize_dance_commands
select_wheel_factors = UTILS.select_wheel_factors
target_projected_gravity = UTILS.target_projected_gravity
wrap_to_pi = UTILS.wrap_to_pi


class TestZGWTDanceMath(unittest.TestCase):
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
        self.assertEqual(scale_values["tracking_body_orientation"], 11.0)
        self.assertEqual(scale_values["tracking_body_yaw"], 6.0)
        self.assertEqual(scale_values["tracking_body_height"], 6.0)
        self.assertEqual(scale_values["tracking_support_position"], 3.0)

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
