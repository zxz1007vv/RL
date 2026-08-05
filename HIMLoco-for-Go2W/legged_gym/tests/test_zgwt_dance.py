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
        self.assertFalse(ast.literal_eval(assignments["curriculum"]))
        forbidden = {
            "initial_stage",
            "max_stage",
            "success_curriculum",
            "stage_mode_probabilities",
            "stage_range_scales",
        }
        self.assertTrue(forbidden.isdisjoint(assignments))

    def test_pose_range_curriculum_is_generated_from_endpoints(self):
        commands = self._commands_config_node()
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in commands.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertTrue(assignments["pose_range_curriculum_enabled"])
        self.assertEqual(assignments["pose_range_curriculum_start_level"], 0)
        self.assertEqual(
            assignments["pose_range_start"],
            {
                "body_yaw": 0.03,
                "body_roll": 0.08,
                "body_pitch": 0.08,
            },
        )
        self.assertEqual(
            assignments["pose_range_final"],
            {
                "body_yaw": 0.10,
                "body_roll": 0.26,
                "body_pitch": 0.26,
            },
        )
        self.assertEqual(assignments["pose_range_num_levels"], 5)
        self.assertEqual(assignments["pose_range_min_level_time_s"], 300.0)
        self.assertEqual(assignments["pose_curriculum_min_episodes"], 1024)
        self.assertEqual(
            assignments["pose_curriculum_tracking_threshold"], 0.85
        )
        self.assertEqual(
            assignments["pose_curriculum_max_failure_rate"], 0.05
        )
        self.assertEqual(
            assignments["pose_curriculum_max_anchor_cost"], 0.20
        )
        self.assertEqual(
            assignments["pose_curriculum_max_contact_cost"], 0.05
        )
        self.assertEqual(
            assignments["pose_curriculum_required_success_windows"], 2
        )
        for removed_name in (
            "pose_curriculum_stage_times_s",
            "pose_mode_curriculum_enabled",
            "pose_mode_curriculum_start_stage",
            "pose_curriculum_ranges",
            "pose_curriculum_mode_probabilities",
        ):
            self.assertNotIn(removed_name, assignments)

    def test_pose_range_levels_interpolate_and_advance(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        methods = {
            node.name: node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name in (
                "_pose_range_limits_for_level",
                "_advance_pose_range_curriculum",
            )
        }
        namespace = {}
        exec(
            compile(
                ast.Module(body=list(methods.values()), type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )
        commands = type(
            "Commands",
            (),
            {
                "pose_range_curriculum_enabled": True,
                "pose_range_start": {
                    "body_yaw": 0.03,
                    "body_roll": 0.08,
                    "body_pitch": 0.08,
                },
                "pose_range_final": {
                    "body_yaw": 0.10,
                    "body_roll": 0.26,
                    "body_pitch": 0.26,
                },
                "pose_range_num_levels": 5,
            },
        )
        dummy = type("Dummy", (), {})()
        dummy.cfg = type("Config", (), {"commands": commands})
        dummy.command_ranges = {"body_height": [0.48, 0.48]}
        limits_for_level = namespace["_pose_range_limits_for_level"]
        expected_yaw = [0.0300, 0.0475, 0.0650, 0.0825, 0.1000]
        expected_tilt = [0.080, 0.125, 0.170, 0.215, 0.260]
        for level in range(5):
            ranges = limits_for_level(dummy, level)
            self.assertAlmostEqual(ranges["body_yaw"][1], expected_yaw[level])
            self.assertAlmostEqual(ranges["body_roll"][1], expected_tilt[level])
            self.assertAlmostEqual(ranges["body_pitch"][1], expected_tilt[level])
            self.assertAlmostEqual(ranges["body_yaw"][0], -expected_yaw[level])
            self.assertEqual(ranges["body_height"], [0.48, 0.48])

        dummy.pose_range_curriculum_level = 0
        dummy.pose_range_level_start_step = 0
        dummy.common_step_counter = 123
        advance = namespace["_advance_pose_range_curriculum"]
        self.assertTrue(advance(dummy))
        self.assertEqual(dummy.pose_range_curriculum_level, 1)
        self.assertEqual(dummy.pose_range_level_start_step, 123)
        dummy.pose_range_curriculum_level = 4
        self.assertFalse(advance(dummy))
        commands.pose_range_curriculum_enabled = False
        self.assertFalse(advance(dummy))

    def test_pose_range_upgrade_requires_time_and_stable_windows(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        methods = {
            node.name: node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name in (
                "_advance_pose_range_curriculum",
                "_update_pose_range_curriculum",
            )
        }
        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(body=list(methods.values()), type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )
        commands = type(
            "Commands",
            (),
            {
                "pose_range_curriculum_enabled": True,
                "pose_range_num_levels": 5,
                "pose_range_min_level_time_s": 300.0,
                "pose_curriculum_min_episodes": 2,
                "pose_curriculum_tracking_threshold": 0.85,
                "pose_curriculum_max_failure_rate": 0.05,
                "pose_curriculum_max_anchor_cost": 0.20,
                "pose_curriculum_max_contact_cost": 0.05,
                "pose_curriculum_required_success_windows": 2,
            },
        )
        dummy = type("Dummy", (), {})()
        dummy.cfg = type("Config", (), {"commands": commands})
        dummy.dt = 1.0
        dummy.common_step_counter = 299
        dummy.pose_range_level_start_step = 0
        dummy.pose_range_curriculum_level = 0
        dummy.pose_curriculum_episode_count = 0
        dummy.pose_curriculum_tracking_sum = 0.0
        dummy.pose_curriculum_failure_count = 0
        dummy.pose_curriculum_anchor_cost_sum = 0.0
        dummy.pose_curriculum_contact_cost_sum = 0.0
        dummy.pose_curriculum_success_windows = 0
        dummy.last_pose_curriculum_tracking_quality = 0.0
        dummy.last_pose_curriculum_failure_rate = 1.0
        dummy.last_pose_curriculum_anchor_cost = 0.0
        dummy.last_pose_curriculum_contact_cost = 0.0
        dummy.episode_length_buf = torch.tensor([10, 10])
        dummy.time_out_buf = torch.tensor([True, True])
        dummy.reward_scales = {
            "tracking_body_roll": 1.0,
            "tracking_body_pitch": 1.0,
            "tracking_body_yaw": 1.0,
            "tracking_body_height": 1.0,
            "feet_anchor": -1.0,
            "feet_contact": -1.0,
        }
        dummy.episode_sums = {
            "tracking_body_roll": torch.tensor([9.0, 9.0]),
            "tracking_body_pitch": torch.tensor([9.0, 9.0]),
            "tracking_body_yaw": torch.tensor([9.0, 9.0]),
            "tracking_body_height": torch.tensor([9.0, 9.0]),
            "feet_anchor": torch.tensor([-1.0, -1.0]),
            "feet_contact": torch.tensor([-0.1, -0.1]),
        }
        advance = namespace["_advance_pose_range_curriculum"]
        dummy._advance_pose_range_curriculum = lambda: advance(dummy)
        update = namespace["_update_pose_range_curriculum"]
        env_ids = torch.tensor([0, 1])

        # 指标虽达标，但本级训练时间不足，不能累计成功窗口。
        update(dummy, env_ids)
        self.assertEqual(dummy.pose_curriculum_success_windows, 0)
        self.assertEqual(dummy.pose_range_curriculum_level, 0)

        # 达到最短时间后，必须连续通过两个完整统计窗口才升级。
        dummy.common_step_counter = 300
        update(dummy, env_ids)
        self.assertEqual(dummy.pose_curriculum_success_windows, 1)
        self.assertEqual(dummy.pose_range_curriculum_level, 0)
        dummy.common_step_counter = 301
        update(dummy, env_ids)
        self.assertEqual(dummy.pose_range_curriculum_level, 1)
        self.assertEqual(dummy.pose_curriculum_success_windows, 0)
        self.assertEqual(dummy.pose_range_level_start_step, 301)

    def test_first_stage_uses_near_full_arm_payload(self):
        dance = self._dance_config_node()
        domain_rand = next(
            node
            for node in dance.body
            if isinstance(node, ast.ClassDef) and node.name == "domain_rand"
        )
        payload_range = next(
            ast.literal_eval(node.value)
            for node in domain_rand.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "payload_mass_range"
        )
        self.assertGreater(payload_range[0], 0.0)
        self.assertGreaterEqual(payload_range[1], payload_range[0])

        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in domain_rand.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        com_min = assignments["equivalent_payload_com_min"]
        com_max = assignments["equivalent_payload_com_max"]
        self.assertEqual(len(com_min), 3)
        self.assertEqual(len(com_max), 3)
        self.assertTrue(all(low < high for low, high in zip(com_min, com_max)))
        self.assertNotIn("equivalent_payload_com", assignments)
        self.assertNotIn("com_displacement_delta", assignments)

    def test_first_stage_birth_and_target_height_contract(self):
        dance = self._dance_config_node()
        init_state = next(
            node
            for node in dance.body
            if isinstance(node, ast.ClassDef) and node.name == "init_state"
        )
        birth_position = next(
            ast.literal_eval(node.value)
            for node in init_state.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "pos"
        )

        rewards = next(
            node
            for node in dance.body
            if isinstance(node, ast.ClassDef) and node.name == "rewards"
        )
        default_height = next(
            ast.literal_eval(node.value)
            for node in rewards.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "default_body_height"
        )

        commands = self._commands_config_node()
        ranges = next(
            node
            for node in commands.body
            if isinstance(node, ast.ClassDef) and node.name == "ranges"
        )
        command_height = next(
            ast.literal_eval(node.value)
            for node in ranges.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "body_height"
        )

        self.assertEqual(birth_position[2], 0.55)
        self.assertEqual(default_height, 0.48)
        self.assertEqual(command_height, [0.48, 0.48])

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
        self.assertEqual(scale_values["support_stability"], 0.0)
        self.assertEqual(scale_values["feet_anchor"], -3.0)
        self.assertEqual(scale_values["feet_contact"], -0.5)
        self.assertEqual(scale_values["feet_outward"], -1.0)
        self.assertNotIn("tracking_body_orientation", scale_values)
        self.assertNotIn("neutral_joint_pose", scale_values)
        self.assertNotIn("action_smoothness", scale_values)

        reward_values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in rewards.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(reward_values["feet_anchor_deadzone"], 0.010)
        self.assertEqual(reward_values["feet_anchor_penalty_scale"], 0.020)
        self.assertEqual(reward_values["feet_anchor_mean_weight"], 0.35)
        self.assertEqual(reward_values["feet_anchor_max_weight"], 0.65)
        self.assertEqual(reward_values["support_min_contact_force"], 5.0)
        self.assertEqual(
            reward_values["severe_individual_foot_drift"], 0.10
        )

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

    def test_pd_to_rl_takeover_is_not_implemented(self):
        config_path = (
            Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_config.py"
        )
        robot_path = (
            Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        )
        combined_source = config_path.read_text(encoding="utf-8")
        combined_source += robot_path.read_text(encoding="utf-8")
        for forbidden in (
            "pd_equivalent_actions",
            "takeover_blend",
            "takeover_elapsed",
            "takeover_duration",
            "_sample_takeover_duration",
            "history_reset_pending",
        ):
            self.assertNotIn(forbidden, combined_source)

    def test_feet_outward_penalty_ignores_deadzone_and_inward_motion(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        method = next(
            node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_feet_outward_excess"
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )

        rewards = type("Rewards", (), {"feet_outward_deadzone": 0.015})
        config = type("Config", (), {"rewards": rewards})
        dummy = type("Dummy", (), {})()
        dummy.cfg = config
        dummy.episode_start_yaw = torch.zeros(1)
        dummy.episode_start_feet_xy = torch.zeros(1, 4, 2)
        dummy.feet_outward_sign = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
        dummy.feet_pos = torch.zeros(1, 4, 3)
        dummy.feet_pos[0, :, 1] = torch.tensor([-0.035, 0.025, 0.035, -0.025])

        excess = namespace["_feet_outward_excess"](dummy)
        torch.testing.assert_close(
            excess,
            torch.tensor([[0.020, 0.010, 0.0, 0.0]]),
        )

        # 四只脚共同横移不会改变支撑形状，不应由外八项重复惩罚。
        dummy.feet_pos[:, :, 1] = 0.10
        translated_excess = namespace["_feet_outward_excess"](dummy)
        torch.testing.assert_close(
            translated_excess, torch.zeros_like(translated_excess)
        )

    def test_feet_anchor_uses_each_world_foot_and_smooth_l1_cost(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        methods = {
            node.name: node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
        }
        anchor_source = ast.dump(methods["_reward_feet_anchor"])
        self.assertIn("_feet_anchor_excess", anchor_source)
        self.assertIn("feet_anchor_penalty_scale", anchor_source)
        self.assertIn("feet_anchor_mean_weight", anchor_source)
        self.assertIn("feet_anchor_max_weight", anchor_source)
        self.assertNotIn("_support_center_drift", anchor_source)
        self.assertNotIn("_reward_support_stability", methods)

        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(
                    body=[
                        methods["_feet_anchor_excess"],
                        methods["_reward_feet_anchor"],
                    ],
                    type_ignores=[],
                ),
                str(path),
                "exec",
            ),
            namespace,
        )
        rewards = type(
            "Rewards",
            (),
            {
                "feet_anchor_deadzone": 0.01,
                "feet_anchor_penalty_scale": 0.02,
                "feet_anchor_mean_weight": 0.35,
                "feet_anchor_max_weight": 0.65,
            },
        )
        dummy = type("Dummy", (), {})()
        dummy.cfg = type("Config", (), {"rewards": rewards})
        dummy.episode_start_feet_xy = torch.zeros(1, 4, 2)
        dummy.feet_pos = torch.zeros(1, 4, 3)
        dummy.feet_pos[0, :, 0] = torch.tensor([0.005, 0.02, 0.03, 0.04])

        excess = namespace["_feet_anchor_excess"](dummy)
        torch.testing.assert_close(
            excess,
            torch.tensor([[0.0, 0.01, 0.02, 0.03]]),
        )
        dummy._feet_anchor_excess = lambda: namespace[
            "_feet_anchor_excess"
        ](dummy)
        cost = namespace["_reward_feet_anchor"](dummy)
        torch.testing.assert_close(cost, torch.tensor([0.7921875]))

    def test_feet_contact_penalizes_each_vertical_force_deficit(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        method = next(
            node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_reward_feet_contact"
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )

        rewards = type("Rewards", (), {"support_min_contact_force": 5.0})
        dummy = type("Dummy", (), {})()
        dummy.cfg = type("Config", (), {"rewards": rewards})
        dummy.feet_indices = torch.tensor([0, 1, 2, 3])
        dummy.contact_forces = torch.zeros(1, 4, 3)
        dummy.contact_forces[0, :, 2] = torch.tensor([5.0, 2.5, 0.0, 10.0])

        cost = namespace["_reward_feet_contact"](dummy)
        torch.testing.assert_close(cost, torch.tensor([0.3125]))

    def test_abad_outward_penalty_uses_left_and_right_joint_direction(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        method = next(
            node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_abad_outward_excess"
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )

        rewards = type("Rewards", (), {"abad_outward_deadzone": 0.03})
        config = type("Config", (), {"rewards": rewards})
        dummy = type("Dummy", (), {})()
        dummy.cfg = config
        dummy.feet_abad_indices = torch.tensor([0, 1, 2, 3])
        dummy.feet_outward_sign = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
        dummy.default_dof_pos = torch.zeros(1, 4)
        dummy._current_roll_pitch = lambda: (torch.zeros(1), torch.zeros(1))
        # FAR、FBL 向外，RAR、RBL 向内；左右两侧使用相反关节符号。
        dummy.dof_pos = torch.tensor([[-0.13, 0.08, 0.20, -0.20]])

        excess = namespace["_abad_outward_excess"](dummy)
        torch.testing.assert_close(
            excess,
            torch.tensor([[0.10, 0.05, 0.0, 0.0]]),
        )

    def test_support_center_loss_remains_a_termination_condition(self):
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
        self.assertNotIn(
            "severe_individual_foot_drift", accessed_attributes
        )

    def test_severe_support_penalty_includes_individual_foot_drift(self):
        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        method = next(
            node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_reward_severe_support_loss"
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )

        rewards = type(
            "Rewards",
            (),
            {
                "severe_support_loss_distance": 0.20,
                "severe_individual_foot_drift": 0.10,
            },
        )
        dummy = type("Dummy", (), {})()
        dummy.cfg = type("Config", (), {"rewards": rewards})
        dummy._support_center_drift = lambda: torch.tensor([0.19, 0.21, 0.0])
        dummy.episode_start_feet_xy = torch.zeros(3, 4, 2)
        dummy.feet_pos = torch.zeros(3, 4, 3)
        dummy.feet_pos[0, 2, 0] = 0.11

        penalty = namespace["_reward_severe_support_loss"](dummy)
        torch.testing.assert_close(penalty, torch.tensor([1.0, 1.0, 0.0]))

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

    def test_command_observation_uses_pre_normalization_fixed_scales(self):
        commands = self._commands_config_node()
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in commands.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(
            assignments["command_scales"],
            [2.0, 2.0, 1.0, 1.0, 1.0, 2.0],
        )

        path = Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_dance_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZgwtDance"
        )
        method = next(
            node
            for node in dance.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_current_observation"
        )
        method_source = ast.dump(method)
        self.assertIn("commands_scale", method_source)
        self.assertIn("_body_yaw_error", method_source)
        self.assertNotIn("normalize_dance_commands", method_source)

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
