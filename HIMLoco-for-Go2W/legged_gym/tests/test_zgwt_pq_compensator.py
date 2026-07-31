import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import isaacgym  # Isaac Gym requires this import to precede torch.
except Exception:
    isaacgym = None
import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "envs" / "zgwt" / "zgwt_pq_compensator.py"
)
SPEC = importlib.util.spec_from_file_location("zgwt_pq_compensator", MODULE_PATH)
PQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PQ)

FixedPQFeatureNetwork = PQ.FixedPQFeatureNetwork
ZGWTPQCompensator = PQ.ZGWTPQCompensator


def pq_cfg(**overrides):
    values = dict(
        enabled=True,
        shadow_mode=False,
        compensate_wheels=False,
        input_dim=36,
        hidden_dims=[64],
        feature_dim=16,
        activation="tanh",
        feature_seed=20260731,
        append_constant_bias_feature=True,
        joint_position_scale=1.0,
        joint_velocity_scale=0.05,
        joint_reference_scale=1.0,
        filter_time_constant=0.05,
        forgetting_rate=5.0,
        adaptation_gain=1.0e-3,
        weight_leak=1.0e-4,
        m0_diag=[1.0] * 12,
        compensation_torque_ratio=0.05,
        compensation_ramp_time=0.0,
        max_weight_abs=3.0,
        max_weight_update_abs=0.5,
        freeze_during_ramp=False,
        freeze_on_reset=False,
        freeze_on_contact_loss=False,
        freeze_on_torque_saturation=False,
        freeze_on_excessive_tilt=False,
        excessive_tilt_threshold=0.35,
        reset_weights_on_episode_reset=True,
        use_spectral_regularization=False,
        use_projection=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class PQFixture:
    def make_compensator(self, num_envs=3, **overrides):
        comp = ZGWTPQCompensator(
            num_envs=num_envs, cfg=pq_cfg(**overrides), device="cpu"
        )
        q = torch.zeros(num_envs, 12)
        qd = torch.zeros_like(q)
        q_ref = torch.zeros_like(q)
        default = torch.zeros_like(q)
        kp = torch.ones_like(q)
        kd = torch.ones_like(q)
        ids = torch.arange(num_envs)
        comp.reset(ids, q, qd, q_ref, default, kp, kd)
        comp.begin_policy_step(q_ref, 0.01)
        return comp, q, qd, default, kp, kd

    def step_once(self, comp, q, qd, default, kp, kd, **kwargs):
        return comp.step(
            q=q,
            qd=qd,
            default_q=default,
            kp=kp,
            kd=kd,
            torque_limits=torch.full((12,), 100.0),
            low_level_dt=0.002,
            **kwargs,
        )


class TestFixedPQFeatures(unittest.TestCase):
    def test_feature_network_shape_and_frozen_parameters(self):
        network = FixedPQFeatureNetwork(36, [64], 16, "tanh", 7)
        output = network(torch.zeros(5, 36))
        self.assertEqual(tuple(output.shape), (5, 16))
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))

    def test_fixed_seed_reproduces_identical_weights(self):
        first = FixedPQFeatureNetwork(36, [64], 16, "tanh", 19)
        second = FixedPQFeatureNetwork(36, [64], 16, "tanh", 19)
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values()
        ):
            torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)

    def test_feature_construction_does_not_advance_global_rng(self):
        torch.manual_seed(123)
        expected = torch.rand(4)
        torch.manual_seed(123)
        FixedPQFeatureNetwork(36, [64], 16, "tanh", 19)
        actual = torch.rand(4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestPQTensorDynamics(unittest.TestCase, PQFixture):
    def test_state_shapes_and_environment_independence(self):
        comp, *_ = self.make_compensator(4)
        self.assertEqual(tuple(comp.W_hat.shape), (4, 17, 12))
        self.assertEqual(tuple(comp.P.shape), (4, 17, 17))
        self.assertEqual(tuple(comp.Q.shape), (4, 17, 12))
        comp.W_hat[0, 0, 0] = 2.0
        self.assertEqual(comp.W_hat[1, 0, 0].item(), 0.0)
        # There is only one fixed feature module shared across the batch.
        self.assertEqual(
            len(list(comp.feature_network.parameters())),
            len(list(comp.feature_network.network.parameters())),
        )

    def test_pq_and_composite_update_shapes(self):
        comp, q, qd, default, kp, kd = self.make_compensator(3)
        qd[:] = 0.1
        output = self.step_once(comp, q, qd, default, kp, kd)
        self.assertEqual(tuple(output.shape), (3, 12))
        self.assertEqual(tuple(comp.last_composite_error.shape), (3, 17, 12))
        self.assertTrue(torch.isfinite(comp.P).all())
        self.assertTrue(torch.isfinite(comp.Q).all())

    def test_compensation_is_leg_only_and_wheels_stay_zero(self):
        comp, q, qd, default, kp, kd = self.make_compensator(2)
        comp.W_hat[:, -1, :] = 1.0
        leg_torque = self.step_once(comp, q, qd, default, kp, kd)
        full = torch.zeros(2, 16)
        leg_indices = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
        wheel_indices = torch.tensor([3, 7, 11, 15])
        full[:, leg_indices] = leg_torque
        self.assertEqual(tuple(leg_torque.shape), (2, 12))
        torch.testing.assert_close(full[:, wheel_indices], torch.zeros(2, 4))

    def test_torque_limit_is_strict(self):
        comp, q, qd, default, kp, kd = self.make_compensator(
            2, compensation_torque_ratio=0.05
        )
        comp.W_hat[:, -1, :] = 3.0
        torque = self.step_once(comp, q, qd, default, kp, kd)
        self.assertTrue(torch.all(torch.abs(torque) <= 5.0))

    def test_shadow_and_disabled_output_zero(self):
        comp, q, qd, default, kp, kd = self.make_compensator(
            2, shadow_mode=True
        )
        comp.W_hat[:, -1, :] = 1.0
        torch.testing.assert_close(
            self.step_once(comp, q, qd, default, kp, kd),
            torch.zeros(2, 12),
        )
        comp.enabled = False
        before = comp.W_hat.clone()
        torch.testing.assert_close(
            self.step_once(comp, q, qd, default, kp, kd),
            torch.zeros(2, 12),
        )
        torch.testing.assert_close(comp.W_hat, before)

    def test_reset_only_changes_selected_environment(self):
        comp, q, qd, default, kp, kd = self.make_compensator(3)
        comp.W_hat[:] = 1.0
        comp.P[:] = 2.0
        comp.Q[:] = 3.0
        comp.reset(torch.tensor([1]), q, qd, q, default, kp, kd)
        self.assertEqual(torch.count_nonzero(comp.W_hat[1]).item(), 0)
        self.assertGreater(torch.count_nonzero(comp.W_hat[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(comp.W_hat[2]).item(), 0)

    def test_one_environment_does_not_update_another(self):
        comp, q, qd, default, kp, kd = self.make_compensator(2)
        qd[0] = 1.0
        qd[1] = 0.0
        self.step_once(comp, q, qd, default, kp, kd)
        self.assertGreater(torch.linalg.vector_norm(comp.W_hat[0]).item(), 0.0)
        self.assertEqual(torch.linalg.vector_norm(comp.W_hat[1]).item(), 0.0)

    def test_nan_input_safely_disables_affected_output(self):
        comp, q, qd, default, kp, kd = self.make_compensator(2)
        comp.W_hat[:, -1, :] = 1.0
        q[0, 0] = float("nan")
        output = self.step_once(comp, q, qd, default, kp, kd)
        torch.testing.assert_close(output[0], torch.zeros(12))
        self.assertTrue(torch.isfinite(output[1]).all())
        self.assertEqual(comp.nan_count[0].item(), 1.0)

    def test_nan_internal_state_is_cleared(self):
        comp, q, qd, default, kp, kd = self.make_compensator(2)
        comp.W_hat[0, 0, 0] = float("nan")
        output = self.step_once(comp, q, qd, default, kp, kd)
        torch.testing.assert_close(output[0], torch.zeros(12))
        self.assertTrue(torch.isfinite(comp.W_hat).all())
        self.assertTrue(torch.isfinite(comp.u_f).all())

    def test_constant_positive_error_produces_opposing_torque(self):
        comp, q, qd, default, kp, kd = self.make_compensator(
            1, adaptation_gain=0.1, weight_leak=0.0
        )
        for parameter in comp.feature_network.parameters():
            parameter.zero_()
        qd[:] = 0.5
        kp.zero_()
        kd.zero_()
        torque = self.step_once(comp, q, qd, default, kp, kd)
        self.assertTrue(torch.all(torque < 0.0))

    def test_adaptation_gain_scales_update(self):
        low, q, qd, default, kp, kd = self.make_compensator(
            1, adaptation_gain=0.001, weight_leak=0.0
        )
        high, *_ = self.make_compensator(
            1, adaptation_gain=0.01, weight_leak=0.0
        )
        qd[:] = 0.2
        self.step_once(low, q, qd, default, kp, kd)
        self.step_once(high, q, qd, default, kp, kd)
        low_norm = torch.linalg.vector_norm(low.W_hat)
        high_norm = torch.linalg.vector_norm(high.W_hat)
        self.assertGreater(high_norm.item(), low_norm.item() * 9.9)

    def test_weight_leak_decays_without_error(self):
        comp, q, qd, default, kp, kd = self.make_compensator(
            1, adaptation_gain=1.0e-6, weight_leak=0.5
        )
        comp.W_hat[:] = 1.0
        before = torch.linalg.vector_norm(comp.W_hat).item()
        self.step_once(comp, q, qd, default, kp, kd)
        after = torch.linalg.vector_norm(comp.W_hat).item()
        self.assertLess(after, before)

    def test_ramp_is_monotonic(self):
        comp, q, qd, default, kp, kd = self.make_compensator(
            1,
            compensation_ramp_time=0.006,
            adaptation_gain=1.0e-9,
            weight_leak=0.0,
        )
        comp.W_hat[:, -1, :] = 1.0
        blends = []
        for _ in range(5):
            self.step_once(comp, q, qd, default, kp, kd)
            blends.append(comp.last_blend.item())
        self.assertEqual(blends[0], 0.0)
        self.assertTrue(all(a <= b for a, b in zip(blends, blends[1:])))
        self.assertAlmostEqual(blends[-1], 1.0, places=6)

    def test_p_remains_symmetric(self):
        comp, q, qd, default, kp, kd = self.make_compensator(2)
        qd[:] = torch.randn_like(qd)
        for _ in range(5):
            self.step_once(comp, q, qd, default, kp, kd)
        torch.testing.assert_close(comp.P, comp.P.transpose(-1, -2))

    def test_feature_export_contains_deployment_contract(self):
        comp, *_ = self.make_compensator(1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "pq_feature_weights.pt")
            PQ.export_pq_feature_network(comp, path)
            exported = torch.load(path)
        self.assertEqual(exported["input_dim"], 36)
        self.assertEqual(exported["actual_feature_dim"], 17)
        self.assertEqual(exported["activation"], "tanh")
        self.assertEqual(tuple(exported["m0_diag"].shape), (12,))
        self.assertIn("network.0.weight", exported["state_dict"])


class TestPQValidation(unittest.TestCase):
    def test_invalid_filter_time_constant(self):
        with self.assertRaises(ValueError):
            ZGWTPQCompensator(1, pq_cfg(filter_time_constant=0.0), "cpu")

    def test_invalid_adaptation_forgetting_and_m0(self):
        for overrides in (
            {"adaptation_gain": 0.0},
            {"forgetting_rate": 0.0},
            {"m0_diag": [1.0] * 11},
            {"m0_diag": [1.0] * 11 + [0.0]},
        ):
            with self.assertRaises(ValueError):
                ZGWTPQCompensator(1, pq_cfg(**overrides), "cpu")


class TestDancePQTorqueRegression(unittest.TestCase):
    """Numerical controller regression without constructing an Isaac simulation."""

    @classmethod
    def setUpClass(cls):
        try:
            from legged_gym.envs.zgwt.zgwt_dance_robot import ZgwtDance
            from legged_gym.envs.zgwt.zgwt_dance_pq_robot import ZgwtDancePQ
        except Exception as exc:  # Isaac Gym is optional for pure tensor CI.
            raise unittest.SkipTest(f"Isaac Gym controller import unavailable: {exc}")
        cls.Dance = ZgwtDance
        cls.DancePQ = ZgwtDancePQ

    def make_robot(self, enabled, shadow, compensation=0.0):
        robot = object.__new__(self.DancePQ)
        robot.num_envs = 2
        robot.num_dof = 16
        robot.num_actions = 16
        robot.device = "cpu"
        robot.sim_params = SimpleNamespace(dt=0.002)
        robot.wheel_indices = torch.tensor([3, 7, 11, 15])
        robot.leg_indices = torch.tensor(
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        )
        robot.default_dof_pos = torch.zeros(1, 16)
        robot.dof_pos = torch.linspace(-0.1, 0.1, 32).reshape(2, 16)
        robot.dof_vel = torch.linspace(0.2, -0.2, 32).reshape(2, 16)
        robot.p_gains = torch.tensor(
            [90.0, 120.0, 120.0, 60.0] * 4
        )
        robot.d_gains = torch.tensor([1.0, 1.0, 1.0, 0.2] * 4)
        robot.Kp_factors = torch.ones(2, 1)
        robot.Kd_factors = torch.ones(2, 1)
        robot.motor_strength_factors = torch.ones(2, 1)
        robot.torque_limits = torch.full((16,), 180.0)
        robot.wheel_park_targets = torch.zeros(2, 4)
        robot.feet_indices = torch.tensor([3, 7, 11, 15])
        robot.contact_forces = torch.zeros(2, 16, 3)
        robot.contact_forces[:, robot.feet_indices, 2] = 50.0
        robot.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
        robot.cfg = SimpleNamespace(
            control=SimpleNamespace(
                control_type="P",
                action_scale=0.25,
                vel_scale=10.0,
                wheel_park_stiffness=25.0,
                wheel_park_damping=2.5,
            ),
            pq=SimpleNamespace(enabled=enabled, shadow_mode=shadow),
        )

        class FakeCompensator:
            def __init__(self):
                self.enabled = enabled
                self.shadow_mode = shadow

            def step(self, **_kwargs):
                return torch.full((2, 12), float(compensation))

        robot.pq_compensator = FakeCompensator()
        return robot

    def test_disabled_torque_matches_original_to_float_precision(self):
        robot = self.make_robot(enabled=False, shadow=False)
        actions = torch.linspace(-0.3, 0.3, 32).reshape(2, 16)
        original = self.Dance._compute_torques(robot, actions)
        disabled = self.DancePQ._compute_torques(robot, actions)
        torch.testing.assert_close(disabled, original, rtol=0, atol=0)

    def test_shadow_matches_original_and_active_preserves_wheel_park(self):
        actions = torch.zeros(2, 16)
        shadow_robot = self.make_robot(
            enabled=True, shadow=True, compensation=1.0
        )
        original = self.Dance._compute_torques(shadow_robot, actions)
        shadow = self.DancePQ._compute_torques(shadow_robot, actions)
        torch.testing.assert_close(shadow, original, rtol=0, atol=0)

        active_robot = self.make_robot(
            enabled=True, shadow=False, compensation=1.0
        )
        active_original = self.Dance._compute_torques(active_robot, actions)
        active = self.DancePQ._compute_torques(active_robot, actions)
        torch.testing.assert_close(
            active[:, active_robot.wheel_indices],
            active_original[:, active_robot.wheel_indices],
            rtol=0,
            atol=0,
        )
        self.assertFalse(
            torch.equal(
                active[:, active_robot.leg_indices],
                active_original[:, active_robot.leg_indices],
            )
        )


if __name__ == "__main__":
    unittest.main()
