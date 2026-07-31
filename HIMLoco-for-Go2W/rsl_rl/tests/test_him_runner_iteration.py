import os
import tempfile
import unittest

import torch

from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner


class _DummyOptimizer:
    def state_dict(self):
        return {"dummy": True}

    def load_state_dict(self, state):
        self.loaded_state = state


class _DummyEstimator:
    def __init__(self):
        self.optimizer = _DummyOptimizer()


class _DummyActorCritic:
    def __init__(self):
        self.estimator = _DummyEstimator()

    def state_dict(self):
        return {"weight": torch.tensor([1.0])}

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state
        return None


class _DummyAlgorithm:
    def __init__(self):
        self.actor_critic = _DummyActorCritic()
        self.optimizer = _DummyOptimizer()


def _make_runner(reset_iteration):
    runner = object.__new__(HIMOnPolicyRunner)
    runner.cfg = {
        "load_actor_only": False,
        "load_optimizer": False,
        "reset_iteration_on_load": reset_iteration,
    }
    runner.alg = _DummyAlgorithm()
    runner.current_learning_iteration = 0
    runner.parent_checkpoint_path = None
    runner.parent_checkpoint_iteration = None
    return runner


class TestHimRunnerIterationLoading(unittest.TestCase):
    def _checkpoint(self, directory, iteration=4000):
        path = os.path.join(directory, "model_4000.pt")
        torch.save(
            {
                "model_state_dict": {"weight": torch.tensor([1.0])},
                "optimizer_state_dict": {},
                "estimator_optimizer_state_dict": {},
                "iter": iteration,
                "infos": None,
            },
            path,
        )
        return path

    def test_new_stage_resets_local_iteration_and_keeps_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(directory)
            runner = _make_runner(reset_iteration=True)

            runner.load(checkpoint)

            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertEqual(runner.parent_checkpoint_iteration, 4000)
            self.assertEqual(runner.parent_checkpoint_path, os.path.abspath(checkpoint))

            saved_path = os.path.join(directory, "model_0.pt")
            runner.save(saved_path, iteration=0)
            saved = torch.load(saved_path)
            self.assertEqual(saved["iter"], 0)
            self.assertEqual(saved["parent_checkpoint_iteration"], 4000)
            self.assertEqual(
                saved["parent_checkpoint_path"], os.path.abspath(checkpoint)
            )
            self.assertTrue(saved["iteration_reset_on_load"])

    def test_same_stage_resume_preserves_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(directory)
            runner = _make_runner(reset_iteration=False)

            runner.load(checkpoint)

            self.assertEqual(runner.current_learning_iteration, 4000)
            self.assertEqual(runner.parent_checkpoint_iteration, 4000)


if __name__ == "__main__":
    unittest.main()
