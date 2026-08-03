from pathlib import Path
import unittest
import warnings

import mujoco
import numpy as np


DOG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_JOINT"
    for leg in ("FAR", "FBL", "RAR", "RBL")
    for joint in ("ABAD", "HIP", "KNEE", "FOOT")
)
ARM_JOINT_NAMES = tuple(f"ROBOT_ARM_JOINT{i}" for i in range(1, 7))


class TestZGWSArmMJCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset_dir = (
            Path(__file__).parents[2]
            / "resources"
            / "robots"
            / "zgwsarm"
        )
        mujoco_warnings = []
        previous_warning_callback = mujoco.get_mju_user_warning()
        try:
            mujoco.set_mju_user_warning(mujoco_warnings.append)
            with warnings.catch_warnings(record=True) as python_warnings:
                warnings.simplefilter("always")
                cls.robot_model = mujoco.MjModel.from_xml_path(
                    str(cls.asset_dir / "zgwsarm.xml")
                )
                cls.scene_model = mujoco.MjModel.from_xml_path(
                    str(cls.asset_dir / "scene_terrain.xml")
                )
        finally:
            mujoco.set_mju_user_warning(previous_warning_callback)

        if python_warnings or mujoco_warnings:
            messages = [str(item.message) for item in python_warnings]
            messages.extend(str(item) for item in mujoco_warnings)
            raise AssertionError("MuJoCo 加载产生 warning: " + " | ".join(messages))

    @staticmethod
    def _body_id(model, name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise AssertionError(f"MJCF 缺少 body: {name}")
        return body_id

    @staticmethod
    def _joint_dof_id(model, name):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise AssertionError(f"MJCF 缺少 joint: {name}")
        return int(model.jnt_dofadr[joint_id])

    @classmethod
    def _subtree_body_ids(cls, model, root_name):
        root_id = cls._body_id(model, root_name)
        descendants = []
        for body_id in range(model.nbody):
            ancestor = body_id
            while ancestor > 0 and ancestor != root_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root_id:
                descendants.append(body_id)
        return descendants

    def test_robot_and_scene_load(self):
        self.assertGreater(self.robot_model.nbody, 1)
        self.assertGreater(self.scene_model.nbody, self.robot_model.nbody)

    def test_robot_subtree_mass_matches_training_contract(self):
        body_ids = self._subtree_body_ids(self.scene_model, "BASE_LINK")
        mass = float(np.sum(self.scene_model.body_mass[body_ids]))
        self.assertAlmostEqual(mass, 47.5524395, delta=1.0e-5)

        expected_masses = {
            "base_connector_link": 0.32036557,
            "ROBOT_ARM_LINK0": 0.84262463,
        }
        for name, expected in expected_masses.items():
            body_id = self._body_id(self.scene_model, name)
            self.assertAlmostEqual(
                float(self.scene_model.body_mass[body_id]),
                expected,
                delta=1.0e-9,
            )

    def test_dog_joint_passive_dynamics_are_zero(self):
        for name in DOG_JOINT_NAMES:
            dof_id = self._joint_dof_id(self.robot_model, name)
            self.assertEqual(float(self.robot_model.dof_damping[dof_id]), 0.0)
            self.assertEqual(
                float(self.robot_model.dof_frictionloss[dof_id]), 0.0
            )
            self.assertEqual(float(self.robot_model.dof_armature[dof_id]), 0.0)

    def test_arm_joint_passive_dynamics_match_training_asset(self):
        for name in ARM_JOINT_NAMES:
            dof_id = self._joint_dof_id(self.robot_model, name)
            self.assertAlmostEqual(
                float(self.robot_model.dof_damping[dof_id]), 0.05
            )
            self.assertAlmostEqual(
                float(self.robot_model.dof_frictionloss[dof_id]), 0.01
            )
            self.assertEqual(float(self.robot_model.dof_armature[dof_id]), 0.0)
