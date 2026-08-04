import ast
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).parents[2]
ORIGINAL = REPOSITORY_ROOT / "resources/robots/zgwt/urdf/zgwt.urdf"
LIGHTWEIGHT = (
    REPOSITORY_ROOT / "resources/robots/zgwt/urdf/zgwt_lightweight.urdf"
)


def element_signature(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(element_signature(child) for child in element),
    )


class TestZGWTLightweightAsset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = ET.parse(ORIGINAL).getroot()
        cls.lightweight = ET.parse(LIGHTWEIGHT).getroot()

    def test_link_inertials_and_collisions_are_exactly_equal(self):
        original_links = {
            link.attrib["name"]: link for link in self.original.findall("link")
        }
        lightweight_links = {
            link.attrib["name"]: link
            for link in self.lightweight.findall("link")
        }
        self.assertEqual(original_links.keys(), lightweight_links.keys())
        for name, original_link in original_links.items():
            lightweight_link = lightweight_links[name]
            self.assertEqual(
                element_signature(original_link.find("inertial")),
                element_signature(lightweight_link.find("inertial")),
                name,
            )
            self.assertEqual(
                [element_signature(item) for item in original_link.findall("collision")],
                [element_signature(item) for item in lightweight_link.findall("collision")],
                name,
            )

    def test_all_sixteen_joints_are_exactly_equal(self):
        original_joints = {
            joint.attrib["name"]: element_signature(joint)
            for joint in self.original.findall("joint")
        }
        lightweight_joints = {
            joint.attrib["name"]: element_signature(joint)
            for joint in self.lightweight.findall("joint")
        }
        self.assertEqual(len(original_joints), 16)
        self.assertEqual(original_joints, lightweight_joints)

    def test_lightweight_visuals_use_no_meshes(self):
        self.assertEqual(len(self.original.findall(".//visual/geometry/mesh")), 17)
        self.assertEqual(len(self.lightweight.findall(".//visual/geometry/mesh")), 0)
        self.assertEqual(len(self.lightweight.findall(".//visual")), 31)
        self.assertEqual(len(self.lightweight.findall(".//collision")), 27)

    def test_only_dance_task_selects_lightweight_asset(self):
        config_path = (
            REPOSITORY_ROOT / "legged_gym/envs/zgwt/zgwt_dance_config.py"
        )
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        dance = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ZGWTDanceCfg"
        )
        asset = next(
            node
            for node in dance.body
            if isinstance(node, ast.ClassDef) and node.name == "asset"
        )
        file_assignment = next(
            node
            for node in asset.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "file"
                for target in node.targets
            )
        )
        self.assertIn(
            "zgwt_lightweight.urdf",
            ast.literal_eval(file_assignment.value),
        )


if __name__ == "__main__":
    unittest.main()
