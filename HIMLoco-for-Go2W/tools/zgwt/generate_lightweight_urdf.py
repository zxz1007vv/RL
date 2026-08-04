#!/usr/bin/env python3
"""为纯狗训练生成物理等价、无 STL visual 的轻量 URDF。"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPOSITORY_ROOT / "resources/robots/zgwt/urdf/zgwt.urdf"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "resources/robots/zgwt/urdf/zgwt_lightweight.urdf"
)


def _indent(element: ET.Element, level: int = 0) -> None:
    """兼容 Python 3.8 的确定性 XML 缩进。"""

    indentation = "\n" + level * "  "
    child_indentation = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indentation
        for child in element:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_indentation
        element[-1].tail = indentation
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def _copy_material(link: ET.Element) -> ET.Element:
    for visual in link.findall("visual"):
        material = visual.find("material")
        if material is not None:
            return copy.deepcopy(material)
    material = ET.Element("material", {"name": "lightweight_gray"})
    ET.SubElement(material, "color", {"rgba": "0.7 0.7 0.7 1"})
    return material


def _collision_visual(
    link_name: str,
    collision: ET.Element,
    material: ET.Element,
    index: int,
) -> ET.Element:
    visual = ET.Element(
        "visual", {"name": f"{link_name}_primitive_visual_{index}"}
    )
    origin = collision.find("origin")
    if origin is not None:
        visual.append(copy.deepcopy(origin))
    geometry = collision.find("geometry")
    if geometry is None:
        raise ValueError(f"{link_name} 的 collision {index} 缺少 geometry")
    visual.append(copy.deepcopy(geometry))
    visual.append(copy.deepcopy(material))
    return visual


def _fallback_visual(link: ET.Element, material: ET.Element) -> ET.Element:
    """无 collision 的 ABAD link 只增加可视 box，不增加物理碰撞。"""

    link_name = link.attrib["name"]
    visual = ET.Element("visual", {"name": f"{link_name}_fallback_visual"})
    inertial_origin = link.find("inertial/origin")
    if inertial_origin is not None:
        visual.append(copy.deepcopy(inertial_origin))
    else:
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": "0.08 0.08 0.08"})
    visual.append(copy.deepcopy(material))
    return visual


def generate(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    for link in root.findall("link"):
        material = _copy_material(link)
        for visual in list(link.findall("visual")):
            link.remove(visual)

        collisions = list(link.findall("collision"))
        visuals = [
            _collision_visual(link.attrib["name"], collision, material, index)
            for index, collision in enumerate(collisions)
        ]
        if not visuals:
            visuals = [_fallback_visual(link, material)]

        first_collision_index = next(
            (
                index
                for index, child in enumerate(list(link))
                if child.tag == "collision"
            ),
            len(link),
        )
        for offset, visual in enumerate(visuals):
            link.insert(first_collision_index + offset, visual)

    _indent(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve())
    print(f"已生成轻量 URDF：{args.output.resolve()}")


if __name__ == "__main__":
    main()
