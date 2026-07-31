"""从组合 URDF 生成快速加载的 Isaac Gym 训练资产。

质量、惯量、关节、限位和 collision 完全保留；高精度 STL visual 被替换为
各 link 的第一个基础 collision 几何。这样不会改变物理模型，同时显著减少
多环境训练启动时的网格导入时间。
"""

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "resources/robots/zgwsarm/zgwsarm.urdf"
DEFAULT_OUTPUT = ROOT / "resources/robots/zgwsarm/zgwsarm_train.urdf"


def _indent(element, level=0):
    prefix = "\n" + "  " * level
    child_prefix = "\n" + "  " * (level + 1)
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in children:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_prefix
        children[-1].tail = prefix
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = prefix


def prepare(source, output):
    tree = ET.parse(source)
    root = tree.getroot()
    replaced = 0
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
        collision = link.find("collision")
        if collision is None:
            continue
        visual = ET.Element("visual")
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        if origin is not None:
            visual.append(deepcopy(origin))
        if geometry is not None:
            visual.append(deepcopy(geometry))
        link.insert(
            list(link).index(collision),
            visual,
        )
        replaced += 1

    _indent(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"已生成训练资产: {output}")
    print(f"使用基础碰撞几何替换 visual 的 link 数: {replaced}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.source, args.output)
