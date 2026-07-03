import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF = os.path.join(ROOT, "resources", "robots", "zgwt", "urdf", "zgwt.urdf")
OUT_DIR = os.path.join(ROOT, "resources", "robots", "zgwt", "mjcf")
OUT_XML = os.path.join(OUT_DIR, "zgwt.xml")
OUT_SCENE = os.path.join(OUT_DIR, "scene.xml")
INCLUDE_VISUAL_MESHES = False


def vec(text, default="0 0 0"):
    return text if text is not None else default


def fmt_float(value):
    return f"{float(value):.10g}"


def half_box(size):
    return " ".join(fmt_float(float(v) * 0.5) for v in size.split())


def pretty(element):
    rough = ET.tostring(element, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


tree = ET.parse(URDF)
robot = tree.getroot()

links = {link.attrib["name"]: link for link in robot.findall("link")}
joints = [joint for joint in robot.findall("joint") if joint.attrib.get("type") != "fixed"]
children = {}
for joint in joints:
    parent = joint.find("parent").attrib["link"]
    children.setdefault(parent, []).append(joint)


mujoco = ET.Element("mujoco", {"model": "zgwt"})
ET.SubElement(mujoco, "compiler", {
    "angle": "radian",
    "meshdir": "../meshes",
    "autolimits": "true",
})
ET.SubElement(mujoco, "option", {
    "cone": "elliptic",
    "impratio": "100",
    "timestep": "0.005",
})

default = ET.SubElement(mujoco, "default")
robot_default = ET.SubElement(default, "default", {"class": "zgwt"})
ET.SubElement(robot_default, "geom", {
    "friction": "0.8 0.02 0.01",
    "margin": "0.001",
    "condim": "3",
})
ET.SubElement(robot_default, "joint", {
    "damping": "0.1",
    "armature": "0.01",
    "frictionloss": "0.0",
})
ET.SubElement(robot_default, "motor", {"ctrlrange": "-180 180"})
visual_default = ET.SubElement(robot_default, "default", {"class": "visual"})
ET.SubElement(visual_default, "geom", {
    "type": "mesh",
    "contype": "0",
    "conaffinity": "0",
    "group": "2",
    "mass": "0",
})
collision_default = ET.SubElement(robot_default, "default", {"class": "collision"})
ET.SubElement(collision_default, "geom", {"group": "3"})

asset = ET.SubElement(mujoco, "asset")
ET.SubElement(asset, "material", {"name": "body_gray", "rgba": "0.75 0.75 0.75 1"})
ET.SubElement(asset, "material", {"name": "dark", "rgba": "0.30 0.30 0.30 1"})
if INCLUDE_VISUAL_MESHES:
    mesh_names = []
    for link in robot.findall("link"):
        mesh = link.find("./visual/geometry/mesh")
        if mesh is None:
            continue
        filename = os.path.basename(mesh.attrib["filename"])
        name = os.path.splitext(filename)[0]
        if name not in mesh_names:
            mesh_names.append(name)
            ET.SubElement(asset, "mesh", {"name": name, "file": filename})

worldbody = ET.SubElement(mujoco, "worldbody")


def add_inertial(body, link):
    inertial = link.find("inertial")
    if inertial is None:
        return
    origin = inertial.find("origin")
    mass = inertial.find("mass").attrib["value"]
    inertia = inertial.find("inertia").attrib
    ET.SubElement(body, "inertial", {
        "pos": vec(origin.attrib.get("xyz") if origin is not None else None),
        "mass": mass,
        "fullinertia": " ".join([
            inertia["ixx"],
            inertia["iyy"],
            inertia["izz"],
            inertia["ixy"],
            inertia["ixz"],
            inertia["iyz"],
        ]),
    })


def add_visual(body, link):
    if not INCLUDE_VISUAL_MESHES:
        return
    visual = link.find("visual")
    if visual is None:
        return
    mesh = visual.find("./geometry/mesh")
    if mesh is None:
        return
    origin = visual.find("origin")
    filename = os.path.basename(mesh.attrib["filename"])
    mesh_name = os.path.splitext(filename)[0]
    material = "dark" if "FOOT" in link.attrib["name"] else "body_gray"
    ET.SubElement(body, "geom", {
        "mesh": mesh_name,
        "class": "visual",
        "material": material,
        "pos": vec(origin.attrib.get("xyz") if origin is not None else None),
        "euler": vec(origin.attrib.get("rpy") if origin is not None else None),
    })


def add_collisions(body, link):
    for collision in link.findall("collision"):
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        attrs = {
            "class": "collision",
            "pos": vec(origin.attrib.get("xyz") if origin is not None else None),
            "euler": vec(origin.attrib.get("rpy") if origin is not None else None),
        }
        box = geometry.find("box") if geometry is not None else None
        cylinder = geometry.find("cylinder") if geometry is not None else None
        sphere = geometry.find("sphere") if geometry is not None else None
        if box is not None:
            attrs.update({"type": "box", "size": half_box(box.attrib["size"])})
        elif cylinder is not None:
            attrs.update({
                "type": "cylinder",
                "size": f"{cylinder.attrib['radius']} {fmt_float(float(cylinder.attrib['length']) * 0.5)}",
            })
        elif sphere is not None:
            attrs.update({"type": "sphere", "size": sphere.attrib["radius"]})
        else:
            continue
        ET.SubElement(body, "geom", attrs)


def add_link(parent_body, link_name, incoming_joint=None, is_root=False):
    link = links[link_name]
    body_attrs = {"name": link_name, "childclass": "zgwt"}
    if is_root:
        body_attrs["pos"] = "0 0 0.45"
    elif incoming_joint is not None:
        origin = incoming_joint.find("origin")
        body_attrs["pos"] = vec(origin.attrib.get("xyz") if origin is not None else None)
        body_attrs["euler"] = vec(origin.attrib.get("rpy") if origin is not None else None)
    body = ET.SubElement(parent_body, "body", body_attrs)

    add_inertial(body, link)
    if is_root:
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "site", {"name": "imu", "pos": "0 0 0"})
    if incoming_joint is not None:
        axis = incoming_joint.find("axis")
        limit = incoming_joint.find("limit")
        joint_attrs = {
            "name": incoming_joint.attrib["name"],
            "axis": vec(axis.attrib.get("xyz") if axis is not None else None, "0 0 1"),
        }
        if limit is not None:
            lower = float(limit.attrib.get("lower", "-3.14159"))
            upper = float(limit.attrib.get("upper", "3.14159"))
            if abs(lower) < 1000 and abs(upper) < 1000:
                joint_attrs["range"] = f"{fmt_float(lower)} {fmt_float(upper)}"
        ET.SubElement(body, "joint", joint_attrs)

    add_visual(body, link)
    add_collisions(body, link)

    for child_joint in children.get(link_name, []):
        child = child_joint.find("child").attrib["link"]
        add_link(body, child, child_joint)


add_link(worldbody, "BASE_LINK", is_root=True)

actuator = ET.SubElement(mujoco, "actuator")
for joint in joints:
    name = joint.attrib["name"]
    limit = joint.find("limit")
    effort = limit.attrib.get("effort", "180") if limit is not None else "180"
    ET.SubElement(actuator, "motor", {
        "name": name.replace("_JOINT", ""),
        "joint": name,
        "ctrlrange": f"-{effort} {effort}",
    })

sensor = ET.SubElement(mujoco, "sensor")
for joint in joints:
    name = joint.attrib["name"]
    ET.SubElement(sensor, "jointpos", {"name": f"{name}_pos", "joint": name})
for joint in joints:
    name = joint.attrib["name"]
    ET.SubElement(sensor, "jointvel", {"name": f"{name}_vel", "joint": name})
for joint in joints:
    name = joint.attrib["name"]
    ET.SubElement(sensor, "jointactuatorfrc", {"name": f"{name}_torque", "joint": name})
ET.SubElement(sensor, "framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu", "noise": "0.0"})
ET.SubElement(sensor, "gyro", {"name": "imu_gyro", "site": "imu", "noise": "0.0"})
ET.SubElement(sensor, "accelerometer", {"name": "imu_acc", "site": "imu"})
ET.SubElement(sensor, "framepos", {"name": "frame_pos", "objtype": "site", "objname": "imu"})
ET.SubElement(sensor, "framelinvel", {"name": "frame_vel", "objtype": "site", "objname": "imu"})

scene = ET.Element("mujoco", {"model": "zgwt scene"})
ET.SubElement(scene, "include", {"file": "zgwt.xml"})
ET.SubElement(scene, "statistic", {"center": "0 0 0.2", "extent": "1.2"})
visual = ET.SubElement(scene, "visual")
ET.SubElement(visual, "headlight", {
    "diffuse": "0.6 0.6 0.6",
    "ambient": "0.3 0.3 0.3",
    "specular": "0 0 0",
})
ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})
scene_asset = ET.SubElement(scene, "asset")
ET.SubElement(scene_asset, "texture", {
    "type": "skybox",
    "builtin": "gradient",
    "rgb1": "0.3 0.5 0.7",
    "rgb2": "0 0 0",
    "width": "512",
    "height": "3072",
})
ET.SubElement(scene_asset, "texture", {
    "type": "2d",
    "name": "groundplane",
    "builtin": "checker",
    "mark": "edge",
    "rgb1": "0.2 0.3 0.4",
    "rgb2": "0.1 0.2 0.3",
    "markrgb": "0.8 0.8 0.8",
    "width": "300",
    "height": "300",
})
ET.SubElement(scene_asset, "material", {
    "name": "groundplane",
    "texture": "groundplane",
    "texuniform": "true",
    "texrepeat": "5 5",
    "reflectance": "0.2",
})
scene_world = ET.SubElement(scene, "worldbody")
ET.SubElement(scene_world, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
ET.SubElement(scene_world, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_XML, "w", encoding="utf-8") as f:
    f.write(pretty(mujoco))
with open(OUT_SCENE, "w", encoding="utf-8") as f:
    f.write(pretty(scene))

print(OUT_XML)
print(OUT_SCENE)
