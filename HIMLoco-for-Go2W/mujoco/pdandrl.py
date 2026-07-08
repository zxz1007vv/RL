import yaml
import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
from pynput import keyboard
import argparse
import os

# ------------------ work ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()
config_path = os.path.abspath(args.config)
config_dir = os.path.dirname(config_path)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = cfg["paths"]
for key, path in list(paths.items()):
    if isinstance(path, str) and not os.path.isabs(path):
        paths[key] = os.path.normpath(os.path.join(config_dir, path))
joints_cfg = cfg.get("joints")
if joints_cfg:
    joint_names = [joint["name"] for joint in joints_cfg]
    wheel_ids = [i for i, joint in enumerate(joints_cfg) if joint.get("wheel", False)]
else:
    joint_names = cfg["joint_names"]
    wheel_ids = cfg["wheel_ids"]
base_body_name = cfg.get("base_body", "base_link")
base_pos = cfg.get("base_pos")
startup_pose = cfg.get("startup_pose", "crouch")

def expand_dof_values(values):
    return values * 4 if len(values) == 4 else values

def get_dof_values(key, legacy_key):
    if joints_cfg:
        return [joint[key] for joint in joints_cfg]
    return expand_dof_values(cfg[legacy_key])

default_dof_pos = torch.tensor(get_dof_values("default", "default_dof_pos"), dtype=torch.float32, device=device)
crouch_dof_pos  = torch.tensor(get_dof_values("crouch", "crouch_dof_pos"), dtype=torch.float32, device=device)

# control
p_gains = torch.tensor(get_dof_values("kp", "p_gains"), dtype=torch.float32, device=device)
d_gains = torch.tensor(get_dof_values("kd", "d_gains"), dtype=torch.float32, device=device)
actions_scale = cfg["actions_scale"]
vel_scale = cfg["vel_scale"]
wheel_control = cfg.get("wheel_control", "velocity")
commands_cfg = cfg.get("commands", {})
command_mode = commands_cfg.get("mode", "heading")
command_ranges = commands_cfg.get("ranges", {})
yaw_kp = commands_cfg.get("yaw_kp", cfg.get("yaw_kp", 2.5))

scale_factors = cfg["scale_factors"]
input_cfg = cfg.get("input", {})

if input_cfg.get("source", "keyboard") == "gamepad":
    from cmd_gamepad import CmdGenerator

    cmd_generator = CmdGenerator(input_cfg.get("gamepad", {}))
    get_cmd = cmd_generator.get_cmd
    pop_mode_request = cmd_generator.pop_mode_request
else:
    from cmd_keyboard import get_cmd

    def pop_mode_request():
        return None

# load scene
m = mujoco.MjModel.from_xml_path(paths["scene_xml"])
if "timestep" in cfg:
    m.opt.timestep = float(cfg["timestep"])
d = mujoco.MjData(m)
ctrl_limits = torch.tensor(m.actuator_ctrlrange, dtype=torch.float32, device=device)

# def function
def get_sensor_data(name):
    id_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if id_ == -1:
        raise ValueError(f"Sensor {name} not found")
    adr, dim = m.sensor_adr[id_], m.sensor_dim[id_]
    return torch.tensor(d.sensordata[adr:adr+dim], device=device, dtype=torch.float32)

def world2self(quat, v):
    q_w, q_vec = quat[0], quat[1:]
    v_vec = torch.tensor(v, device=device, dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    return a - b + c

def get_obs(actions, default_dof_pos, commands):
    sf = scale_factors
    commands_scale = torch.tensor([sf["scale_lin_vel"], sf["scale_lin_vel"], sf["scale_ang_vel"]], device=device)
    base_quat = get_sensor_data("imu_quat")
    projected_gravity = world2self(base_quat, torch.tensor([0., 0., -1.], device=device))
    imu_gyro = get_sensor_data("imu_gyro")

# get dos pos & vel from sensor
# dof pos of wheel -> zero
    dof_pos = torch.zeros(16, device=device)
    for i, n in enumerate(joint_names):
        dof_pos[i] = get_sensor_data(n + "_pos")[0]
    dof_pos[wheel_ids] = 0.0

    dof_vel = torch.zeros(16, device=device)
    for i, n in enumerate(joint_names):
        dof_vel[i] = get_sensor_data(n + "_vel")[0]

    cmds = torch.tensor(commands, device=device)
    return torch.cat([
        imu_gyro * sf["scale_ang_vel"],
        projected_gravity,
        cmds * commands_scale,
        (dof_pos - default_dof_pos) * sf["scale_dof_pos"],
        dof_vel * sf["scale_dof_vel"],
        actions
    ], dim=-1)

def clamp_command(value, name):
    limits = command_ranges.get(name)
    if limits is None:
        return value
    return float(np.clip(value, limits[0], limits[1]))

def prepare_commands(raw_commands):
    commands = [
        clamp_command(raw_commands[0], "lin_vel_x"),
        clamp_command(raw_commands[1], "lin_vel_y"),
        raw_commands[2],
    ]
    if command_mode == "heading":
        base_quat = get_sensor_data("imu_quat")
        q_w, q_x, q_y, q_z = base_quat
        yaw_now = torch.atan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y*q_y + q_z*q_z))
        yaw_target = torch.tensor(commands[2], device=device, dtype=torch.float32)
        yaw_err = torch.atan2(torch.sin(yaw_target - yaw_now), torch.cos(yaw_target - yaw_now))
        commands[2] = clamp_command((yaw_kp * yaw_err).item(), "ang_vel_yaw")
        return commands, yaw_now.item()

    commands[2] = clamp_command(commands[2], "ang_vel_yaw")
    return commands, None

def fill_obs_buffer(actions, commands):
    obs_now = torch.clip(get_obs(actions, default_dof_pos, commands), -100, 100)
    return obs_now.repeat(6, 1)

def apply_ctrl(act):
    act = torch.max(torch.min(act, ctrl_limits[:, 1]), ctrl_limits[:, 0])
    d.ctrl[:] = act.detach().cpu().numpy()

def main():
    global control_mode
    control_mode = 0 # 0 -> Zero torque  1 -> PD stand  2 -> RL

    # load policy
    try:
        policy = torch.jit.load(paths["policy_path"])
        policy.eval().to(device)
        print("Success to load policy network")
    except Exception as e:
        policy = None
        print("Fail to load policy network", e)

    startup_dof_pos = default_dof_pos if startup_pose == "default" else crouch_dof_pos
    for i, name in enumerate(joint_names):
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[jnt_id]] = startup_dof_pos[i].item()
    if base_pos is not None:
        d.qpos[:3] = np.asarray(base_pos, dtype=np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    print("Zero torque mode......")
    print("Gamepad: A -> RL   B -> zero torque")
    print("Keyboard fallback: 1 -> PD stand   2 -> RL")

    actions = torch.zeros(16, device=device)
    obs_buffer = torch.zeros((6, 57), device=device)

    # control by keyboard
    def on_press(key):
        nonlocal actions, obs_buffer
        global control_mode
        try:
            if key.char == '1' and control_mode == 0:
                control_mode = 1
                print(" PD mode ......")
            elif key.char == '2' and policy is not None:
                actions = torch.zeros(16, device=device)
                obs_buffer = fill_obs_buffer(actions, [0., 0., 0.])
                control_mode = 2
                print(" RL mode ......")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # running
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mode_request = pop_mode_request()
            if mode_request == "rl":
                if policy is not None:
                    actions = torch.zeros(16, device=device)
                    obs_buffer = fill_obs_buffer(actions, [0., 0., 0.])
                    control_mode = 2
                    print("\n RL mode ......")
                else:
                    print("\nCannot enter RL mode: policy is not loaded")
            elif mode_request == "zero":
                actions = torch.zeros(16, device=device)
                obs_buffer = torch.zeros((6, 57), device=device)
                control_mode = 0
                print("\n Zero torque mode ......")

            dof_pos = torch.cat([get_sensor_data(n+"_pos") for n in joint_names]).to(device)
            dof_vel = torch.cat([get_sensor_data(n+"_vel") for n in joint_names]).to(device)
            dof_err = default_dof_pos - dof_pos
            dof_err[wheel_ids] = 0.0

            # rl control
            if control_mode == 2:
                commands, yaw_now = prepare_commands(get_cmd())
                status = f"\rRL cmd: vx={commands[0]:+4.1f}  vy={commands[1]:+4.1f}  wz={commands[2]:+4.1f}"
                if yaw_now is not None:
                    status += f"  yaw_now={yaw_now:+4.2f}"
                print(status, end='')
            else:
                commands = [0., 0., 0.]
            # pd control
            if control_mode == 1:
                act = torch.zeros(16, device=device)
                for i in range(16):
                    if i in wheel_ids:
                        act[i] = -d_gains[i]*dof_vel[i]
                    else:
                        act[i] = (1.2 * 1.25 * p_gains[i]*dof_err[i] - d_gains[i]*dof_vel[i])
                apply_ctrl(act)

            elif control_mode == 2 and policy is not None:

                obs_now = get_obs(actions, default_dof_pos, commands)
                obs_now = torch.clip(obs_now, -100, 100)
                obs_buffer = torch.cat([obs_now.unsqueeze(0), obs_buffer[:-1]], dim=0)
                obs_seq = obs_buffer.flatten()
                actions = policy(obs_seq)
                actions_scaled = actions * actions_scale
                if wheel_control == "torque":
                    pos_actions_scaled = actions_scaled.clone()
                    pos_actions_scaled[wheel_ids] = 0.0
                    act = p_gains * (pos_actions_scaled + dof_err) - d_gains * dof_vel
                    act[wheel_ids] = (
                        actions_scaled[wheel_ids] * p_gains[wheel_ids]
                        - d_gains[wheel_ids] * dof_vel[wheel_ids]
                    )
                else:
                    actions_scaled[wheel_ids] = 0.0
                    vel_ref = torch.zeros_like(actions_scaled)
                    vel_ref[wheel_ids] = actions[wheel_ids] * vel_scale
                    act = p_gains * (actions_scaled + dof_err) + d_gains * (vel_ref - dof_vel)
                apply_ctrl(act)
            else:
                d.ctrl[:] = 0.0

            step_start = time.time()
            for _ in range(cfg["sim_steps_per_loop"]):
                mujoco.mj_step(m, d)
            # camera on robot
            viewer.cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)]
            viewer.sync()
            time_until_next_step = m.opt.timestep * cfg["sim_steps_per_loop"] - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main() 
