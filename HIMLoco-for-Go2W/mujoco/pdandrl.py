import yaml
import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
import argparse
import os

# ------------------ work ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()
config_path = os.path.abspath(args.config)
config_dir = os.path.dirname(config_path)

def merge_config(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    with open(path, "r") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
    return merge_config(load_config(os.path.normpath(parent_path)), loaded)


cfg = load_config(config_path)
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
pd_transition_time = max(0.0, float(cfg.get("pd_transition_time", 0.5)))

def expand_dof_values(values):
    return values * 4 if len(values) == 4 else values

def get_dof_values(key, legacy_key):
    if joints_cfg:
        return [joint[key] for joint in joints_cfg]
    return expand_dof_values(cfg[legacy_key])

default_dof_pos = torch.tensor(get_dof_values("default", "default_dof_pos"), dtype=torch.float32, device=device)
crouch_dof_pos  = torch.tensor(get_dof_values("crouch", "crouch_dof_pos"), dtype=torch.float32, device=device)

startup_cfg = cfg.get("startup_sequence", {})
startup_enabled = bool(startup_cfg.get("enabled", False))
startup_leg_order = startup_cfg.get("leg_order", ["FAR", "FBL", "RAR", "RBL"])

def build_startup_pose(abad_values, hip_values, knee_values):
    if not (
        len(abad_values) == len(hip_values) == len(knee_values) == len(startup_leg_order)
    ):
        raise ValueError("Startup pose arrays must match startup_sequence.leg_order")
    pose = default_dof_pos.clone()
    leg_to_index = {name: index for index, name in enumerate(startup_leg_order)}
    for dof_index, joint_name in enumerate(joint_names):
        leg_name = joint_name.split("_", 1)[0]
        if leg_name not in leg_to_index:
            continue
        leg_index = leg_to_index[leg_name]
        if "ABAD" in joint_name:
            pose[dof_index] = abad_values[leg_index]
        elif "HIP" in joint_name:
            pose[dof_index] = hip_values[leg_index]
        elif "KNEE" in joint_name:
            pose[dof_index] = knee_values[leg_index]
        elif dof_index in wheel_ids:
            pose[dof_index] = 0.0
    return pose


if startup_enabled:
    fold_dof_pos = build_startup_pose(
        startup_cfg["fold_abad_pos"],
        startup_cfg["fold_hip_pos"],
        startup_cfg["fold_knee_pos"],
    )
    stand_dof_pos = build_startup_pose(
        startup_cfg["stand_abad_pos"],
        startup_cfg["stand_hip_pos"],
        startup_cfg["stand_knee_pos"],
    )
    fold_duration = max(float(startup_cfg.get("fold_duration", 2.0)), 1.0e-6)
    stand_duration = max(float(startup_cfg.get("stand_duration", 2.0)), 1.0e-6)
else:
    fold_dof_pos = None
    stand_dof_pos = None
    fold_duration = 0.0
    stand_duration = 0.0

if "pd_hold_dof_pos" in cfg:
    pd_hold_dof_pos = torch.tensor(
        expand_dof_values(cfg["pd_hold_dof_pos"]), dtype=torch.float32, device=device
    )
elif joints_cfg and all("pd_hold" in joint for joint in joints_cfg):
    pd_hold_dof_pos = torch.tensor(
        [joint["pd_hold"] for joint in joints_cfg], dtype=torch.float32, device=device
    )
else:
    raise ValueError(
        "Define an explicit PD hold pose with pd_hold_dof_pos, or pd_hold for every joint."
    )

# control
p_gains = torch.tensor(get_dof_values("kp", "p_gains"), dtype=torch.float32, device=device)
d_gains = torch.tensor(get_dof_values("kd", "d_gains"), dtype=torch.float32, device=device)
pd_p_gains = torch.tensor(
    expand_dof_values(cfg.get("pd_hold_p_gains", p_gains.detach().cpu().tolist())),
    dtype=torch.float32,
    device=device,
)
pd_d_gains = torch.tensor(
    expand_dof_values(cfg.get("pd_hold_d_gains", d_gains.detach().cpu().tolist())),
    dtype=torch.float32,
    device=device,
)
actions_scale = cfg["actions_scale"]
vel_scale = cfg["vel_scale"]
wheel_control = cfg.get("wheel_control", "velocity")
commands_cfg = cfg.get("commands", {})
command_mode = commands_cfg.get("mode", "heading")
command_ranges = commands_cfg.get("ranges", {})
command_deadband = float(commands_cfg.get("deadband", 0.05))
policy_command_names = commands_cfg.get(
    "policy_order", ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]
)
pose_defaults = commands_cfg.get("pose_defaults", {})
dance_trajectory = commands_cfg.get("dance_trajectory", {})
yaw_kp = commands_cfg.get("yaw_kp", cfg.get("yaw_kp", 2.5))

scale_factors = cfg["scale_factors"]
input_cfg = cfg.get("input", {})
input_source = input_cfg.get("source", "keyboard")
history_length = int(cfg.get("history_length", 6))

if input_source == "gamepad":
    from cmd_gamepad import CmdGenerator

    cmd_generator = CmdGenerator(input_cfg.get("gamepad", {}))
    get_cmd = cmd_generator.get_cmd
    pop_mode_request = cmd_generator.pop_mode_request
elif input_source == "keyboard":
    from cmd_keyboard import get_cmd

    def pop_mode_request():
        return None
elif input_source == "none":
    def get_cmd():
        return [0.0, 0.0, 0.0]

    def pop_mode_request():
        return None
else:
    raise ValueError(f"Unsupported input source: {input_source}")

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
    default_command_scales = [
        sf["scale_lin_vel"], sf["scale_lin_vel"], sf["scale_ang_vel"]
    ]
    commands_scale = torch.tensor(
        commands_cfg.get("scales", default_command_scales), device=device
    )
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
    yaw_now = None
    if command_mode == "heading":
        base_quat = get_sensor_data("imu_quat")
        q_w, q_x, q_y, q_z = base_quat
        yaw_now = torch.atan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y*q_y + q_z*q_z))
        yaw_target = torch.tensor(commands[2], device=device, dtype=torch.float32)
        yaw_err = torch.atan2(torch.sin(yaw_target - yaw_now), torch.cos(yaw_target - yaw_now))
        commands[2] = clamp_command((yaw_kp * yaw_err).item(), "ang_vel_yaw")
    else:
        commands[2] = clamp_command(commands[2], "ang_vel_yaw")

    if all(abs(command) < command_deadband for command in commands[:3]):
        commands[:3] = [0.0, 0.0, 0.0]

    # Dance policies append [body_roll, body_pitch, body_height]. Input devices
    # may provide these explicitly; otherwise use the safe pose defaults.
    for index, name in enumerate(policy_command_names[3:], start=3):
        value = raw_commands[index] if len(raw_commands) > index else pose_defaults[name]
        commands.append(clamp_command(value, name))

    return commands, None if yaw_now is None else yaw_now.item()

def fill_obs_buffer(actions, commands):
    obs_now = torch.clip(get_obs(actions, default_dof_pos, commands), -100, 100)
    return obs_now.repeat(history_length, 1)

def apply_ctrl(act):
    act = torch.max(torch.min(act, ctrl_limits[:, 1]), ctrl_limits[:, 0])
    d.ctrl[:] = act.detach().cpu().numpy()

def main():
    global control_mode
    if startup_enabled:
        control_mode = 0  # zero torque
    else:
        control_mode = 2 if cfg.get("start_in_rl", False) else 1
    pd_phase = "zero" if startup_enabled else "hold"
    pd_target_dof_pos = pd_hold_dof_pos.clone()
    pd_transition_start = None
    pd_transition_elapsed = 0.0 if startup_enabled else pd_transition_time
    trajectory_start_time = time.time()

    # load policy
    try:
        policy = torch.jit.load(paths["policy_path"])
        policy.eval().to(device)
        print("Success to load policy network")
    except Exception as e:
        policy = None
        print("Fail to load policy network", e)

    if not startup_enabled:
        for i, name in enumerate(joint_names):
            jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[jnt_id]] = pd_hold_dof_pos[i].item()
    if base_pos is not None:
        d.qpos[:3] = np.asarray(base_pos, dtype=np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    if startup_enabled:
        print("Zero-torque mode ......")
        print("Gamepad: A -> fold then stand   B -> RL (after standing)")
        print("Keyboard fallback: A -> fold then stand   B -> RL")
    else:
        print("PD hold mode ......")
        print("Gamepad: A -> RL   B -> PD hold")
        print("Keyboard fallback: 1 -> PD hold   2 -> RL")

    actions = torch.zeros(16, device=device)
    one_step_obs_size = 6 + len(policy_command_names) + 3 * len(joint_names)
    obs_buffer = torch.zeros((history_length, one_step_obs_size), device=device)
    if control_mode == 2 and policy is not None:
        obs_buffer = fill_obs_buffer(actions, prepare_commands(get_cmd())[0])
        print("RL mode ......")

    # control by keyboard
    keyboard_mode_request = None

    def on_press(key):
        nonlocal actions, obs_buffer, pd_target_dof_pos, pd_transition_start
        nonlocal pd_transition_elapsed, keyboard_mode_request
        global control_mode
        try:
            pressed = key.char.lower()
            if startup_enabled and pressed == 'a':
                keyboard_mode_request = "stand"
            elif startup_enabled and pressed == 'b':
                keyboard_mode_request = "rl"
            elif pressed == '1':
                pd_transition_start = None
                pd_transition_elapsed = pd_transition_time
                pd_target_dof_pos = pd_hold_dof_pos.clone()
                control_mode = 1
                print(" PD hold mode ......")
            elif pressed == '2' and policy is not None:
                actions = torch.zeros(16, device=device)
                obs_buffer = fill_obs_buffer(
                    actions, prepare_commands(get_cmd())[0]
                )
                control_mode = 2
                print(" RL mode ......")
        except AttributeError:
            pass

    # Gamepad control does not require pynput. Import it only for the optional
    # numeric-key fallback so a missing X11 display does not prevent startup.
    listener = None
    if input_source != "none":
        try:
            from pynput import keyboard as pynput_keyboard

            listener = pynput_keyboard.Listener(on_press=on_press)
            listener.start()
        except ImportError as exc:
            print(f"Keyboard fallback disabled: {exc}")

    # running
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            dof_pos = torch.cat([get_sensor_data(n+"_pos") for n in joint_names]).to(device)
            dof_vel = torch.cat([get_sensor_data(n+"_vel") for n in joint_names]).to(device)

            mode_request = pop_mode_request()
            if mode_request is None and keyboard_mode_request is not None:
                mode_request = keyboard_mode_request
                keyboard_mode_request = None

            if startup_enabled and mode_request == "stand":
                actions = torch.zeros(16, device=device)
                pd_transition_start = dof_pos.clone()
                pd_transition_elapsed = 0.0
                pd_phase = "fold"
                control_mode = 1
                print("\nPD sequence: moving to folded pose ......")
            elif startup_enabled and mode_request == "rl":
                if pd_phase != "hold":
                    print("\nRL blocked: press A and wait until standing is complete")
                elif policy is not None:
                    actions = torch.zeros(16, device=device)
                    obs_buffer = fill_obs_buffer(
                        actions, prepare_commands(get_cmd())[0]
                    )
                    control_mode = 2
                    print("\nRL mode ......")
                else:
                    print("\nCannot enter RL mode: policy is not loaded")
            elif not startup_enabled and mode_request == "rl":
                if policy is not None:
                    actions = torch.zeros(16, device=device)
                    obs_buffer = fill_obs_buffer(
                        actions, prepare_commands(get_cmd())[0]
                    )
                    control_mode = 2
                    print("\n RL mode ......")
                else:
                    print("\nCannot enter RL mode: policy is not loaded")
            elif not startup_enabled and mode_request == "pd":
                actions = torch.zeros(16, device=device)
                obs_buffer = torch.zeros(
                    (history_length, one_step_obs_size), device=device
                )
                # Blend from the RL exit pose to the fixed PD target instead of
                # applying a large position error in one simulation step.
                pd_transition_start = dof_pos.clone()
                pd_transition_elapsed = 0.0
                control_mode = 1
                print("\n PD hold mode ......")

            rl_dof_err = default_dof_pos - dof_pos
            rl_dof_err[wheel_ids] = 0.0

            # rl control
            if control_mode == 2:
                raw_commands = get_cmd()
                if dance_trajectory.get("enabled", False) and len(policy_command_names) >= 6:
                    elapsed = time.time() - trajectory_start_time
                    frequency = float(dance_trajectory.get("frequency", 0.25))
                    phase = 2.0 * np.pi * frequency * elapsed
                    raw_commands = list(raw_commands[:3]) + [
                        float(dance_trajectory.get("roll_amplitude", 0.15)) * np.sin(phase),
                        float(dance_trajectory.get("pitch_amplitude", 0.12)) * np.sin(phase + np.pi / 2.0),
                        float(dance_trajectory.get("height_center", 0.52))
                        + float(dance_trajectory.get("height_amplitude", 0.06)) * np.sin(phase * 0.5),
                    ]
                commands, yaw_now = prepare_commands(raw_commands)
                status = f"\rRL cmd: vx={commands[0]:+4.1f}  vy={commands[1]:+4.1f}  wz={commands[2]:+4.1f}"
                if len(commands) >= 6:
                    status += (
                        f"  roll={commands[3]:+4.2f} pitch={commands[4]:+4.2f}"
                        f" height={commands[5]:4.2f}"
                    )
                if yaw_now is not None:
                    status += f"  yaw_now={yaw_now:+4.2f}"
                print(status, end='')
            else:
                commands = [0., 0., 0.]
            # pd control
            if control_mode == 1:
                control_dt = m.opt.timestep * cfg["sim_steps_per_loop"]
                if startup_enabled and pd_phase in ("fold", "stand"):
                    target_pose = fold_dof_pos if pd_phase == "fold" else stand_dof_pos
                    duration = fold_duration if pd_phase == "fold" else stand_duration
                    blend = min(pd_transition_elapsed / duration, 1.0)
                    blend = blend * blend * (3.0 - 2.0 * blend)
                    pd_target_dof_pos = torch.lerp(
                        pd_transition_start, target_pose, blend
                    )
                    pd_transition_elapsed += control_dt
                    if pd_transition_elapsed >= duration:
                        if pd_phase == "fold":
                            pd_phase = "stand"
                            pd_transition_start = dof_pos.clone()
                            pd_transition_elapsed = 0.0
                            print("\nPD sequence: moving to standing pose ......")
                        else:
                            pd_phase = "hold"
                            pd_target_dof_pos = stand_dof_pos.clone()
                            print("\nPD standing hold ready; press B to enter RL")
                elif startup_enabled and pd_phase == "hold":
                    pd_target_dof_pos = stand_dof_pos
                elif pd_transition_start is not None and pd_transition_elapsed < pd_transition_time:
                    blend = pd_transition_elapsed / pd_transition_time
                    pd_target_dof_pos = torch.lerp(pd_transition_start, pd_hold_dof_pos, blend)
                    pd_transition_elapsed += control_dt
                else:
                    pd_target_dof_pos = pd_hold_dof_pos
                pd_dof_err = pd_target_dof_pos - dof_pos
                pd_dof_err[wheel_ids] = 0.0
                act = torch.zeros(16, device=device)
                for i in range(16):
                    if i in wheel_ids:
                        act[i] = -pd_d_gains[i]*dof_vel[i]
                    else:
                        act[i] = pd_p_gains[i] * pd_dof_err[i] - pd_d_gains[i] * dof_vel[i]
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
                    act = p_gains * (pos_actions_scaled + rl_dof_err) - d_gains * dof_vel
                    if abs(commands[0]) < command_deadband and abs(commands[1]) < command_deadband and abs(commands[2]) < command_deadband:
                        act[wheel_ids] = -d_gains[wheel_ids] * dof_vel[wheel_ids]
                    else:
                        act[wheel_ids] = (
                            actions_scaled[wheel_ids] * p_gains[wheel_ids]
                            - d_gains[wheel_ids] * dof_vel[wheel_ids]
                        )
                else:
                    actions_scaled[wheel_ids] = 0.0
                    vel_ref = torch.zeros_like(actions_scaled)
                    vel_ref[wheel_ids] = actions[wheel_ids] * vel_scale
                    if abs(commands[0]) < command_deadband and abs(commands[1]) < command_deadband and abs(commands[2]) < command_deadband:
                        vel_ref[wheel_ids] = 0.0
                    act = p_gains * (actions_scaled + rl_dof_err) + d_gains * (vel_ref - dof_vel)
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
