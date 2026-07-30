# ZGWT 鸡头模式舞蹈：人工分阶段训练说明

本任务不包含自动课程学习。程序运行期间不会自动修改命令概率、命令范围、噪声、
domain randomization、delay、push 或 disturbance。每个阶段都由人工修改同一个
`zgwt_dance_config.py`，然后重新启动训练。

## 固定接口

- actor 单帧 observation：60 维。
- actor 历史 observation：6 帧，共 360 维。
- privileged observation：78 维。
- action：16 维。
- command 顺序：`[vx, vy, relative_yaw_error, roll, pitch, absolute_height]`。
- `vx=0`、`vy=0`，高度只训练 `0.40～0.54 m` 的下蹲动作。
- 四个 wheel action 通道保留，但在控制、reward 和下一帧 observation 前清零。

所有阶段都使用同一套 observation、action 和核心 reward。人工切换阶段时不要修改
这些接口或 command 顺序。

## 人工训练流程

1. 在 `zgwt_dance_config.py` 的 `MANUAL TRAINING STAGE` 区域选择当前阶段的
   `mode_probabilities` 和 `ranges`。
2. 初始无载版本可在阶段 0～3 关闭随机化；面向带载部署时，从 standv2
   开始保留已经验证过的窄范围动力学随机化。
3. 启动训练并人工检查 TensorBoard 与仿真效果。
4. 当前阶段效果稳定后保存 checkpoint。
5. 修改同一个配置文件进入下一阶段。
6. 设置 `resume=True`，加载上一阶段 checkpoint 继续训练。

首次训练命令：

```bash
python legged_gym/scripts/train.py --task zgwt_dance --headless
```

续训时，在 `zgwt_dance_config.py` 的 runner 区域设置：

```python
resume = True
load_run = "上一阶段的运行目录名"
checkpoint = -1
load_actor_only = False
load_optimizer = True
```

## 推荐人工阶段

### 阶段 0：稳定站立

- `neutral` 概率为 1，其余模式为 0。
- yaw、roll、pitch 均为 0。
- height 固定为 `0.54 m`。
- noise、delay、随机化、push、disturbance 全部关闭。

standv1 已完成该阶段；当前配置不再启用原始阶段 0。

### standv2：低随机化稳定站立

- 从 Jul29_16-37-27_standv1/model_3000.pt 继续训练，不从零开始。
- neutral 概率为 1，yaw、roll、pitch 均为 0，高度固定为 0.54 m。
- payload mass：0～6 kg。
- base COM：在 URDF 名义值上进行三轴 ±0.02 m 偏移。
- friction：0.75～1.15，覆盖当前 MuJoCo 地面的 0.8。
- Kp/Kd：名义值的 0.90～1.10。
- motor strength、link mass、noise、delay、push、disturbance 和初始状态扰动保持关闭。

当前配置启用 standv2。训练稳定并通过带载 MuJoCo 检查后，再进入 stage1v2；
stage1v2 应保留上述窄随机化。

### 阶段 1：小范围单轴姿态

- neutral、height、roll、pitch、yaw 和 roll+pitch 混合采样。
- yaw：`[-0.025, 0.025]`。
- roll/pitch：`[-0.08, 0.08]`。
- height：`[0.49, 0.54]`。
- 初始无载训练可关闭 noise 和随机化；从 standv2 继续训练 stage1v2 时，
  保留 standv2 的窄范围动力学随机化，noise 仍关闭。

### 阶段 2：完整姿态，height 不与姿态混合

- 启用独立 height 和完整 roll/pitch/yaw 组合。
- 所有含 height+姿态的模式概率保持为 0。
- yaw：`[-0.10, 0.10]`。
- roll/pitch：`[-0.26, 0.26]`。
- height：`[0.40, 0.54]`。

### 阶段 3：完整 height 混合

- 启用全部 16 种人工 command mode。
- 命令范围保持阶段 2 不变。
- noise 和动力学随机化仍然可以保持关闭，先确认完整任务能够稳定跟踪。

### 阶段 4：Sim2Real 微调

命令概率和范围保持阶段 3 不变，只人工开启较窄随机化：

```python
class noise:
    add_noise = True
    noise_level = 0.20

class domain_rand:
    randomize_friction = True
    friction_range = [0.85, 1.15]
    randomize_motor_strength = True
    motor_strength_range = [0.90, 1.10]
    randomize_kp = True
    kp_range = [0.90, 1.10]
    randomize_kd = True
    kd_range = [0.90, 1.10]
    delay = True
    observation_delay = False
    push_robots = False
    disturbance = False
```

当前暂不加入大载荷、较大 CoM 偏移、link mass 随机化、大推扰或自动正弦轨迹。

## TensorBoard 检查项

- active roll/pitch/yaw/height error。
- inactive roll/pitch/yaw/height drift。
- 机身 XY、支撑中心、平均足端和最大单轮漂移。
- auxiliary multiplier 的 mean 和 p05。
- fall、contact loss、collision、joint limit、torque saturation、action saturation 比例。
- PPO value loss、policy loss、approximate KL、clip fraction、explained variance、
  action std 和 gradient norm。

程序只记录指标，不会依据这些指标自动切换阶段。

## Reward 调整原则

- 所有人工阶段默认保持相同 reward。
- 动作抖动时，后期可以小幅增加 `action_rate` 或 `action_smoothness`。
- `class scales` 是全部 reward 的唯一权重来源。父类统一乘入的 `dt` 会在
  tracking/hold 加权平均的分子和分母中抵消，因此配置中的比例保持不变。
- 足端漂移时，可以小幅提高 `tracking_support_position`、
  `tracking_feet_position` 或 `tracking_max_foot_position` 的 scale。
- 将某个 scale 设为 0 会关闭对应 reward，不需要同步修改另一组权重。
- 每次只修改 1～2 个参数，单次调整尽量不超过 20%～30%。
- 核心 reward 结构或数值尺度发生大改时，只加载 actor/estimator，并重置 critic
  和 optimizer。

## Checkpoint 兼容性

只修改 mode probabilities、command ranges、noise 或窄范围随机化时：

```python
resume = True
load_actor_only = False
load_optimizer = True
```

只小幅修改 auxiliary reward 权重时：

```python
resume = True
load_actor_only = False
load_optimizer = False
```

修改核心 reward、reward 数值尺度或 privileged observation 时：

```python
resume = True
load_actor_only = True
load_optimizer = False
```

修改 actor observation 维度/顺序或 action 维度后，旧 actor checkpoint 不再兼容。

## C++ 部署命令归一化

训练和部署必须使用完全相同的归一化。归一化上限是固定网络语义，不随人工训练阶段
的采样范围改变。

```cpp
inline float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(x, hi));
}

inline float wrapToPi(float angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

std::array<float, 6> normalizeDanceCommand(
    float target_yaw, float current_yaw, float reset_yaw,
    float roll, float pitch, float absolute_height) {
  const float relative_yaw = wrapToPi(current_yaw - reset_yaw);
  const float yaw_error = wrapToPi(target_yaw - relative_yaw);
  return {
      0.0f,
      0.0f,
      clampf(yaw_error / 0.10f, -1.0f, 1.0f),
      clampf(roll / 0.26f, -1.0f, 1.0f),
      clampf(pitch / 0.26f, -1.0f, 1.0f),
      clampf((absolute_height - 0.54f) / (0.54f - 0.40f), -1.0f, 0.0f),
  };
}
```

命令滤波使用：

```cpp
const float alpha = 1.0f - std::exp(-dt / 0.5f);
filtered += alpha * (target - filtered);
```

yaw 使用 `0.30 rad/s` slew rate。策略输出后，在构造下一帧 observation 和计算
执行器控制量前，把四个 wheel action 通道清零。
