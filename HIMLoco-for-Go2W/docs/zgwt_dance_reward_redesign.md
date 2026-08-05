# ZGWT 姿态跟踪 reward 与带臂训练重设计

本文对应 `ZGWTDanceCfg`、`ZgwtDance` 和继承它们的完整机械臂任务。当前代码已经按本文的 Stage 0A 配置落地；observation 保持 360 维、privileged observation 保持 78 维、action 保持 16 维，command 顺序仍为 `[vx, vy, relative_yaw_error, roll, pitch, height]`。

## 1. 当前问题诊断

### 1.1 Reward 重复和缺失

- 原 `tracking_body_orientation` 在 `_tracking_cost()` 中把 projected gravity 的 x/y 误差求和，roll 与 pitch 无法分开诊断。再与 yaw、height 做加权平均时，一个轴的严重错误会被其他三个较好轴稀释。
- `tracking_lin_vx`、`tracking_lin_vy` 现在独立启用并跟踪固定零命令，便于直接提高平面防漂移权重和分轴诊断；`body_stability` 不再重复计算 xy 线速度，只负责 z 线速度与角速度。
- 原支撑中心约束无法发现四脚对称外八或单轮滑动。当前把四个轮足的 episode 初始世界坐标作为 soft anchor；机身仍可围绕固定支撑完成小范围姿态和高度跟踪。
- `feet_anchor` 和 `feet_contact` 现在是两个独立的连续负惩罚，分别负责世界坐标防滑和逐轮有效接触，便于单独调权与诊断。
- 原 `neutral_joint_pose = sum(abs(q-q_default))` 无法区分四腿共模下蹲和单腿差模折叠，并会直接阻碍主动 height 命令，Stage 0 删除。
- 原 `action_rate` 与 `action_smoothness` 分别是一阶、二阶动作差分；当前带载站立阶段同时使用会重复限制策略的快速承载反应。Stage 0 只保留 `action_rate`。
- 原 `torques` 会鼓励策略省力。带载站立阶段的主要失败正是支撑不足，因此 Stage 0 删除；稳定后再以很小权重加入。
- 原实现缺少独立 roll/pitch、明显向下塌陷的 soft pressure、四腿差模协调性、hard low-height、hard tilt 和显式 termination penalty。

### 1.2 原乘法结构的问题

原 `ZgwtDance.compute_reward()` 使用：

```python
task_reward * auxiliary_multiplier + hard_safety_penalty + termination_reward
```

其具体问题是：

- `tracking_cost` 是加权平均，严重单轴误差可被其他轴稀释。
- `hold_gate = exp(-tracking_cost / sigma)`，tracking 越差，站姿和支撑保持越弱；这与跌倒前最需要稳定约束的时机相反。
- 协调性如果放进 soft multiplier，会在 multiplier 触底后失去梯度区分能力。
- 多个负 reward 先相加、再经过指数和 floor，无法从日志直接反推某个权重对总 reward 的贡献。

乘法结构适合“只有任务接近成功才评价动作质量”的最终精修，但不适合当前 Stage 0 的承载生存问题。当前实现直接沿用 `LeggedRobot` 的 reward 调度：`scales.xxx` 自动调用 `_reward_xxx()`，由父类直接加权求和，各项始终有效：

```python
reward = task_tracking + stance_stability + control_quality + safety_constraints
```

源码不再维护 reward 名称集合，也不重写 `_prepare_reward_function()` 或 `compute_reward()`。四类只用于排列 `_reward_xxx()` 函数和中文分区注释；训练日志由父类按每个具体 reward 名称记录。

### 1.3 Domain randomization 诊断

原 `ZGWTDanceCfg.domain_rand` 同时存在：

- `payload_mass_range = [0, 10]`；
- 三轴相同 `com_displacement_range = [-0.08, 0.08]`；
- payload 与 COM 独立无条件采样；
- `randomize_kp = True`、`randomize_kd = True`；
- `disturbance = True`、`push_robots = True`。

这会产生“很轻 payload 配极端 COM”“很重 payload 配相反方向 COM”等不符合固定机械臂的样本，还把质量、偏载、控制增益、外扰同时混合，策略容易用异常折腿覆盖极端角落。原注释称“不叠加 motor、noise、push、delay 或初始状态扰动”，但实际 push、disturbance、Kp、Kd 已开启，注释和布尔开关不一致。

根据组合 URDF 计算，连接件、机械臂和工具总质量为 `6.5071195 kg`；home 姿态下这部分质心在 `BASE_LINK` 坐标系约为 `[-0.1742, 0.0012, 0.2422] m`。当前 `_process_rigid_body_props()` 用质量加权公式把 payload 与 base COM 偏移绑定，不再独立采样一个三轴大立方体。

## 2. 最终四类 reward 表

| 类别 | Reward | 处理 | 说明 |
|---|---|---:|---|
| Task Tracking | `tracking_body_orientation` | 拆分/删除 | 替换为 roll、pitch 两项，旧函数仅留兼容代码 |
| Task Tracking | `tracking_body_roll` | 新增 | 当前权重 6 |
| Task Tracking | `tracking_body_pitch` | 新增 | 当前权重 6；这是人工调高后的明确配置 |
| Task Tracking | `tracking_body_yaw` | 保留 | reset 朝向仍是 relative yaw 参考 |
| Task Tracking | `tracking_body_height` | 修改 | 目标使用当前 command，内部加入 height deficit |
| Stance Stability | `tracking_lin_vx`、`tracking_lin_vy` | 保留并加强 | 权重各 4，命令固定为 0 |
| Stance Stability | `yaw_rate` | 合并 | 并入 `body_stability` |
| Stance Stability | `feet_vertical_motion` | 合并/删除 | 机身速度与接触稳定已覆盖，Stage 0 不单列 |
| Stance Stability | `stance_coordination` | 新增 | 区分共模压缩与差模折叠，排除 wheel joint |
| Stance Stability | `tracking_support_position` | diagnostics/终止 | 支撑中心只用于整体漂移诊断和严重支撑丢失终止 |
| Stance Stability | `feet_anchor` | 新增 | 四轮平均和最大单轮世界坐标漂移的 Smooth-L1 负惩罚 |
| Stance Stability | `feet_contact` | 新增 | 每个轮足垂直接触力不足的独立负惩罚 |
| Stance Stability | `support_stability` | 关闭 | scale 设为 0，旧函数删除 |
| Stance Stability | `feet_outward` | 弱保留 | scale 降至 -1；anchor 稳定后可设为 0，仅保留 diagnostics |
| Stance Stability | `neutral_joint_pose` | 删除 | 会惩罚协调下蹲，不能识别单腿折叠 |
| Safety Constraints | `feet_stumble` | diagnostics/删除 | 平地 Stage 0 不需要；障碍任务再启用 |
| Safety Constraints | `severe_support_loss` | 保留并终止 | 支撑中心漂移超过 0.20 m 时给 hard penalty 并结束 episode |
| Control Quality | `action_rate` | 保留 | Stage 0 唯一平滑项 |
| Control Quality | `action_smoothness` | 删除 | 与 action rate 部分重复，稳定后消融加入 |
| Control Quality | `torques` | 删除/后期加入 | Stage 0 不鼓励省力 |
| Control Quality | `torque_rate`、`dof_vel`、`dof_acc` | 后期可选 | 一次只加入一个并做消融 |
| Safety Constraints | `collision` | 保留 | 非足端碰撞 |
| Safety Constraints | `dof_pos_limits` | 保留 | 关节位置软限位 |
| Safety Constraints | `torque_limits` | 保留且减小 | 只惩罚 99% 饱和，权重 -0.05 |
| Safety Constraints | `severe_wheel_park` | 保留 | 驻车轮角严重偏离 |
| Safety Constraints | `base_height_too_low` | 新增 | `h < 0.36 m` hard penalty |
| Safety Constraints | `excessive_tilt` | 新增 | 倾角大于 0.45 rad hard penalty |
| Safety Constraints | `termination` | 新增 | 非 timeout 失败，权重 -10 |

## 3. 最小可行 reward 方案

Stage 0 当前只启用：

```text
Task Tracking:
  tracking_body_roll       +6.0
  tracking_body_pitch      +6.0
  tracking_body_yaw        +6.0
  tracking_body_height     +6.0

Stance Stability:
  tracking_lin_vx          +4.0
  tracking_lin_vy          +4.0
  body_stability           +3.0
  stance_coordination      +4.0
  support_stability         0.0
  feet_anchor              -2.0
  feet_contact             -0.5
  feet_outward             -1.0

Control Quality:
  action_rate              -0.01

Safety Constraints:
  collision                -1.0
  dof_pos_limits           -2.0
  torque_limits            -0.05
  severe_wheel_park        -4.0
  severe_support_loss      -4.0
  base_height_too_low      -4.0
  excessive_tilt           -4.0
  termination             -10.0
```

所有 scale 仍由父类统一乘 `dt`。不启用 `only_positive_rewards`，否则 safety penalty 会被裁掉。

## 4. 数学定义

### 4.1 Roll、pitch、yaw

令当前欧拉角为 `r, p`，命令为 `r*, p*`：

```text
R_roll  = exp(-(r-r*)² / 0.012)
R_pitch = exp(-(p-p*)² / 0.012)
```

二者使用相同 sigma；`sqrt(0.012) ≈ 0.11 rad` 是初始容差尺度。相对 yaw 仍为：

```text
yaw_rel = wrap_to_pi(yaw_current - yaw_at_episode_reset)
e_yaw   = wrap_to_pi(yaw_command - yaw_rel)
R_yaw   = exp(-e_yaw² / 0.0025)
```

### 4.2 Height 与单侧 deficit

```text
e_h = h_current - h_command
d_h = relu(h_command - h_current - 0.020)
R_h = exp(-e_h²/0.0016 - d_h²/0.0004)
```

`height_deadzone` 推荐初始范围为 `0.015～0.025 m`，当前取 `0.020 m`。对称项保持正常上下误差的连续梯度，单侧项只加强超过死区的明显塌陷。因为 `d_h` 始终相对当前过滤后的 `h_command` 计算，命令主动从 0.54 降到 0.45 m 时不会把 0.45 m 当成塌陷；只有低于 0.43 m 才触发额外项。

### 4.3 平面零速跟踪与 Body stability

```text
R_vx = exp(-(vx_command-vx)²/0.010), vx_command=0
R_vy = exp(-(vy_command-vy)²/0.010), vy_command=0

C_body = v_z²/0.010
       + ||omega_xy||²/0.250
       + omega_z²/0.090
R_body = exp(-C_body)
```

vx/vy 独立记录，避免平面漂移被 z 速度或角速度平均掩盖；`body_stability` 保留其余机身稳定量，避免对 xy 线速度重复计分。

### 4.4 Stance coordination

ZGWT 每腿使用 ABAD、HIP、KNEE、FOOT 四个关节。协调伸展量只使用 KNEE；FOOT/wheel 必须排除。由于前腿默认 knee 为 `-1.2 rad`、后腿为 `+1.2 rad`，直接比较 knee 角需要前后符号对齐；当前用 `cos(q_knee)` 计算腿长，天然消除了这个符号差异：

```text
L_i = sqrt(l1² + l2² + 2*l1*l2*cos(q_knee_i))
l1 = 0.26 m, l2 = 0.28 m
```

左右腿不需要额外符号对齐。为避免阻碍 roll/pitch，先计算命令姿态下髋点本来需要的高度差：

```text
delta_i = -sin(pitch*) * x_i
        + cos(pitch*) * sin(roll*) * y_i
delta_i <- delta_i - mean(delta)

residual_i = L_i - mean(L) - delta_i
deadzone = 0.015
         + 0.010 * (|roll*|/0.26 + |pitch*|/0.26)
C_coord = mean(relu(|residual_i|-deadzone)²)
R_coord = exp(-C_coord / 0.0004)
```

`mean(L)` 是允许的共模压缩；只有差模残差受罚。固定死区推荐 `0.012～0.025 m`。当前的命令补偿比简单 `exp(-command²)` gating 更合适，因为它不会在大姿态命令时完全关闭协调约束；额外连续放宽只用于吸收简化几何和合理偏载不对称。

### 4.5 Feet anchor 与 feet contact

```text
d_i = ||foot_xy_i-foot_xy_i_reset||
e_i = relu(d_i - 0.010)
n_i = e_i / 0.020
Huber(n_i) = 0.5*n_i², n_i < 1
           = n_i-0.5, n_i >= 1
C_anchor = 0.35*mean(Huber(n_i)) + 0.65*max(Huber(n_i))

c_i = clip((5N-Fz_i)/5N, 0, 1)
C_contact = mean(c_i²)

R_anchor  = -2.0*C_anchor
R_contact = -0.5*C_contact
```

轮足 anchor 使用世界坐标而非机身相对坐标，因此不会把正常的机身 yaw、roll、pitch 和 height 变化当作足端误差。平均项约束四足整体滑动，权重更大的最大项捕捉单轮异常；接触项只要求每条腿达到最小有效接触力，不要求接触力相等。支撑中心继续用于 diagnostics、严重支撑丢失 hard penalty 和原有 termination；单轮超过 0.10 m 目前只触发 hard penalty，不触发 termination。

### 4.6 Action rate

```text
C_action = sum_j (a_t,j - a_t-1,j)²
R_action = -0.01 * C_action
```

wheel action 在进入控制、reward 和 history 前已经清零，因此不会污染该项。

## 5. 代码修改

### 5.1 `zgwt_dance_config.py`

已修改 `ZGWTDanceCfg.commands` 为 Stage 0A neutral command，修改 `domain_rand` 为固定 `6.5071 kg` 等效负载并关闭 push、disturbance、Kp/Kd、friction 随机化；`rewards.scales` 已替换成最小四类集合。核心参数集中在同一个 `rewards` 类中，observation/action 配置没有变化。

Stage 0B 只需修改：

```python
payload_mass_range = [5.8, 7.2]
com_displacement_delta = [0.008, 0.005, 0.008]
randomize_friction = True
friction_range = [0.75, 0.90]
```

质量变化时 `_process_rigid_body_props()` 会自动重新计算关联 COM；`com_displacement_delta` 只是各轴小残差，不是另一个 ±8 cm 立方体。

### 5.2 `zgwt_dance_robot.py`

已修改：

- 四类 reward 集合和直接加权和 `compute_reward()`；
- `_reward_tracking_body_roll()`、`_reward_tracking_body_pitch()`；
- `_reward_tracking_body_height()` 内部 deficit；
- `_reward_body_stability()`；
- `_leg_extension()`、`_expected_extension_differential()`、`_reward_stance_coordination()`；
- `_reward_feet_anchor()`、`_reward_feet_contact()`；
- hard low-height、hard tilt、termination；
- payload/COM 物理关联；
- height deficit、腿长 spread、协调残差 p95、body/support score 等 episode diagnostics。

relative yaw 的 reset 参考、command 顺序、wheel action mask 均保持不变。

### 5.3 兼容修改

`zgwt_dance_utils.py` 增加纯 tensor 数学函数，便于脱离 Isaac Gym 做单元测试。完整机械臂配置同步到相同四类 reward，并明确 `correlate_payload_and_com = False`，因为完整机械臂的质量、COM 和惯量已由组合 URDF 显式表达。

这是核心 reward 和回报尺度变化。旧 checkpoint 只能加载 actor/estimator；critic、optimizer 和 iteration 必须重置。actor observation 和 action 形状未变，因此 actor 权重兼容。

## 6. 分阶段训练配置

### Stage 0A：固定等效负载站立

- Command：100% neutral，roll/pitch/yaw=0，height=0.54 m。
- Payload/COM：6.5071 kg；home 等效质心固定，无残差。
- Push/disturbance：关闭。
- Kp/Kd、motor、initial state：关闭随机化。
- Noise：关闭。
- Checkpoint：从旧 reward 只加载 actor/estimator，重置 critic、optimizer、iteration；也可从头训练作为基线。
- 进入指标：至少 99% episode 非跌倒；稳态 mean height deficit < 1 cm；height deficit ratio < 5%；p95 coordination residual < 3 cm；joint limit hit 接近 0；在固定负载 sim2sim 连续站立 30 s 无单腿折叠。

### Stage 0B：窄范围负载随机化

- Command：仍为 neutral。
- Payload：建议先 `[5.8, 7.2] kg`，即 nominal 周围约 ±10%；不要立即回到 `[0,10]`。
- COM：由 payload 自动关联；各轴额外残差建议 `[±8, ±5, ±8] mm`。范围应根据机械臂实际姿态库统计再校准。
- Friction：可加入窄范围 `[0.75,0.90]`；其余随机化先关闭。
- Push/disturbance、Kp/Kd：关闭。
- Noise：先关闭；末期可加低幅 observation noise。
- Checkpoint：reward 不变，可加载 actor、critic、estimator；建议重置 optimizer 以便清晰归因，若范围非常窄也可做保留 optimizer 的对照。
- 进入指标：整个质量/COM 分位点上 fall rate < 2%；p95 coordination residual 不比 0A 恶化超过约 20%；最差 5% 环境无持续单侧折腿；sim2sim 等效负载通过。

### Stage 0C：完整机械臂模型微调

- Command：先 neutral；机械臂先固定 home，再逐步加入安全静态姿态库。
- Payload/COM：完整 URDF 名义动力学，额外 payload 只做小幅模型误差随机化，不再套用纯狗的等效 COM 关联。
- Push/disturbance：先关闭。
- Kp/Kd：先固定；站稳后只加约 ±5% Kp、±10% Kd。
- Noise：先关闭，最终加入与部署一致的低幅噪声。
- Checkpoint：从 0B 只加载 actor/estimator，重置 critic、optimizer、iteration。
- 进入指标：home 和代表性姿态库均通过 30 s；arm control reaction 下不出现单腿折叠。
- 预算：可把约 15%～25% 用于 0A、45%～60% 用于 0B、15%～30% 用于 0C；这是吞吐和失败模式驱动的起点，不是固定规律。

纯狗＋等效质量/COM 能覆盖总重量、静态重力偏心力矩和一部分粗略基座惯量变化；不能准确覆盖各连杆姿态相关惯量、机械臂运动引起的动量交换和关节控制器反作用力。因此机械臂若会运动或主动输出力矩，最终必须用完整模型微调。

### Stage 1：小范围单轴姿态

- Command 概率建议：neutral 0.35、height 0.10、roll 0.20、pitch 0.20、yaw 0.15；不启用组合姿态。
- 范围：roll/pitch ±0.08 rad、yaw ±0.025 rad、height 0.49～0.54 m。
- Payload/COM：沿用通过的 0B 窄分布。
- Push/disturbance、Kp/Kd：关闭；noise 仍关闭。
- Checkpoint：reward 不变，可加载 actor/critic/estimator；建议重置 optimizer。
- 进入指标：各 active 轴 mean absolute error 分别达标；inactive 轴不漂移；协调残差与 0B 相近；低 height command 不触发虚假的 height deficit。

### 后续组合姿态

- 先加入 roll+pitch、roll+yaw、pitch+yaw，再加入三轴和 height 混合；每次只增加少量 mode 概率。
- 保持窄等效负载分布，组合姿态通过后再叠加 noise、Kp/Kd 或小 push，不能同时开启。
- 只改变 command distribution 时可继续加载 actor/critic/estimator；optimizer 是否保留做一个小对照。
- 最终回到完整机械臂，对代表性静态姿态和运动轨迹做短程微调与 sim2sim 验证。

## 7. 最小消融实验

所有实验固定 seed 集合、训练步数和评估负载网格，记录 fall rate、height deficit ratio、p95 coordination residual、四腿 extension spread 和 torque/action saturation。

| 实验 | 唯一变化 | 预期现象 | 判断标准 |
|---|---|---|---|
| A：COM 强度 | 0B 窄关联 COM vs 独立三轴 ±8 cm | 大立方体组更易学出扭曲姿态，训练方差和最差分位失败上升 | 若宽 COM 组 residual、单侧折叠率显著更高，说明随机化过强 |
| B：协调项 | `stance_coordination=4` vs `0` | 无协调项仍可能追踪高度，但超载时单腿/单侧先折叠 | 在相近 height error 下比较 extension spread 和折叠事件 |
| C：控制约束 | 当前仅 action rate vs 同时加较强 torque、action smoothness | 过强组力矩使用下降，但承载高度和恢复速度变差 | torque 下降同时 deficit/fall 明显上升即为过强 |
| D：完整臂耦合 | 等效负载 vs 完整臂 home/姿态库 | home 静态接近，但臂伸展或运动时等效模型误差增大 | 仅完整臂出现的倾斜或振荡说明动态耦合未覆盖 |

每次消融只改变表中一个因素。若 A、B、C 同时变化，即使结果改善也无法判断是训练分布、协调偏好还是控制限制造成的。
