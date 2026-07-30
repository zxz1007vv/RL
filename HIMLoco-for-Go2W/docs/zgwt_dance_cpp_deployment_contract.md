# ZGWT Dance 与 C++ WholeBodyChickenHead 部署同步契约

本文档是 ZGWT 鸡头舞蹈训练端与 C++ MuJoCo 部署端之间的接口契约，重点覆盖
`wholebodycheckedhead` / `WholeBodyChickenHead` 模式。C++ 中可能存在大小写或历史
拼写差异，接入时必须先定位真实的模式分发和控制调用链。

本文档不是调参建议。部署端必须逐项对齐观测、历史、命令、动作、关节映射、控制
频率和模式切换。仅凭仿真画面或已有 debug 文本不能证明已经对齐。

## 1. 权威来源与同步规则

发生冲突时，按以下优先级确定真实行为：

1. 当前训练运行代码和实际生效配置。
2. 导出的 TorchScript 模型输入/输出 schema。
3. 本文档记录的部署契约快照。
4. Python MuJoCo 参考实现。
5. C++ 旧实现、旧日志、旧 README。

训练端权威文件：

- `legged_gym/envs/zgwt/zgwt_dance_config.py`
- `legged_gym/envs/zgwt/zgwt_dance_robot.py`
- `legged_gym/envs/zgwt/zgwt_dance_utils.py`
- `legged_gym/envs/zgwt/zgwt_config.py`
- `legged_gym/utils/helpers.py`
- `rsl_rl/rsl_rl/modules/him_actor_critic.py`
- `rsl_rl/rsl_rl/modules/him_estimator.py`
- `mujoco/pdandrl.py`
- `mujoco/config_zgwt.yaml`
- `mujoco/config_zgwt_dance.yaml`

以后凡是修改以下任一项目，必须在同一个提交中更新本文档最后的“同步改动记录”：

- observation 维度、顺序、尺度或历史长度；
- command 语义、范围、滤波或归一化；
- action clip、action scale、关节顺序或默认关节角；
- PD 增益、轮子驻车、仿真频率或 decimation；
- policy 导出格式、文件命名或目标 checkpoint；
- C++ WholeBodyChickenHead 的模式切换和 actuator 合并方式。

## 2. 策略身份必须明确

部署配置必须使用包含 `run_name` 和 checkpoint 的版本化文件名，例如：

```text
policy_stage1v2_ckpt4750.pt
```

不得静默使用来源不明的 `policy.pt`。C++ 启动时至少打印：

- 策略绝对路径、文件大小和 SHA-256；
- 配置文件绝对路径和 SHA-256；
- 当前 Git commit；
- 可执行文件构建时间；
- 实际进入的模式名；
- 本文档契约版本。

如果版本化策略不存在，应直接报错，不得自动回退到其他策略。

当前 Stage1v2 从 standv2 checkpoint 4000 恢复，所以新 run 的 checkpoint 编号从
4000 延续。`stage1v2/model_4000.pt` 几乎仍是 standv2，不能把编号 4000 理解为
已经进行了 4000 次 Stage1 更新。当前离线曲线推荐优先评估 checkpoint 4750，最终
选择结果必须写入本文档；选择完成前，部署端不得自行认为“最新就是最好”。

## 3. TorchScript 策略接口

导出的 HIM 策略已包含 estimator 和 actor：

- 输入：360 个 `float32`；
- 输出：16 个 `float32` action；
- 单帧观测：60 维；
- 历史长度：6 帧；
- C++ 不得额外运行 estimator，也不得自行添加 base linear velocity。

加载模型后必须检查 TorchScript schema，并对输入、输出尺寸和 finite 状态做
fail-fast 校验。

## 4. 单帧 60 维 observation

| 索引 | 维度 | 内容 | 缩放与要求 |
| --- | ---: | --- | --- |
| `[0:3]` | 3 | base angular velocity | body frame，rad/s，乘 `0.25` |
| `[3:6]` | 3 | projected gravity | `R_world_to_body * [0,0,-1]` |
| `[6:12]` | 6 | dance commands | 顺序和归一化见下一节 |
| `[12:28]` | 16 | `q - q_default` | 乘 `1.0`，轮子位置误差强制为 0 |
| `[28:44]` | 16 | `qd` | 乘 `0.05`，轮子速度保留 |
| `[44:60]` | 16 | previous effective action | 未乘 `action_scale`，轮子 action 为 0 |

整帧最终 clip 到 `[-100,100]`。策略不接收机械臂关节、机械臂末端状态或 base
linear velocity。机械臂对狗的影响通过 MuJoCo 动力学自然体现。

### 4.1 projected gravity 与坐标系

必须计算：

```text
projected_gravity = R_world_to_body * [0,0,-1]
```

检查约定：

- 水平站立：`[0,0,-1]`；
- 正 roll：gravity 的 y 分量为负；
- 正 pitch：gravity 的 x 分量为正；
- 只改变 yaw：projected gravity 不变。

MuJoCo free joint 常用 `[w,x,y,z]`，Isaac Gym root state 使用 `[x,y,z,w]`。C++ 不得
复制未确认 quaternion 顺序的公式。angular velocity 必须是狗 `BASE_LINK` 的
body-frame 角速度。

## 5. 六帧历史

历史展平顺序必须为：

```text
[obs_t, obs_t-1, obs_t-2, obs_t-3, obs_t-4, obs_t-5]
```

最新帧在最前。进入 RL 时用当前合法 observation 重复六次，上一 action 为零。不能
使用未初始化历史，也不能继承上一模式的历史。

每个 policy 周期先使用“上一周期实际应用的 effective action”构造新 observation，
再移入历史并执行本次策略推理。

## 6. Dance command 契约

命令顺序固定为：

```text
[lin_vel_x, lin_vel_y, body_yaw_error,
 body_roll_target, body_pitch_target, absolute_body_height_target]
```

这是固定支撑舞蹈策略，不是 locomotion 策略：

- `lin_vel_x = 0`；
- `lin_vel_y = 0`；
- `body_yaw` 是相对进入 RL 时朝向的角度目标，不是 yaw rate；
- height 只支持下蹲，不支持高于 `0.54 m` 的上升命令。

### 6.1 当前 Stage1v2 物理命令范围

```text
body_yaw:    [-0.025,0.025] rad
body_roll:   [-0.08,0.08] rad
body_pitch:  [-0.08,0.08] rad
body_height: [0.49,0.54] m
```

当前阶段只训练小范围单轴命令。部署评估时暂不发送组合姿态和超范围命令。

### 6.2 固定 observation 归一化

训练采样范围与固定归一化上限不是同一概念。人工切换训练阶段时，不修改以下除数：

```text
yaw_norm = clamp(body_yaw_error / 0.10, -1, 1)
roll_norm = clamp(filtered_roll_target / 0.26, -1, 1)
pitch_norm = clamp(filtered_pitch_target / 0.26, -1, 1)
height_norm = clamp((filtered_height_target - 0.54) / 0.14, -1, 0)
vx_norm = 0
vy_norm = 0
```

必须通过以下固定样例：

| 输入 | 正确归一化结果 |
| --- | ---: |
| height `0.54` | `0.0` |
| height `0.49` | `-0.357142857` |
| height `0.40` | `-1.0` |
| yaw error `0.0067` | `0.067` |
| roll `0.08` | `0.3076923` |
| pitch `-0.08` | `-0.3076923` |

禁止沿用旧 locomotion 的 `height * 2` 或姿态命令 `* 0.25`。

### 6.3 yaw reference、限速和滤波

进入 RL 时捕获当前 base yaw：

```text
relative_yaw = wrap_to_pi(current_yaw - rl_entry_yaw)
body_yaw_error = wrap_to_pi(filtered_yaw_target - relative_yaw)
wrap_to_pi(x) = atan2(sin(x), cos(x))
```

进入 RL 时命令初始化为 `yaw=roll=pitch=0, height=0.54`。命令以 100 Hz 更新：

- yaw target slew rate：`0.30 rad/s`，每个 policy step 最大变化 `0.003 rad`；
- roll、pitch、height 使用一阶低通；
- `time_constant=0.5 s`；
- `alpha = 1-exp(-0.01/0.5)`。

策略 observation 使用滤波后的命令，而不是手柄原始值。

## 7. 关节顺序与 action

策略关节顺序：

```text
0  FAR_ABAD_JOINT       8  RAR_ABAD_JOINT
1  FAR_HIP_JOINT        9  RAR_HIP_JOINT
2  FAR_KNEE_JOINT      10  RAR_KNEE_JOINT
3  FAR_FOOT_JOINT      11  RAR_FOOT_JOINT
4  FBL_ABAD_JOINT      12  RBL_ABAD_JOINT
5  FBL_HIP_JOINT       13  RBL_HIP_JOINT
6  FBL_KNEE_JOINT      14  RBL_KNEE_JOINT
7  FBL_FOOT_JOINT      15  RBL_FOOT_JOINT
```

轮子索引为 `[3,7,11,15]`。默认关节角：

```text
[0,0.6,-1.2,0, 0,0.6,-1.2,0,
 0,-0.6,1.2,0, 0,-0.6,1.2,0]
```

MuJoCo 的 qpos、qvel 和 actuator 顺序不得假设与策略一致。必须按名称建立显式映射，
并在启动时打印：

```text
policy index -> joint name -> qpos address -> dof address -> actuator id
```

策略 action 处理：

```text
effective_action = clamp(policy_action, -2.0, 2.0)
effective_action[wheel_indices] = 0
q_des = q_default + 0.25 * effective_action
```

腿部名义 PD：ABAD `kp=90,kd=1`，HIP/KNEE `kp=120,kd=1`。最终 torque 按训练
URDF 或当前 MuJoCo actuator ctrlrange 裁剪。

## 8. 轮子驻车

Dance 策略忽略四个 wheel action，必须同时满足：

1. policy wheel action 清零；
2. previous-action observation 中 wheel action 清零；
3. `q-q_default` 中 wheel position error 清零；
4. wheel qd 保留；
5. 进入 RL 时捕获当前 wheel angle；
6. RL 期间始终使用位置驻车。

```text
wheel_torque = 25.0 * wheel_position_error - 2.5 * wheel_velocity
```

wheel position error 是否 wrap 必须依据实际 joint 类型验证，不能让跨 `±pi` 产生异常
大力矩。

## 9. 控制频率与 WholeBodyChickenHead 合并

训练端时序：

```text
physics dt = 0.002 s       # 500 Hz
decimation = 5
policy dt = 0.010 s        # 100 Hz
```

每 5 个 physics tick 更新一次命令、observation、history 和 policy action。每个 2 ms
physics tick 都必须用保持的 policy action 和最新 q/qd 重新计算腿部 PD torque、轮子
驻车 torque 和机械臂控制 torque，然后执行一次 `mj_step`。保持的是 policy action，
不是五个子步内完全不变的 PD torque。

WholeBodyChickenHead 中：

- 狗 RL 只写狗的 16 个 actuator；
- 机械臂传统控制只写机械臂 actuator；
- 两组 actuator 按名称映射，并验证不重复；
- 每个 physics tick 构造一次完整 ctrl vector；
- dog/arm torque 写入各自 actuator id 后统一提交；
- 禁止两个控制器先后用 `ctrl[:]` 覆盖彼此；
- 机械臂关节不得插入 RL observation 或 action；
- 若存在多线程，必须排除并发写 `ctrl/qpos/qvel` 和重复 `mj_step`。

## 10. RL 入场初始化

进入 RL 时必须一次性完成：

1. 确认 fold -> stand 已完成且机械臂传统控制稳定；
2. 捕获当前 base yaw 为相对 yaw 基准；
3. 捕获当前四个 wheel angle 为驻车目标；
4. 清空 previous action；
5. neutral command 设为 `yaw=roll=pitch=0,height=0.54`；
6. 用当前 observation 重复初始化六帧历史；
7. 清除其他模式残留的 locomotion command 和 history。

允许保留 1 秒 smoothstep action blend：

```text
s = clamp(elapsed/1.0, 0, 1)
blend = s*s*(3-2*s)
effective_action = clipped_policy_action * blend
```

previous-action observation 必须记录实际 blended effective action。

## 11. Debug 可信度与新鲜度要求

### 11.1 Debug 只能是线索

已有日志可能来自旧二进制、旧配置、不再执行的模式分支，或者来自单独重算的“显示
变量”，而不是真正送入策略/控制器的变量。也可能只是日志标签更新了，实际控制路径
仍使用旧数据。

因此不能仅根据 `scaled_obs_cmd=...` 断言实际 policy input 就是该值。必须从配置读取、
命令状态、observation 写入、history flatten、TorchScript forward、action 后处理、
torque 写入到 `mj_step` 逐级追踪真实数据流。

### 11.2 直接打印实际消费的数据

禁止为了日志再次独立计算一份 normalized command。应直接打印：

- 真正写入 `obs[6:12]` 的六个元素；
- 真正传入 TorchScript 的 360 维 tensor 对应切片；
- TorchScript 原始输出；
- clip、blend、wheel mask 后的 effective action；
- 实际 q_des；
- 最终写入 `d.ctrl` 的 dog/arm torque。

打印代码应紧邻实际消费点，并引用同一对象或同一不可变快照，避免显示值与执行值分叉。

### 11.3 日志来源标识

每组关键日志至少包含：

```text
contract_version
git_commit
build_timestamp
executable_path
mode_name
policy_path + policy_sha256
config_path + config_sha256
physics_tick
policy_tick
sim_time
```

缺少这些信息的旧日志不得作为最终对齐证据。

### 11.4 验证 debug 实时更新

执行确定性命令注入：

1. height 保持 `0.54`，确认实际 `obs[11]` 为 `0`；
2. 改为 `0.50`，确认后续 policy tick 中滤波值和 `obs[11]` 连续变化；
3. 注入 yaw error `0.0067`，确认实际 observation 对应元素约为 `0.067`；
4. 同时核对 360 维 policy input 最新帧的对应位置；
5. 确认旧历史帧按时间后移，而不是六帧同时覆盖；
6. 确认日志 tick、sim time 和输入变化同步递增。

如果控制效果变化但 debug 不变，或 debug 变化而 TorchScript 输入不变，应判定 debug
路径失效并修复，不能继续依据该日志分析策略。

### 11.5 排除旧二进制

C++ 修改后必须重新构建，并通过运行时打印的 Git commit、build timestamp 和可执行文件
绝对路径确认启动的是新二进制。不能只根据终端目录判断。

## 12. Python/C++ 对拍验收

使用同一组固定原始状态生成 golden vector：

1. neutral；
2. 小范围 roll/pitch/yaw；
3. crouch；
4. yaw 接近 `±pi` wrap 边界。

逐元素比较 60 维 observation、360 维 history、16 维 policy action、effective action、
12 个腿关节 q_des 和名义 dog torque。建议容差：

```text
observation max abs diff <= 1e-6
policy action max abs diff <= 1e-5
q_des max abs diff <= 1e-6
torque max abs diff <= 1e-4
```

还必须做 actuator 隔离测试：机械臂输出关闭时，同一 MuJoCo state 下 dog-only dance 与
WholeBodyChickenHead 的 dog observation、action 和 torque 必须一致。开启机械臂后，
狗的索引和数据构造算法保持不变，机械臂只能通过真实动力学影响下一状态。

## 13. 给 C++ AI/开发者的执行要求

接入本文档时，不要只输出审查报告。应完成：

1. 确认 WholeBodyChickenHead 真实入口和调用链；
2. 建立“训练端 / C++ 当前实现 / 是否一致 / 修改位置”差异表；
3. 直接修改所有不一致的数据路径；
4. 增加模型、配置、关节和 actuator 的 fail-fast 校验；
5. 增加 debug 来源和新鲜度验证；
6. 编译并运行 golden-vector 对拍；
7. 运行短时 headless neutral 和单轴命令测试；
8. 用中文报告改动文件、测试命令、最大误差和遗留动力学差异；
9. 在同一次提交中更新本文档“同步改动记录”。

只有观测、动作、控制和时序对拍通过后，才能把带臂单侧下沉归因于 payload、COM、
惯量或域随机化不足。

## 14. 同步改动记录

每次修改在表格顶部追加一行，并在表格下补充必要迁移说明。

| 日期 | 契约版本 | 训练 run/checkpoint | 改动摘要 | C++ 同步状态 |
| --- | --- | --- | --- | --- |
| 2026-07-30 | `zgwt-dance-deploy-v1` | Stage1v2，最终 checkpoint 待确认 | 建立观测、命令、动作、轮子驻车、时序、actuator 隔离和 debug 新鲜度契约 | 待 C++ 对拍 |

### 待同步事项

- 导出并确认最终 Stage1v2 版本化 policy 及 SHA-256；
- C++ 打印实际 build/config/policy 身份；
- C++ 使用实际 policy input 验证旧 debug 是否有效；
- 完成 Python/C++ golden-vector 对拍；
- 对拍通过后，再评估带机械臂的长期单侧承载稳定性。
