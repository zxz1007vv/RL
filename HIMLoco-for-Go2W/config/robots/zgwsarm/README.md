# ZGWSARM 带臂训练说明

本目录是 Isaac Gym 与 C++ MuJoCo 的机械臂控制契约副本。当前训练仍然是
“轮足策略 + 传统机械臂控制”，不是全身强化学习。

环境源码已独立放在 `legged_gym/envs/zgwt_arm/`，与普通 `zgwt/` 并列；资产
生成和验证命令位于 `tools/zgwsarm/`。

## 策略接口

- actor 历史观测：360 维；
- critic 观测：78 维；
- 策略动作：16 维，只对应四条轮腿；
- 六个机械臂关节不进入 actor、critic 或 estimator；
- 机械臂姿态只作为环境物理参数和 TensorBoard 诊断信息。

完整资产有 22 个可动 DOF：原轮足 16 个，加机械臂 6 个。组合 URDF 已删除
原模型顶部错误的 `world -> BASE_LINK` 转动关节，否则会出现第 23 个 DOF。
末端执行器碰撞网格已替换为保守圆柱，视觉网格保持不变。

同目录 `zgwsarm.xml` 的 MuJoCo 参考模型显式保留固定连接件
`base_connector_link=0.32036557 kg` 和 `ROBOT_ARM_LINK0=0.84262463 kg` 的完整
惯性，修正后 `BASE_LINK` 子树总质量同样为 `47.5524395 kg`。参考 MJCF 的
16 个狗关节不添加 URDF 中不存在的被动动力学；六个机械臂关节使用
`damping/frictionloss/armature=0.05/0.01/0`。这些条件由
`legged_gym/tests/test_zgwsarm_mjcf.py` 自动检查。

训练、play、静态姿态验证和动力学计算现在统一加载 `zgwsarm.urdf`，使用完整
STL visual。质量、惯量、关节和 collision 也全部来自这一份文件，不再维护
轻量视觉副本。完整网格会增加 Isaac Gym 首次导入和多环境创建时间，这是恢复
完整外观后的预期代价。

## 新站立阶段：ArmStand Aligned V1

任务名：

```bash
python legged_gym/scripts/train.py --task zgwt_dance_arm
```

新 run name 为：

```text
armstand_aligned_v1
```

它从 `Jul31_18-46-40_armstandv1/model_20000.pt` 只读取 actor 和 estimator
作为初始化，critic、Adam 和本阶段迭代号全部重置。因此新 run 从第 0 代开始，
不会覆盖旧 checkpoint，也不会继续旧控制闭环。

仓库不再提供控制律枚举或旧控制分支，机械臂统一使用：

```text
tau = bias(q,dq)
    + M(q) [qdd_ref + Kp(q_ref-q) + Kd(dq_ref-dq)]
```

其中 `Kp=500`、`Kd=45` 是加速度域闭环增益。反馈先形成
`desired_acceleration`，再统一进入逆动力学，返回后不再叠加 PD。

控制器还包括：

- 500 Hz 传统机械臂关节控制，不属于强化学习动作；
- 根据 URDF 质量、惯量和安装外参，用递归 Newton–Euler 算法计算
  `bias(q,dq)`；
- `bias(q,dq)` 包含随底盘姿态变化的重力、机械臂科氏/离心项、底盘角速度
  引起的惯性项以及固定末端工具惯量；
- 目标速度限制：锁定模式 0.8 rad/s，手动模式 0.5 rad/s；
- 目标加速度限制 2.0 rad/s²，速度滤波时间常数 0.005 s；
- 100 N·m 力矩限制；
- 1000 N·m/s 力矩变化率限制；
- 新目标下发前检查整条关节插值路径的限位和 Jacobian；
- 运行时机械臂碰撞力超过 5 N 时终止 episode；
- reset 时机械臂物理状态与目标一致，避免模式进入跳变。

这里没有使用 Isaac Gym 不存在的 `qfrc_bias` 接口，也没有用常量近似重力；
`bias` 和 `M` 都从同一份 URDF 刚体模型实时计算。按 C++ 导出契约，底盘角
速度参与偏置力计算，底盘线加速度和角加速度前馈保持关闭。

针对 sim2sim 中底盘被压弯和四腿外八，新站立阶段加强了高度、姿态、单轮足
位置和自然关节姿态约束，并增加 ABAD 专项自然姿态 penalty。训练使用低幅
质量、质心、摩擦、电机强度、腿部 Kp/Kd 和初始关节位置随机化；wheel-park
仍保持 `Kp=25、Kd=2.5`。

评估：

```bash
python legged_gym/scripts/play.py --task zgwt_dance_arm --checkpoint <ckpt>
```

新 run 产生后，需要把配置中的 `load_run` 指向该 run 再执行 play。导出文件
继续使用 `policy_<run_name>_ckpt<实际 checkpoint>.pt`，同时更新部署快捷文件
`policy.pt`。

## 机械臂姿态库适应阶段

任务名：

```bash
python legged_gym/scripts/train.py --task zgwt_dance_arm_pose_library
```

姿态库适应阶段使用同一个唯一机械臂控制器，不再切换控制律。开始该阶段前，
应把它的父 checkpoint 更新为新完成的 `armstand_aligned_v1`。

该任务只读取以下条件全部满足的 CSV 行：

- `simulation_verified=true`
- `collision_checked=true`
- `singularity_checked=true`
- 实算关节限位余量不小于 0.15 rad
- `min_jacobian_sigma` 不小于 0.02

合成姿态永远不会自动变为 `hardware_verified=true`。

### 1. 生成无奇异候选姿态

```bash
python tools/zgwsarm/generate_pose_candidates.py --count 500
```

该脚本根据 URDF 计算完整 6×6 几何 Jacobian，并拒绝关节限位余量不足或
最小奇异值小于 0.02 的姿态。输出是
`arm_pose_candidates.csv`，所有行仍是 `simulation_verified=false`。

### 2. 在 PhysX 中验证碰撞与静态保持

```bash
python tools/zgwsarm/validate_static_poses.py \
  --device cuda:0 \
  --duration 2.0
```

验证器使用固定基座、完整重力和与训练环境完全相同的逆动力学加关节阻抗，
检查：

- 机械臂 link 的最大接触力；
- 最终最大关节保持误差；
- 最大机械臂力矩占比。

验证器把全部候选及其结果写入 `arm_safe_static_poses.csv`，只有通过的行才
标记 `simulation_verified=true`；训练加载器会再次按标志位和实算阈值过滤。
这些姿态仍然不是经过真机验证的姿态。

当前仓库已用唯一控制器重新验证 500 个候选，2.0 秒 PhysX 静态保持有
455 个通过，`hardware_verified` 仍全部为 false。

### 3. 训练机械臂姿态库适应

完成新站立模型后，把 `zgwt_dance_arm_pose_library` 的父 checkpoint 指向该模型。
每个环境在 reset 时抽取一个安全姿态，并在整个 episode 内保持不变；机械臂
姿态不会加入策略观测。

### 4. 验证动态关节目标控制

在开始 PPO 训练前运行：

```bash
python tools/zgwsarm/validate_dynamic_control.py \
  --task zgwt_dance_arm_pose_library \
  --headless \
  --num_envs 1 \
  --duration 10.0
```

本次 5 秒验证结果为：

```text
最大参考速度：       0.500000 rad/s
最大参考加速度：     2.000000 rad/s²
最大力矩占限幅比例： 0.253345
最大力矩变化率：     999.999939 N·m/s
最终跟踪误差：       0.070416 rad
剩余命令误差：       0.000000 rad
reset 次数：         0
结果：                FAIL（最终误差阈值为 0.05 rad）
```

速度、加速度、力矩、力矩变化率和 reset 均通过，失败项只有最终跟踪误差。
验证器默认时长已改为 10 秒，应先延长收敛窗口复测，不降低 0.05 rad 阈值。
该动态验证不是固定 home 站立阶段的启动前置条件，但进入姿态库或动态臂阶段
前必须通过。它也不代表任意遥操作轨迹都已安全。
后续动态姿态随机化或真机遥操作回放仍需对每条轨迹做完整的碰撞、限位和奇异性
验证。

## 当前训练前门禁

当前版本在未启动 PPO 的情况下已完成：

- 29 项舞蹈任务、机械臂控制与 MuJoCo 参考模型回归测试；
- 训练 URDF 可重复生成，资产为 22 DOF；
- 500 个候选姿态可重复生成；
- 唯一控制器的 2.0 秒 PhysX 静态筛选得到 455 个姿态；
- 5.0 秒动态验证只有最终误差未通过，待用 10 秒窗口复测；
- 旧 ArmStandV1 checkpoint 只作为 actor/estimator 初始化，新站立阶段重置
  critic、优化器和迭代号。

固定 home 的新站立阶段现在可以启动。开始姿态库或动态机械臂阶段前，必须先
完成 10 秒动态复测并把父 checkpoint 更新为新站立模型。

## 数据真实性

`simulation_home` 的 J2 和 J3 位于 URDF 硬限位，只用于第一阶段站立对齐
当前 C++ 仿真，不能进入普通安全姿态库。

真实遥操作轨迹不是固定 Home 站立/姿态库适应阶段的前置条件。Isaac 端接收已经通过检查的
六关节目标，不替代 C++ 的笛卡尔 IK、奇异规避和遥操作映射。将来训练动态
机械臂时，姿态端点和两点之间的完整轨迹都必须逐点检查限位、碰撞与
Jacobian；仅检查两个端点不够。真机编码器零位、方向、控制增益和工具质量
确认前，所有合成数据都保持 `hardware_verified=false`。
