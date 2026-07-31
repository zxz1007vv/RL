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

完整资产有 22 个可动 DOF：原轮足 16 个，加机械臂 6 个。训练 URDF 已删除
原模型顶部错误的 `world -> BASE_LINK` 转动关节，否则会出现第 23 个 DOF。
末端执行器碰撞网格已替换为保守圆柱，视觉网格保持不变。

Isaac 任务实际加载 `zgwsarm_train.urdf`。它保留原模型的质量、惯量、关节和
碰撞，只把高精度 STL visual 换成基础几何，避免每次启动都花大量时间导入
网格。原始 `zgwsarm.urdf` 仍保留完整视觉模型。源 URDF 更新后重新生成：

```bash
python tools/zgwsarm/prepare_training_asset.py
```

## ArmStandV1

任务名：

```bash
python legged_gym/scripts/train.py --task zgwt_dance_arm
```

默认从 `Jul31_14-10-28_stage1v4/model_800.pt` 继承网络权重，新阶段保存到：

```text
logs/ZGWT_DANCE_ARM/*_armstandv1/
```

这一阶段使用完整的 `47.5524395 kg` 带臂模型，机械臂固定在当前 MuJoCo
仿真 home。舞蹈命令、额外 payload/COM、推力、噪声和初始状态随机化全部
关闭，只训练自然站立适应。

当前 Isaac 控制器已实现：

- 500 Hz 传统机械臂关节控制，不属于强化学习动作；
- 根据 URDF 质量、惯量和安装外参，用递归 Newton–Euler 算法计算
  `bias(q,dq)`；
- `bias(q,dq)` 包含随底盘姿态变化的重力、机械臂科氏/离心项、底盘角速度
  引起的惯性项以及固定末端工具惯量；
- 参考加速度前馈 `M(q) qdd_ref`；
- 质量矩阵外的关节阻抗
  `Kp(q_ref-q) + Kd(dq_ref-dq)`，其中 `Kp=500`、`Kd=45`；
- 目标速度限制：锁定模式 0.8 rad/s，手动模式 0.5 rad/s；
- 目标加速度限制 2.0 rad/s²，速度滤波时间常数 0.005 s；
- 100 N·m 力矩限制；
- 1000 N·m/s 力矩变化率限制；
- 新目标下发前检查整条关节插值路径的限位和 Jacobian；
- 运行时机械臂碰撞力超过 5 N 时终止 episode；
- reset 时机械臂物理状态与目标一致，避免模式进入跳变。

控制律为：

```text
tau = bias(q,dq) + M(q) qdd_ref
    + Kp (q_ref-q) + Kd (dq_ref-dq)
```

这里没有使用 Isaac Gym 不存在的 `qfrc_bias` 接口，也没有用常量近似重力；
`bias` 和 `M` 都从同一份 URDF 刚体模型实时计算。按 C++ 导出契约，底盘角
速度参与偏置力计算，底盘线加速度和角加速度前馈保持关闭。

仓库中没有 `ChickenHeadController.cpp`、MuJoCo `mjModel` 的实际参数读取代码
和最终编译配置，因此无法宣称逐行等同于 C++。当前实现是对
`arm_control_contract.yaml` 和组合 URDF 的公式级、参数级对齐。若 C++ 后续
修改了 `qfrc_bias` 的使用方式、附加摩擦或工具质量，必须同步更新控制契约并
重新验证。

评估：

```bash
python legged_gym/scripts/play.py --task zgwt_dance_arm --checkpoint 800
```

这里的 `--checkpoint` 会覆盖任务配置。ArmStandV1 真正训练完成后，还要把
`load_experiment_name/load_run/checkpoint` 改到对应的
`ZGWT_DANCE_ARM/*_armstandv1`，不要继续加载父模型的 800。

## ArmStandV2-Sim

任务名：

```bash
python legged_gym/scripts/train.py --task zgwt_dance_arm_static
```

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

当前仓库已用 `seed=1` 生成 500 个候选，并以 2.0 秒静态保持完成 PhysX
验证，其中 344 个通过。可直接启动 ArmStandV2-Sim；重新生成或修改模型后
应重新运行上述两步验证。

### 3. 训练静态姿态随机化

先把 `zgwt_dance_arm_static` 的父 checkpoint 指向训练完成的 ArmStandV1，
再启动训练。每个环境在 reset 时抽取一个安全姿态，并在整个 episode 内保持
不变；机械臂姿态不会加入策略观测。

### 4. 验证动态关节目标控制

在开始 PPO 训练前运行：

```bash
python tools/zgwsarm/validate_dynamic_control.py \
  --task zgwt_dance_arm_static \
  --headless \
  --num_envs 1 \
  --duration 5.0
```

当前仓库记录的 CPU/PhysX 验证结果为：

```text
最大参考速度：       0.500000 rad/s
最大参考加速度：     2.000000 rad/s²
最大力矩占限幅比例： 0.187262
最大力矩变化率：     999.999939 N·m/s
最终跟踪误差：       0.010966 rad
剩余命令误差：       0.000000 rad
reset 次数：         0
结果：                PASS
```

该测试覆盖传统控制器的一条安全动态轨迹，不代表任意遥操作轨迹都已安全。
后续动态姿态随机化或真机遥操作回放仍需对每条轨迹做完整的碰撞、限位和奇异性
验证。

## 当前训练前门禁

当前版本在未启动 PPO 的情况下已完成：

- 20 项舞蹈任务与机械臂控制单元测试；
- 训练 URDF 可重复生成，资产为 22 DOF；
- 500 个候选姿态可重复生成；
- 2.0 秒 PhysX 静态筛选可重复得到 344 个可用姿态，输出与仓库 CSV
  逐字一致；
- 5.0 秒动态传统控制测试通过；
- 原 `stage1v4/model_800.pt` 可在 ArmStandV1 完整模型中加载，actor、
  critic、estimator 和动作维度不变；短时自然站立测试无 reset、无力矩或
  动作饱和，结果为 `PASS`。

上述 PhysX 联调在当前可用的 CPU 设备上完成；正式训练前仍建议在 4090 上先
运行一次相同的动态验证命令，确认本机 CUDA/PhysX 环境，然后再启动
ArmStandV1。

## 数据真实性

`simulation_home` 的 J2 和 J3 位于 URDF 硬限位，只用于 ArmStandV1 对齐
当前 C++ 仿真，不能进入普通安全姿态库。

真实遥操作轨迹不是 ArmStandV1/V2 的前置条件。Isaac 端接收已经通过检查的
六关节目标，不替代 C++ 的笛卡尔 IK、奇异规避和遥操作映射。将来训练动态
机械臂时，姿态端点和两点之间的完整轨迹都必须逐点检查限位、碰撞与
Jacobian；仅检查两个端点不够。真机编码器零位、方向、控制增益和工具质量
确认前，所有合成数据都保持 `hardware_verified=false`。
