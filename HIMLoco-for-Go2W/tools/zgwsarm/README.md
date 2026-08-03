# ZGWSARM 工具说明

这里放置 ZGWSARM 的离线姿态工具和训练前验证工具。它们不是 PPO 训练入口，
也不会自行开始训练。

## 文件用途

### `generate_pose_candidates.py`

根据 URDF 关节限位和 6×6 几何 Jacobian 离线生成机械臂静态姿态候选。输出
只代表“关节余量与奇异值通过”，不能直接用于训练，因为还没有经过 PhysX
碰撞和保持验证。

```bash
python tools/zgwsarm/generate_pose_candidates.py --count 500
```

### `validate_static_poses.py`

在固定基座 PhysX 中对全部候选姿态执行 2 秒静态保持，检查碰撞力、关节保持
误差和力矩占比，并生成 ArmStandV2 使用的
`arm_safe_static_poses.csv`。
验证控制器与 ArmStandV2 一致，使用惯量归一化计算力矩；控制律发生变化后
必须重新生成验证结果。

```bash
python tools/zgwsarm/validate_static_poses.py \
  --device cuda:0 \
  --duration 2.0
```

### `validate_dynamic_control.py`

创建一个完整的 `zgwt_dance_arm_static` 环境，但不加载 PPO runner。它让机械臂
沿一条通过 Jacobian 检查的安全关节路径运动，验证速度、加速度、力矩、力矩
变化率、最终跟踪误差和 reset 数量。

```bash
python tools/zgwsarm/validate_dynamic_control.py \
  --task zgwt_dance_arm_static \
  --headless \
  --num_envs 1 \
  --duration 10.0
```

5 秒验证曾在其他指标正常时留下 `0.070416 rad` 最终误差，因此默认使用
10 秒收敛窗口，最终误差阈值仍保持 `0.05 rad`。

## 推荐执行顺序

源 URDF 未修改时，不需要每次训练都重建所有文件。完整流程为：

```text
源 URDF 变化
  → generate_pose_candidates.py
  → validate_static_poses.py
  → validate_dynamic_control.py
  → legged_gym/scripts/train.py
```

训练、play 和验证均直接使用完整 `resources/robots/zgwsarm/zgwsarm.urdf`。
日常继续训练通常只运行 `train.py`。修改控制器但没有修改资产或候选采样逻辑
时，至少重新运行静态和动态验证。
