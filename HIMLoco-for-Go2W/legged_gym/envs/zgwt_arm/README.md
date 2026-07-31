# ZGWT 带臂环境架构

本目录与 `legged_gym/envs/zgwt/` 并列，集中保存完整带臂资产使用的环境逻辑。
普通 ZGWT 和鸡头舞蹈任务不依赖本目录；带臂任务单向复用
`zgwt.zgwt_dance_robot.ZgwtDance`。

## 文件职责

- `zgwt_arm_config.py`：ArmStandV1、ArmStandV2-Sim 的环境、完整 reward 和
  PPO 配置；带臂 reward 不继承普通鸡头任务，后续调参只修改该文件；
- `zgwt_arm_robot.py`：22 DOF 资产映射、16 DOF 策略接口、传统机械臂控制和碰撞终止；
- `zgwt_arm_dynamics.py`：从 URDF 读取质量惯量并执行批量递归 Newton–Euler 逆动力学；
- `zgwt_arm_control_utils.py`：姿态 CSV 过滤、类别采样权重、参考轨迹和力矩变化率限制；
- `__init__.py`：只导出任务注册所需的配置类和环境类。

## 包边界

```text
legged_gym/envs/base/       通用资产 DOF 与动作接口能力
          ↓
legged_gym/envs/zgwt/       普通轮足与鸡头舞蹈任务
          ↓
legged_gym/envs/zgwt_arm/   完整机械臂物理模型和传统臂控制
```

机械臂策略观测、动作和 reward 不应加入 `zgwt/`。如果以后真正实现全身强化学习，
应新建独立任务包，而不是改变当前 `zgwt_arm` 的 360/78/16 策略接口。

机器人模型和训练数据不放在环境源码目录：

- URDF/MJCF/STL：`resources/robots/zgwsarm/`；
- 控制契约和姿态 CSV：`config/robots/zgwsarm/`；
- 资产生成与验证命令：`tools/zgwsarm/`；
- 回归测试：`legged_gym/tests/test_zgwt_arm_control.py`。
