# 运行入口说明

本目录只保留强化学习日常使用的通用入口，不放某一台机器人专用的数据生成或
验证工具。

## `train.py`

PPO 训练入口。它根据 `--task` 从任务注册表读取环境和训练配置，处理父
checkpoint、续训、日志目录及迭代编号。

例如：

```bash
python legged_gym/scripts/train.py --task zgwt_dance_arm --headless
```

## `play.py`

策略评估和部署 policy 导出入口。它加载指定任务及 checkpoint，运行姿态、
稳定性和安全指标统计；是否持续运行由文件末尾的 `evaluation_duration` 控制。

例如：

```bash
python legged_gym/scripts/play.py --task zgwt_dance_arm --checkpoint 800
```

ZGWSARM 的姿态库生成以及静态/动态控制验证位于
`tools/zgwsarm/`，具体用途和执行顺序见该目录的 README。
