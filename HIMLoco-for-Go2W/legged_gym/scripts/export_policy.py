"""不启动仿真环境，直接将 HIM checkpoint 导出为 TorchScript policy。"""

import argparse
import copy
import re
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from rsl_rl.modules import HIMActorCritic


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class PolicyExporterHIM(torch.nn.Module):
    """只保留部署所需的 actor 和 estimator encoder。"""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = actor_critic.num_one_step_obs

    def forward(self, obs_history):
        parts = self.estimator(obs_history).squeeze(0)[0:19]
        vel, latent = parts[:3], parts[3:]
        latent = F.normalize(latent, dim=-1, p=2.0)
        obs_current = obs_history.squeeze(0)[:self.num_one_step_obs]
        actor_input = torch.cat([obs_current, vel, latent], dim=0)
        return self.actor(actor_input.unsqueeze(0)).squeeze(0)


def _export_policy_as_jit(actor_critic, output_path):
    """在 CPU 上导出与 play.py 相同接口的 TorchScript policy。"""
    exporter = PolicyExporterHIM(actor_critic).cpu()
    scripted_policy = torch.jit.script(exporter)
    scripted_policy.save(str(output_path))


def _linear_shapes(state_dict, prefix):
    """按 Sequential 层号返回指定网络中所有 Linear 权重形状。"""
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.weight$")
    layers = []
    for name, value in state_dict.items():
        match = pattern.match(name)
        if match is not None:
            layers.append((int(match.group(1)), tuple(value.shape)))
    if not layers:
        raise ValueError(f"checkpoint 中找不到网络参数：{prefix}*.weight")
    return sorted(layers)


def _build_actor_critic(checkpoint, activation):
    """从 checkpoint 的张量形状恢复与训练时一致的 HIM 网络结构。"""
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 缺少 model_state_dict")
    state_dict = checkpoint["model_state_dict"]

    actor_layers = _linear_shapes(state_dict, "actor.")
    critic_layers = _linear_shapes(state_dict, "critic.")
    estimator_layers = _linear_shapes(state_dict, "estimator.encoder.")

    num_actor_obs = estimator_layers[0][1][1]
    estimator_output_dim = estimator_layers[-1][1][0]
    num_one_step_obs = actor_layers[0][1][1] - estimator_output_dim
    num_critic_obs = critic_layers[0][1][1]
    num_actions = actor_layers[-1][1][0]

    if num_one_step_obs <= 0 or num_actor_obs % num_one_step_obs != 0:
        raise ValueError(
            "无法从 checkpoint 推断合法的 history observation 结构："
            f"num_actor_obs={num_actor_obs}, "
            f"num_one_step_obs={num_one_step_obs}"
        )

    actor_hidden_dims = [shape[0] for _, shape in actor_layers[:-1]]
    critic_hidden_dims = [shape[0] for _, shape in critic_layers[:-1]]
    actor_critic = HIMActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_one_step_obs=num_one_step_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
    )
    actor_critic.load_state_dict(state_dict, strict=True)
    actor_critic.eval()
    return actor_critic, num_actor_obs, num_one_step_obs, num_actions


def _export_filename(checkpoint_path):
    """生成包含 run 名和 checkpoint 编号的可追溯文件名。"""
    run_directory = checkpoint_path.parent.name
    run_match = re.match(
        r"^[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2}_(.+)$",
        run_directory,
    )
    run_label = run_match.group(1) if run_match else run_directory
    run_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label).strip("_.-")
    checkpoint_match = re.match(r"model_(\d+)\.pt$", checkpoint_path.name)
    checkpoint_label = (
        checkpoint_match.group(1)
        if checkpoint_match
        else checkpoint_path.stem
    )
    return f"policy_{run_label or 'unnamed_run'}_ckpt{checkpoint_label}.pt"


def export_checkpoint(checkpoint_path, output_dir, activation="elu"):
    """加载 checkpoint，导出版本化 policy，并更新 policy.pt。"""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    actor_critic, num_obs, num_one_step_obs, num_actions = (
        _build_actor_critic(checkpoint, activation)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = output_dir / _export_filename(checkpoint_path)
    _export_policy_as_jit(actor_critic, versioned_path)
    latest_path = output_dir / "policy.pt"
    if versioned_path.resolve() != latest_path.resolve():
        shutil.copy2(str(versioned_path), str(latest_path))

    # 重新加载并做一次前向对照，确认 TorchScript 与 checkpoint 网络一致。
    scripted_policy = torch.jit.load(str(versioned_path), map_location="cpu")
    validation_obs = torch.zeros(1, num_obs)
    with torch.inference_mode():
        expected_action = actor_critic.act_inference(validation_obs).squeeze(0)
        exported_action = scripted_policy(validation_obs)
    torch.testing.assert_close(exported_action, expected_action)
    max_export_error = torch.max(
        torch.abs(exported_action - expected_action)
    ).item()
    print(f"加载 checkpoint：{checkpoint_path}")
    print(f"checkpoint iteration：{checkpoint.get('iter', '未知')}")
    print(
        "网络接口："
        f"history_obs={num_obs}, one_step_obs={num_one_step_obs}, "
        f"actions={num_actions}"
    )
    print(f"TorchScript 前向最大误差：{max_export_error:.3e}")
    print(f"版本化 policy：{versioned_path}")
    print(f"部署 policy：{latest_path}")
    return versioned_path


def main():
    parser = argparse.ArgumentParser(
        description="不启动 play/仿真，直接导出 HIM TorchScript policy"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="model_*.pt 的路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPOSITORY_ROOT
            / "logs"
            / "ZGWT_DANCE"
            / "exported"
            / "policies"
        ),
        help="policy 输出目录",
    )
    parser.add_argument(
        "--activation",
        default="elu",
        choices=("elu", "relu", "selu", "crelu", "lrelu", "tanh", "sigmoid"),
        help="训练 actor 使用的激活函数",
    )
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output_dir, args.activation)


if __name__ == "__main__":
    main()
