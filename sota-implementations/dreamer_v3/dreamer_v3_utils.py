# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run logging, RNG streams, reference diagnostics and evaluation.

Part of the DreamerV3 example. See ``dreamer_v3.py`` for the run loop,
``dreamer_v3_replay.py`` for the replay stream, and ``dreamer_v3_agent.py``
for the networks, optimizer and builders.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import unravel_key

from torchrl._utils import logger as torchrl_logger
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


# ----------------------------------------------------------------------------
# JAX RNG streams
# ----------------------------------------------------------------------------


POLICY_RNG_STREAM = 0
LEARNER_RNG_STREAM = 1
REPLAY_RNG_STREAM = 2


def jax_torch_seed(seed: int, counter: int, stream: int) -> int:
    """Map JAX's two-word per-call seed to one deterministic Torch seed.

    ``stream`` separates independent consumers. JAX derives the learner and the
    policy from different key paths, so they must not share a seed.
    """
    rng = np.random.default_rng(seed=[seed, counter, stream])
    words = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return (int(words[0]) << 32) | int(words[1])


# ----------------------------------------------------------------------------
# Run logging and episode bookkeeping
# ----------------------------------------------------------------------------


def append_jsonl(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def latent_state_dim(cfg: DictConfig) -> int:
    return cfg.networks.num_categoricals * cfg.networks.num_classes


def training_episode_returns(
    data: TensorDictBase,
    running_return: torch.Tensor,
    num_envs: int,
) -> list[tuple[int, int, float]]:
    reward = data.get(("next", "reward")).squeeze(-1)
    done = data.get(("next", "done")).squeeze(-1)
    if num_envs == 1:
        reward = reward.reshape(1, -1)
        done = done.reshape(1, -1)
    completed = []
    for time_index in range(reward.shape[-1]):
        running_return.add_(reward[..., time_index].cpu())
        finished = done[..., time_index].cpu()
        completed.extend(
            (time_index, int(env_index), float(running_return[env_index]))
            for env_index in finished.nonzero().flatten()
        )
        running_return.masked_fill_(finished, 0)
    return completed


# ----------------------------------------------------------------------------
# Reference diagnostics
# ----------------------------------------------------------------------------


def _imagined_values(
    value_loss: DreamerV3ValueLoss, fake_data: TensorDictBase
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online and slow critic predictions over all H+1 imagined features."""
    features = {}
    for key in value_loss.value_model.in_keys:
        root_key = unravel_key(key)
        next_key = (
            ("next", root_key) if isinstance(root_key, str) else ("next", *root_key)
        )
        root = fake_data.get(root_key)
        features[root_key] = torch.cat(
            (root[..., :1, :], fake_data.get(next_key)), dim=-2
        ).detach()
    first = next(iter(features.values()))
    online = TensorDict(features, batch_size=first.shape[:-1], device=first.device)
    slow = online.copy()
    with value_loss.value_model_params.to_module(
        value_loss.value_model, preserve_module_state=False
    ):
        value_loss.value_model(online)
    with value_loss.target_value_model_params.to_module(
        value_loss.value_model, preserve_module_state=False
    ):
        value_loss.value_model(slow)
    key = value_loss.tensor_keys.value
    return online.get(key), slow.get(key)


def _full_horizon_weight(
    actor_loss: DreamerV3ActorLoss, fake_data: TensorDictBase
) -> torch.Tensor:
    """Continuation weights over all H+1 imagined features.

    ``discount_weight`` holds the reference's ``weight[:, :-1]`` -- the H
    entries the policy loss uses. The reference averages the full H+1 series,
    so extend it by one step: ``w_H = w_{H-1} * gamma * con_H``.
    """
    weight = fake_data.get(actor_loss.tensor_keys.discount_weight).float()
    continuation = fake_data.get(
        ("next", actor_loss.tensor_keys.continuation), default=None
    )
    if continuation is None:
        return weight
    gamma = actor_loss.value_estimator.gamma.to(weight)
    last = weight[..., -1:, :] * gamma * continuation[..., -1:, :].float()
    return torch.cat([weight, last], dim=-2)


@torch.no_grad()
def reference_diagnostics(
    *,
    model_loss: DreamerV3ModelLoss,
    actor_loss: DreamerV3ActorLoss,
    value_loss: DreamerV3ValueLoss,
    sample: TensorDictBase,
    state_dim: int,
    rnn_hidden_dim: int,
    use_bfloat16: bool,
    device: torch.device,
) -> dict[str, float]:
    """Recompute the reference's imagination diagnostics for one replay batch.

    World-model loss terms are reported unweighted, as the reference does. The
    pass is read-only: the return-normalization EMA is frozen and the random
    stream is restored afterwards.
    """
    was_training = actor_loss.training
    actor_loss.eval()
    try:
        with torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else []
        ), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16
        ):
            model_loss_td, model_out = model_loss(sample)
            post_state = model_out.get(("next", "state")).reshape(-1, state_dim)
            post_belief = model_out.get(("next", "belief")).reshape(-1, rnn_hidden_dim)
            actor_td, fake_data = actor_loss(
                TensorDict(
                    {"state": post_state, "belief": post_belief},
                    [post_state.shape[0]],
                )
            )
            all_values, slow_values = _imagined_values(value_loss, fake_data)
            policy_dist = actor_loss.actor_model.get_dist(
                fake_data.select(*actor_loss.actor_model.in_keys, strict=False)
            )
            entropy = policy_dist.entropy()
            full_weight = _full_horizon_weight(actor_loss, fake_data)
    finally:
        actor_loss.train(was_training)

    lambda_target = fake_data.get("lambda_target").float()
    all_values = all_values.float()
    return_scale = actor_td["return_scale"].float()
    return_low = actor_td["return_low"].float()
    advantage = (lambda_target - all_values[..., :-1, :]) / return_scale
    normalized_return = (lambda_target - return_low) / return_scale

    def _unweighted(key: str, weight: float) -> float:
        # The loss module returns terms pre-multiplied by their coefficients.
        value = model_loss_td[key].float().mean().item()
        return value / weight if weight else value

    kl_weight = model_loss.lambda_kl
    return {
        "val": all_values.mean().item(),
        "slowval": slow_values.float().mean().item(),
        "ret": normalized_return.mean().item(),
        "ret_max": normalized_return.max().item(),
        "ret_rate": (normalized_return.abs() >= 1.0).float().mean().item(),
        "adv": advantage.mean().item(),
        "adv_mag": advantage.abs().mean().item(),
        "adv_std": advantage.std().item(),
        "ent_action": entropy.float().mean().item(),
        "weight": full_weight.mean().item(),
        "con": actor_td["continuation_mean"].float().item(),
        "rew": fake_data.get(("next", "reward")).float().mean().item(),
        "return_scale": return_scale.item(),
        "return_low": return_low.item(),
        "return_high": actor_td["return_high"].float().item(),
        "loss_dynamic": _unweighted(
            "loss_model_dynamic", kl_weight * model_loss.lambda_dynamic
        ),
        "loss_representation": _unweighted(
            "loss_model_representation", kl_weight * model_loss.lambda_representation
        ),
        "loss_continue": _unweighted("loss_model_continue", model_loss.lambda_continue),
    }


# ----------------------------------------------------------------------------
# Evaluation and plotting
# ----------------------------------------------------------------------------


@torch.no_grad()
def eval_episode_reward(
    env: EnvBase,
    actor: TensorDictModuleBase,
    num_episodes: int,
    max_episode_steps: int,
) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(
                max_steps=max_episode_steps,
                policy=actor,
                break_when_any_done=True,
                auto_cast_to_device=True,
            )
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


def plot_enabled(cfg: DictConfig) -> bool:
    """Whether a run should collect the per-update losses the figure needs."""
    return bool(cfg.logger.output_plot) and _has_matplotlib


def save_run_plot(
    cfg: DictConfig,
    eval_steps: list[int],
    eval_returns: list[torch.Tensor],
    loss_history: list[torch.Tensor],
) -> None:
    if not _has_matplotlib:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )
        return
    import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dep)

    returns = (
        (torch.stack(eval_returns) if eval_returns else torch.empty(0)).cpu().numpy()
    )
    losses = (torch.cat(loss_history) if loss_history else torch.empty(0, 6)).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(eval_steps, returns, marker="o")
    axes[0].set_title(f"{cfg.env.name} eval reward (real env)")
    axes[0].set_xlabel("env_step")
    axes[0].set_ylabel("avg episode return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses[:, 1], label="reco", alpha=0.8)
    axes[1].plot(losses[:, 2], label="reward", alpha=0.8)
    axes[1].plot(losses[:, 0], label="kl", alpha=0.8)
    axes[1].set_title("World-model losses (update step)")
    axes[1].set_xlabel("update step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps"
    )
    fig.tight_layout()
    fig.savefig(cfg.logger.output_plot, dpi=120)
    torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
