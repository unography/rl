# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run logging, RNG streams and evaluation of the DreamerV3 example."""
from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from torchrl._utils import logger as torchrl_logger
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


# --- RNG streams -------------------------------------------------------------


POLICY_RNG_STREAM = 0
LEARNER_RNG_STREAM = 1
REPLAY_RNG_STREAM = 2


def stream_seed(seed: int, counter: int, stream: int) -> int:
    """Make one deterministic Torch seed from a seed, a counter and a stream.

    ``stream`` keeps its users independent: a change in the number of draws of
    one stream does not change the sequences of the others.
    """
    rng = np.random.default_rng(seed=[seed, counter, stream])
    words = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return (int(words[0]) << 32) | int(words[1])


# --- Run logging and episode bookkeeping -------------------------------------


def append_jsonl(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def latent_state_dim(cfg: DictConfig) -> int:
    return cfg.networks.num_categoricals * cfg.networks.num_classes


def running_episode_state(num_envs: int) -> TensorDict:
    """Return the return and length of the current episode of each environment."""
    return TensorDict(
        {
            "episode_return": torch.zeros(num_envs),
            "episode_length": torch.zeros(num_envs, dtype=torch.long),
        },
        [num_envs],
    )


def completed_training_episodes(
    data: TensorDictBase,
    running: TensorDictBase,
    num_envs: int,
    *,
    extra_keys: Sequence[NestedKey] = (),
) -> TensorDict:
    """Return one record for each episode that ends in a collector batch.

    ``running``, from :func:`running_episode_state`, holds the return and
    length of the current episode of each environment and is advanced in
    place. The records are in driver order: ``time_index`` and ``env_index``
    locate the final transition in ``data``; ``episode_return``,
    ``episode_length`` and ``terminated`` describe the episode; each key of
    ``extra_keys`` holds the ``next`` value of that entry at the final
    transition, such as the achievement counts of a Crafter episode.
    """
    reward = data.get(("next", "reward")).reshape(num_envs, -1).cpu()
    done = data.get(("next", "done")).reshape(num_envs, -1).cpu()
    terminated = data.get(("next", "terminated")).reshape(num_envs, -1).cpu()
    episode_return = torch.empty_like(reward)
    episode_length = torch.empty(reward.shape, dtype=torch.long)
    total = running.get("episode_return")
    length = running.get("episode_length")
    for time_index in range(reward.shape[1]):
        total.add_(reward[:, time_index])
        length.add_(1)
        episode_return[:, time_index] = total
        episode_length[:, time_index] = length
        finished = done[:, time_index]
        total.masked_fill_(finished, 0)
        length.masked_fill_(finished, 0)
    # Time-major, so the records follow the driver order.
    time_index, env_index = done.T.nonzero().unbind(-1)
    records = TensorDict(
        {
            "time_index": time_index,
            "env_index": env_index,
            "episode_return": episode_return[env_index, time_index],
            "episode_length": episode_length[env_index, time_index],
            "terminated": terminated[env_index, time_index],
        },
        [time_index.numel()],
    )
    for key in extra_keys:
        value = data.get(("next", key))
        value = value.reshape(num_envs, -1, *value.shape[data.ndim :]).cpu()
        records.set(key, value[env_index, time_index])
    return records


# --- Evaluation and plotting -------------------------------------------------


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
    """Return True if the run must record the per-update losses."""
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
    import matplotlib.pyplot as plt  # noqa: PLC0415

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
