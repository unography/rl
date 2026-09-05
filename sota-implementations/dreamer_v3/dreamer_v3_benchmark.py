# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fixed inputs for DreamerV3 learner benchmarks."""

from __future__ import annotations

import torch

from dreamer_v3_utils import latent_state_dim
from omegaconf import DictConfig
from tensordict import TensorDict


def make_replay_sample(
    cfg: DictConfig,
    *,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
) -> TensorDict:
    """Build one fixed-shape replay sample."""
    batch = cfg.replay_buffer.batch_size
    length = cfg.replay_buffer.seq_len
    generator = torch.Generator(device=device).manual_seed(cfg.env.seed + 11)
    is_init = torch.zeros(batch, length, 1, dtype=torch.bool, device=device)
    is_init[::2, 0] = True
    is_init[1::4, length // 2] = True
    done = torch.zeros_like(is_init)
    terminated = torch.zeros_like(is_init)
    done[1::4, length // 2 - 1] = True
    terminated[1::4, length // 2 - 1] = True
    return TensorDict(
        {
            "state": torch.zeros(batch, length, latent_state_dim(cfg), device=device),
            "belief": torch.zeros(
                batch, length, cfg.networks.rnn_hidden_dim, device=device
            ),
            "action": torch.randn(
                batch,
                length,
                action_dim,
                generator=generator,
                device=device,
            ).clamp_(-1, 1),
            "is_init": is_init,
            "next": {
                "observation": torch.randn(
                    batch,
                    length,
                    obs_dim,
                    generator=generator,
                    device=device,
                ),
                "reward": torch.randn(
                    batch,
                    length,
                    1,
                    generator=generator,
                    device=device,
                ),
                "done": done,
                "terminated": terminated,
            },
        },
        [batch, length],
    )
