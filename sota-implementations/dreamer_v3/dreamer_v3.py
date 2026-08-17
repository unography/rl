# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 on Pendulum-v1 — minimal end-to-end training script.

State-based (not pixel-based) to keep the script compact — the 3-D obs is
treated as a flat feature vector whose reconstruction loss sums its event
dimension before averaging batch and time in the model
loss. The wiring is still real:

- collector to replay buffer of sequences
- world model = MLP encoder + RSSMPriorV3 + RSSMPosteriorV3 + MLP decoder + reward head
- RSSM unrolled over each sequence via ``RSSMRolloutV3``
- actor trained via REINFORCE in imagination (``DreamerV3ActorLoss``)
- value trained on the same imagined rollout (``DreamerV3ValueLoss``)
- periodic eval rollouts in the real env, episode reward logged

Plots ``dreamer_v3_pendulum.png`` with two curves: (a) average eval reward,
(b) world-model KL / reconstruction / reward losses.

Usage::

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        collector.total_frames=5000 logger.eval_every=500
"""
from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)

from torchrl import timeit
from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.envs import SerialEnv, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import (
    CatTensors,
    ClipTransform,
    DoubleToFloat,
    InitTracker,
    TensorDictPrimer,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import DreamerV3MLP, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import IndependentNormal
from torchrl.modules.models.model_based_v3 import (
    _dreamer_v3_init,
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
    symlog,
)
from torchrl.objectives.dreamer_v3 import _DreamerV3ImaginationRollout
from torchrl.objectives.utils import SoftUpdate, ValueEstimators

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None
_has_dm_control = importlib.util.find_spec("dm_control") is not None

ReplayIndex: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
ReplaySampleInfo: TypeAlias = dict[str, ReplayIndex]
_REPLAY_CONTEXT_VALID_KEY = ("collector", "context_valid")


def _append_jsonl(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def _to_float(value: torch.Tensor) -> torch.Tensor:
    """Cast distribution parameters and predictions to FP32 like JAX outputs."""
    return value.float()


def _compile_dreamer_v3_learner_heads(
    world_model: TensorDictSequential,
    reward_model: torch.nn.Module,
    continuation_model: torch.nn.Module,
    actor_model: ProbabilisticTensorDictSequential,
) -> None:
    """Compile the stable-shape learner MLPs after recurrent scan warmup."""
    modules = (
        world_model[0][1].module,
        world_model[2].module,
        reward_model,
        continuation_model,
        actor_model[0].module,
    )
    for module in modules:
        # These modules are shared with the collector. Its carrier retains
        # outputs across policy calls, which is incompatible with Inductor's
        # default CUDA graph output-buffer reuse in ``reduce-overhead`` mode.
        module.compile(
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )


def _training_episode_returns(
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


@torch.no_grad()
def _reference_diagnostics(
    *,
    model_loss,
    actor_loss,
    value_loss,
    sample: TensorDictBase,
    state_dim: int,
    rnn_hidden_dim: int,
    use_bfloat16: bool,
    device: torch.device,
) -> dict[str, float]:
    """Recompute the reference's imagination diagnostics for one replay batch.

    Mirrors the scalars that the JAX implementation logs under ``train/`` -- the
    imagined value, lambda return, advantage, policy entropy, continuation
    weight and predicted reward -- so that a Torch run can be compared with a
    reference run term by term. World-model loss terms are reported unweighted,
    as the reference does. The pass is read-only: the return-normalization EMA
    is frozen and the random stream is restored afterwards.
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
            _, all_values, slow_values = value_loss._shared_value_forward(fake_data)
            policy_dist = actor_loss.actor_model.get_dist(
                fake_data.select(*actor_loss.actor_model.in_keys, strict=False)
            )
            entropy = policy_dist.entropy()
    finally:
        actor_loss.train(was_training)

    lambda_target = fake_data.get("lambda_target").float()
    all_values = all_values.float()
    return_scale = actor_td["return_scale"].float()
    return_low = actor_td["return_low"].float()
    advantage = (lambda_target - all_values[..., :-1, :]) / return_scale
    normalized_return = (lambda_target - return_low) / return_scale

    def _unweighted(key: str, weight: float) -> float:
        # The reference logs the raw per-term losses, while the loss module
        # returns them already multiplied by their configured coefficients.
        # Undo the coefficient so both sides are directly comparable.
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
        "weight": fake_data.get(actor_loss.tensor_keys.discount_weight)
        .float()
        .mean()
        .item(),
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


def _collector_action_budget(
    record_budget: int,
    num_envs: int,
    max_episode_steps: int,
) -> int:
    """Convert a JAX driver-record budget to actual control transitions."""
    if record_budget % num_envs:
        raise ValueError(
            "A driver-record budget must be divisible by the number of "
            f"environments, got {record_budget} and {num_envs}."
        )
    vector_records = record_budget // num_envs
    reset_records = (vector_records + max_episode_steps) // (max_episode_steps + 1)
    return (vector_records - reset_records) * num_envs


def _jax_torch_seed(seed: int, counter: int) -> int:
    """Map JAX's two-word per-call seed to one deterministic Torch seed."""
    rng = np.random.default_rng(seed=[seed, counter])
    words = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return (int(words[0]) << 32) | int(words[1])


def _driver_step_for_action(
    action_index: int,
    env_index: int,
    num_envs: int,
    max_episode_steps: int,
) -> int:
    """Return JAX's per-worker global step for a one-based action index."""
    reset_records = 1 + (action_index - 1) // max_episode_steps
    vector_record = action_index + reset_records
    return (vector_record - 1) * num_envs + env_index + 1


def _learner_updates_for_records(
    record_count: int,
    train_ratio: float | None,
    replay_batch_size: int,
    sequence_length: int,
    default_updates: int,
    *,
    first_eligible_record: bool = False,
) -> int:
    """Return the update count produced by JAX's ratio scheduler."""
    if first_eligible_record:
        return 1
    if train_ratio is None:
        return default_updates
    return max(
        1,
        round(record_count * train_ratio / (replay_batch_size * sequence_length)),
    )


def _refresh_replay_context(
    replay_buffer: ReplayBuffer,
    sample_indices: torch.Tensor | tuple[torch.Tensor, ...],
    sample_generations: torch.Tensor,
    state: torch.Tensor,
    belief: torch.Tensor,
) -> None:
    """Write newly inferred posterior entries to the following replay records."""
    if not isinstance(sample_indices, tuple):
        sample_indices = (sample_indices,)
    batch_size, sequence_length = state.shape[:2]
    context_length = sequence_length + 1
    destination_indices = tuple(
        index.reshape(batch_size, context_length)[:, 1:].reshape(-1)
        for index in sample_indices
    )
    destination_generation = sample_generations.reshape(batch_size, context_length)[
        :, 1:
    ].reshape(-1)
    # Replay samples commonly contain overlapping slices from one stream. JAX
    # applies sequence rows in batch order, so the last row deterministically
    # owns every overlapping record. A single CUDA advanced-index write does
    # not define a duplicate-index winner; retain each coordinate's final
    # flattened occurrence before issuing the generation-checked patch.
    coordinates = torch.stack(destination_indices, -1)
    linear_coordinate = coordinates[:, 0]
    for coordinate, size in zip(
        coordinates[:, 1:].unbind(-1), replay_buffer.storage.shape[1:]
    ):
        linear_coordinate = linear_coordinate * int(size) + coordinate
    row = torch.arange(
        linear_coordinate.numel(),
        device=linear_coordinate.device,
        dtype=linear_coordinate.dtype,
    )
    composite = linear_coordinate * (linear_coordinate.numel() + 1) + row
    order = composite.argsort()
    ordered_coordinate = linear_coordinate[order]
    keep_ordered = torch.ones_like(ordered_coordinate, dtype=torch.bool)
    keep_ordered[:-1] = ordered_coordinate[:-1] != ordered_coordinate[1:]
    keep = order[keep_ordered]
    destination_indices = tuple(index[keep] for index in destination_indices)
    destination_index = (
        torch.stack(destination_indices, -1)
        if len(destination_indices) > 1
        else destination_indices[0]
    )
    destination_generation = destination_generation[keep]
    replay_buffer.update_if_present(
        index=destination_index,
        generation=destination_generation,
        patch={
            "state": state.detach()
            .float()
            .reshape(-1, state.shape[-1])[keep.to(state.device)],
            "belief": belief.detach()
            .float()
            .reshape(-1, belief.shape[-1])[keep.to(belief.device)],
            _REPLAY_CONTEXT_VALID_KEY: torch.ones(
                (keep.numel(), 1), dtype=torch.bool, device=state.device
            ),
        },
    )


class _DreamerV3ReplayPipeline:
    """Keep replay one sample ahead and latent writeback one update behind.

    The reference stream prefetches the next batch before the current learner
    call. Its asynchronous train wrapper also returns replay entries from the
    preceding call. Consequently, after ``N`` learner updates it has sampled
    ``N + 1`` batches and applied ``N - 1`` latent refreshes.
    """

    def __init__(self) -> None:
        self._prefetched: tuple[TensorDictBase, ReplaySampleInfo] | None = None
        self._pending_context: tuple[
            ReplaySampleInfo, torch.Tensor, torch.Tensor
        ] | None = None

    @property
    def has_prefetched(self) -> bool:
        return self._prefetched is not None

    @property
    def has_pending_context(self) -> bool:
        return self._pending_context is not None

    def prefetch(self, replay_buffer: ReplayBuffer) -> None:
        """Fill the single reference-sized prefetch slot if it is empty."""
        if self._prefetched is None:
            self._prefetched = replay_buffer.sample(return_info=True)

    def take(
        self, replay_buffer: ReplayBuffer
    ) -> tuple[TensorDictBase, ReplaySampleInfo]:
        """Return the current batch after sampling its successor."""
        self.prefetch(replay_buffer)
        current = self._prefetched
        self._prefetched = replay_buffer.sample(return_info=True)
        return current

    def apply_pending_context(self, replay_buffer: ReplayBuffer) -> None:
        """Apply the preceding refresh after its successor was prefetched.

        This must run before the next compiled RSSM call. Inductor CUDA graphs
        reuse their output buffers, so retaining those outputs across another
        invocation would overwrite the latent entries before replay consumes
        them. Sampling the successor first preserves the reference pipeline:
        the refresh remains invisible to both the current and prefetched
        batches and can first affect the following sample.
        """
        if self._pending_context is not None:
            pending_info, pending_state, pending_belief = self._pending_context
            _refresh_replay_context(
                replay_buffer,
                pending_info["index"],
                pending_info["index_generation"],
                pending_state,
                pending_belief,
            )
            self._pending_context = None

    def stage_context(
        self,
        sample_info: ReplaySampleInfo,
        state: torch.Tensor,
        belief: torch.Tensor,
    ) -> None:
        """Retain the current learner output for the next pipeline step."""
        if self._pending_context is not None:
            raise RuntimeError(
                "The preceding replay context must be applied before staging "
                "another learner output."
            )
        self._pending_context = (sample_info, state, belief)


class _DreamerV3ReplayRecordBuilder:
    """Convert collector transitions into the continuous JAX replay stream."""

    def __init__(self, num_streams: int) -> None:
        self.num_streams = num_streams
        self._started = torch.zeros(num_streams, dtype=torch.bool)

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        if self.num_streams == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2 or data.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected collector data with shape [num_streams, time], got "
                f"{tuple(data.shape)} for {self.num_streams} streams."
            )

        records = []
        stream_ids = torch.arange(self.num_streams, device=data.device)
        record_keys = (
            "action",
            "is_init",
            "state",
            "belief",
            ("collector", "traj_ids"),
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
        )
        for time_index in range(data.shape[1]):
            collector_step = data[:, time_index]
            reset = collector_step.get("is_init").reshape(self.num_streams, -1).any(-1)
            insert_reset = reset & self._started.to(reset.device)
            if insert_reset.any() and not insert_reset.all():
                raise RuntimeError(
                    "The 2D DreamerV3 replay stream requires synchronized episode "
                    "resets across collector environments."
                )

            transition = collector_step.select(*record_keys, strict=True).clone()
            # The collector's root is_init marks the already-filtered current
            # observation. The RSSM transition targets next.observation, so this
            # real action must not be reset. Resets are represented explicitly by
            # the synthetic terminal-to-reset record below.
            transition.set("is_init", torch.zeros_like(transition.get("is_init")))
            transition.set(("collector", "replay_stream"), stream_ids)
            transition.set(
                _REPLAY_CONTEXT_VALID_KEY,
                torch.ones_like(transition.get("is_init"), dtype=torch.bool),
            )

            if insert_reset.any():
                reset_transition = transition.clone()
                reset_transition.set(
                    "action", torch.zeros_like(reset_transition.get("action"))
                )
                reset_transition.set(
                    "state", torch.zeros_like(reset_transition.get("state"))
                )
                reset_transition.set(
                    "belief", torch.zeros_like(reset_transition.get("belief"))
                )
                reset_transition.set(
                    "is_init", torch.ones_like(reset_transition.get("is_init"))
                )
                reset_transition.set(
                    ("next", "observation"),
                    collector_step.get("observation").clone(),
                )
                reset_transition.set(
                    ("next", "reward"),
                    torch.zeros_like(reset_transition.get(("next", "reward"))),
                )
                reset_transition.set(
                    ("next", "done"),
                    torch.zeros_like(reset_transition.get(("next", "done"))),
                )
                reset_transition.set(
                    ("next", "terminated"),
                    torch.zeros_like(reset_transition.get(("next", "terminated"))),
                )
                records.append(reset_transition)

            records.append(transition)
            self._started.fill_(True)

        return torch.stack(records, 1)


class _DreamerV3ShiftedReplayWriter:
    """Materialize JAX's initial record and mutable final context slot.

    Collector transitions already have the shifted representation consumed by
    the Torch learner: their root action and latent context target the next
    observation. JAX replay additionally contains the initial reset record and
    the newest record that has no outgoing transition yet. The latter is kept
    as a generation-tracked tail slot and finalized by the next collector
    transition without changing its replay identity.
    """

    def __init__(self, num_streams: int) -> None:
        self.num_streams = num_streams
        self._tail_index: torch.Tensor | None = None
        self._tail_generation: torch.Tensor | None = None

    @staticmethod
    def _tail_placeholder(records: TensorDictBase) -> TensorDictBase:
        tail = records[:, -1].clone()
        tail.get("action").zero_()
        tail.get("is_init").zero_()
        tail.get("state").zero_()
        tail.get("belief").zero_()
        tail.get(("next", "reward")).zero_()
        tail.get(("next", "done")).zero_()
        tail.get(("next", "terminated")).zero_()
        tail.get(_REPLAY_CONTEXT_VALID_KEY).zero_()
        return tail.unsqueeze(1)

    @staticmethod
    def _storage_get(
        replay_buffer: ReplayBuffer, index: torch.Tensor
    ) -> TensorDictBase:
        if replay_buffer.storage.ndim == 1:
            return replay_buffer.storage[index]
        return replay_buffer.storage[tuple(index.unbind(-1))]

    def _finalize_tail(
        self, replay_buffer: ReplayBuffer, records: TensorDictBase
    ) -> None:
        tail_index = self._tail_index
        tail_generation = self._tail_generation
        if tail_index is None or tail_generation is None:
            return

        stored = self._storage_get(replay_buffer, tail_index)
        incoming = records[:, 0].clone().to(stored.device)
        context_valid = stored.get(_REPLAY_CONTEXT_VALID_KEY)
        incoming.set(
            "state",
            torch.where(context_valid, stored.get("state"), incoming.get("state")),
        )
        incoming.set(
            "belief",
            torch.where(context_valid, stored.get("belief"), incoming.get("belief")),
        )
        incoming.set(
            _REPLAY_CONTEXT_VALID_KEY,
            torch.ones_like(context_valid, dtype=torch.bool),
        )
        result = replay_buffer.update_if_present(
            index=tail_index,
            generation=tail_generation,
            patch=incoming,
        )
        if result.updated_count != self.num_streams:
            raise RuntimeError(
                "The mutable DreamerV3 replay tail was recycled before it "
                "could be finalized."
            )

    def extend(
        self,
        replay_buffer: ReplayBuffer,
        replay_sampler: _DreamerV3ReplaySampler,
        records: TensorDictBase,
    ) -> torch.Tensor:
        """Finalize the prior tail, append new records, and publish them once."""
        if records.ndim != 2 or records.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected replay records with shape [num_streams, time], got "
                f"{tuple(records.shape)} for {self.num_streams} streams."
            )
        self._finalize_tail(replay_buffer, records)
        placeholder = self._tail_placeholder(records)
        if self._tail_index is None:
            appended = torch.cat([records, placeholder], 1)
        else:
            appended = torch.cat([records[:, 1:], placeholder], 1)
        replay_indices = replay_buffer.extend(
            appended if self.num_streams > 1 else appended.reshape(-1)
        )
        replay_sampler.observe_extend(replay_indices, replay_buffer.storage)

        coordinates = torch.as_tensor(replay_indices, dtype=torch.long).reshape(
            appended.shape[1], self.num_streams, replay_buffer.storage.ndim
        )
        tail_index = coordinates[-1]
        if replay_buffer.storage.ndim == 1:
            tail_index = tail_index[:, 0]
        self._tail_index = tail_index.clone()
        self._tail_generation = replay_buffer.writer.generations_of(
            self._tail_index
        ).clone()
        return replay_indices


class _DreamerV3ReplaySampler(SliceSampler):
    """Slice sampler with DreamerV3's small FIFO online-replay component."""

    def __init__(self, *args, online: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.online = online
        self._stream_lengths: torch.Tensor | None = None
        self._online_queue: list[torch.Tensor] = []
        self._sample_count = 0

    @property
    def online_queue_size(self) -> int:
        return len(self._online_queue)

    def observe_extend(self, index: torch.Tensor, storage: LazyTensorStorage) -> None:
        """Queue each newest non-overlapping block once, as JAX replay does."""
        if not self.online:
            return
        index = torch.as_tensor(index, dtype=torch.long)
        if storage.ndim == 1:
            coordinates = index.reshape(-1, 1, 1)
            num_streams = 1
        else:
            num_streams = storage.shape[1:].numel()
            coordinates = index.reshape(-1, num_streams, storage.ndim)
        if self._stream_lengths is None:
            self._stream_lengths = torch.zeros(num_streams, dtype=torch.long)
        elif self._stream_lengths.numel() != num_streams:
            raise RuntimeError(
                "The number of replay streams changed after initialization."
            )

        max_time = storage._max_size_along_dim0()
        for coordinate_row in coordinates:
            self._stream_lengths.add_(1)
            enqueue = (self._stream_lengths > self.slice_len) & (
                (self._stream_lengths - 1).remainder(self.slice_len) == 0
            )
            if enqueue.any():
                starts = coordinate_row[enqueue].clone()
                starts[:, 0].sub_(self.slice_len - 1).remainder_(max_time)
                self._online_queue.extend(starts.unbind(0))

    def sample(self, storage, batch_size: int):
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        # The reference prefetches one uniform batch as soon as replay first has
        # an item, before training becomes eligible and before online blocks are
        # queued. Its second batch drains the warmup backlog. Afterwards each
        # sequential worker add contributes at most one online item to the next
        # prefetched batch; vectorized collection admits one per learner batch.
        if self._sample_count == 0:
            online_limit = 0
        elif self._sample_count == 1:
            online_limit = num_slices
        else:
            online_limit = 1
        num_online = min(online_limit, num_slices, len(self._online_queue))
        num_uniform = num_slices - num_online
        if num_uniform:
            if storage.ndim > 2:
                raise RuntimeError(
                    "DreamerV3 continuous replay supports 1D or 2D storage."
                )
            stored_time = storage.shape[0]
            num_starts = stored_time - seq_length + 1
            if num_starts < 1:
                raise RuntimeError(
                    f"Replay streams have length {stored_time}, but sampling "
                    f"requires {seq_length} records."
                )
            num_streams = 1 if storage.ndim == 1 else storage.shape[1]
            flat_start = torch.randint(
                num_starts * num_streams,
                (num_uniform,),
                generator=self._rng,
            )
            relative_time = flat_start.div(num_streams, rounding_mode="floor")
            stream = flat_start.remainder(num_streams)
            oldest_time = (
                (int(storage._last_cursor_index) + 1) % stored_time
                if storage._is_full
                else 0
            )
            start_time = (relative_time + oldest_time).remainder(stored_time)
            if storage.ndim == 1:
                uniform_starts = start_time.unsqueeze(-1)
            else:
                uniform_starts = torch.stack([start_time, stream], -1)
            uniform_coordinates = self._tensor_slices_from_startend(
                seq_length,
                uniform_starts,
                stored_time,
            ).reshape(num_uniform, seq_length, storage.ndim)
            index_device = uniform_starts.device
        else:
            uniform_coordinates = None
            index_device = self._online_queue[0].device

        if num_online:
            online_starts = torch.stack(
                [self._online_queue.pop(0) for _ in range(num_online)]
            ).to(index_device)
            online_coordinates = self._tensor_slices_from_startend(
                seq_length,
                online_starts,
                storage.shape[0],
            ).reshape(num_online, seq_length, storage.ndim)
        else:
            online_coordinates = None
        if online_coordinates is not None and uniform_coordinates is not None:
            coordinates = torch.cat([online_coordinates, uniform_coordinates], 0)
        elif uniform_coordinates is not None:
            coordinates = uniform_coordinates
        else:
            coordinates = online_coordinates
        self._sample_count += 1
        return coordinates.reshape(-1, storage.ndim).unbind(-1), {}

    def _empty(self) -> None:
        super()._empty()
        self._stream_lengths = None
        self._online_queue.clear()
        self._sample_count = 0


class _DreamerV3Decoder(torch.nn.Module):
    """Shared decoder trunk with one optimizer leaf per observation event."""

    def __init__(
        self,
        cfg: DictConfig,
        input_dim: int,
        event_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not event_dims or any(size <= 0 for size in event_dims):
            raise ValueError(
                f"Decoder event dimensions must be positive: {event_dims}."
            )
        self.backbone = DreamerV3MLP(
            input_dim,
            None,
            depth=cfg.networks.decoder_layers,
            num_cells=cfg.networks.hidden_dim,
            norm_eps=cfg.networks.norm_eps,
        )
        self.output_heads = torch.nn.ModuleList(
            torch.nn.Linear(cfg.networks.hidden_dim, size) for size in event_dims
        )
        self.output_heads.apply(_dreamer_v3_init)

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(state, belief)
        return torch.cat(tuple(head(hidden) for head in self.output_heads), -1)


class _DreamerV3Actor(torch.nn.Module):
    def __init__(self, cfg: DictConfig, action_dim: int):
        super().__init__()
        state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
        self.backbone = DreamerV3MLP(
            state_dim + cfg.networks.rnn_hidden_dim,
            None,
            depth=cfg.networks.actor_layers,
            num_cells=cfg.networks.hidden_dim,
            norm_eps=cfg.networks.norm_eps,
        )
        self.mean_head = torch.nn.Linear(cfg.networks.hidden_dim, action_dim)
        self.std_head = torch.nn.Linear(cfg.networks.hidden_dim, action_dim)
        self.mean_head.apply(_dreamer_v3_init)
        self.std_head.apply(_dreamer_v3_init)
        with torch.no_grad():
            self.mean_head.weight.mul_(0.01)
            self.std_head.weight.mul_(0.01)
        self.action_dim = action_dim
        self.min_std = cfg.networks.policy_min_std
        self.max_std = cfg.networks.policy_max_std

    def forward(
        self, state: torch.Tensor, belief: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # JAX feat2tensor concatenates the deterministic feature before the
        # flattened stochastic feature for every decision/prediction head.
        hidden = self.backbone(belief, state)
        mean = self.mean_head(hidden)
        std = self.std_head(hidden)
        mean = mean.tanh()
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2) + self.min_std
        # JAX's outs.Normal promotes bounded-Normal parameters and sampling to
        # float32 after the BF16 policy network.
        return mean.float(), std.float()


class _DreamerV3PolicyFilter(torch.nn.Module):
    """Update the recurrent latent from the current observation."""

    def __init__(
        self,
        prior_net: torch.nn.Module,
        posterior_net: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.prior_net = prior_net
        self.posterior_net = posterior_net

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        previous_action: torch.Tensor,
        encoded_latents: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reset = is_init
        while reset.ndim < state.ndim:
            reset = reset.unsqueeze(-1)
        state = torch.where(reset, 0, state)
        belief = torch.where(reset, 0, belief)
        previous_action = torch.where(reset, 0, previous_action)
        # Policy filtering only needs the dynamics core. JAX observe() skips the
        # prior predictor here and immediately conditions on the observation.
        belief = self.prior_net._update_belief(state, belief, previous_action)
        _, state = self.posterior_net(belief, encoded_latents)
        # Collector carrier and replay entries use the FP32 entry-space dtype;
        # the next policy call autocasts these values back to BF16 compute.
        return state.float(), belief.float()


class _DreamerV3PolicyCarry(torch.nn.Module):
    """Carry the filtered latent and chosen action to the next environment step."""

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The reference policy computes its RSSM in BF16 but device_get stores
        # replay entries as FP32. Feeding these FP32 carries back through the
        # next autocast policy step recovers the same BF16 compute semantics.
        return state.float(), belief.float(), action.float()


class _DreamerV3AutocastPolicy(TensorDictModuleBase):
    """Run real-world policy networks in BF16 while keeping FP32 outputs."""

    def __init__(self, module: TensorDictModuleBase, enabled: bool) -> None:
        super().__init__()
        self.module = module
        self.enabled = enabled
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        state = tensordict.get("state")
        with torch.autocast(
            device_type=state.device.type,
            dtype=torch.bfloat16,
            enabled=self.enabled and state.device.type == "cuda",
        ):
            return self.module(tensordict)


class _DreamerV3BehaviorPolicySync:
    """Reproduce JAX's delayed policy-parameter handoff.

    The JAX wrapper stages the pre-update policy parameters once, keeps training
    without replacing that pending snapshot, and applies it only after the next
    policy call has already produced its action. The collector therefore owns a
    distinct behavior policy rather than reading live learner parameters.
    """

    def __init__(
        self,
        learner_policy: torch.nn.Module,
        behavior_policy: torch.nn.Module,
    ) -> None:
        self._learner_policy = learner_policy
        self._behavior_policy = behavior_policy
        learner_parameters = tuple(learner_policy.named_parameters())
        behavior_parameters = tuple(behavior_policy.named_parameters())
        learner_names = tuple(name for name, _ in learner_parameters)
        behavior_names = tuple(name for name, _ in behavior_parameters)
        if learner_names != behavior_names:
            raise RuntimeError(
                "Learner and behavior policies must have identical parameter trees."
            )
        self._learner_parameters = tuple(
            parameter for _, parameter in learner_parameters
        )
        self._behavior_parameters = tuple(
            parameter for _, parameter in behavior_parameters
        )
        if any(
            learner.shape != behavior.shape
            for learner, behavior in zip(
                self._learner_parameters, self._behavior_parameters
            )
        ):
            raise RuntimeError(
                "Learner and behavior policy parameter shapes must be identical."
            )
        self._pending: tuple[torch.Tensor, ...] | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @torch.no_grad()
    def stage_before_training(self) -> None:
        """Stage one pre-update snapshot without replacing an older snapshot."""
        if self._pending is None:
            self._pending = tuple(
                parameter.detach().clone() for parameter in self._learner_parameters
            )

    @torch.no_grad()
    def apply_after_action(self) -> None:
        """Apply a staged snapshot after the behavior action was produced."""
        if self._pending is None:
            return
        for target, source in zip(self._behavior_parameters, self._pending):
            target.copy_(source)
        self._pending = None


class _DreamerV3SeededPolicy(TensorDictModuleBase):
    """Give policy inference its own JAX-style per-call random stream."""

    def __init__(self, module: TensorDictModuleBase, seed: int) -> None:
        super().__init__()
        self.module = module
        self.seed = seed
        self.counter = 0
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys

    def reset_counter(self) -> None:
        """Restart the real-action stream after non-environment policy probes."""
        self.counter = 0

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        reference = tensordict.get("state", None)
        if reference is None:
            reference = tensordict.get(self.in_keys[0])
        devices = [reference.device] if reference.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(_jax_torch_seed(self.seed, self.counter))
            self.counter += 1
            return self.module(tensordict)


class _DreamerV3Optimizer(torch.optim.Optimizer):
    """JAX DreamerV3's AGC, RMS scaling, momentum, and warmup chain."""

    def __init__(
        self,
        parameters,
        *,
        lr: float = 4e-5,
        agc: float = 0.3,
        parameter_norm_min: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": lr,
                "agc": agc,
                "parameter_norm_min": parameter_norm_min,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "warmup_steps": warmup_steps,
                "step": 0,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            step = group["step"]
            warmup_steps = group["warmup_steps"]
            schedule_step = step - 1
            warmup = min(1.0, schedule_step / warmup_steps) if warmup_steps else 1.0
            learning_rate = group["lr"] * warmup

            # Updating each small Dreamer head parameter independently issues
            # thousands of tiny CUDA kernels. Bucket parameters by device and
            # dtype so the mathematically identical optimizer transforms can
            # use PyTorch's multi-tensor kernels.
            buckets: dict[
                tuple[torch.device, torch.dtype], list[torch.nn.Parameter]
            ] = {}
            for parameter in group["params"]:
                if parameter.grad is not None:
                    buckets.setdefault((parameter.device, parameter.dtype), []).append(
                        parameter
                    )

            for parameters in buckets.values():
                gradients = [parameter.grad.float() for parameter in parameters]
                if group["agc"]:
                    gradient_norms = list(torch._foreach_norm(gradients))
                    parameter_norms = list(
                        torch._foreach_norm(
                            [parameter.detach().float() for parameter in parameters]
                        )
                    )
                    torch._foreach_clamp_min_(
                        parameter_norms, group["parameter_norm_min"]
                    )
                    maximum_norms = torch._foreach_mul(parameter_norms, group["agc"])
                    gradient_denominators = torch._foreach_maximum(
                        gradient_norms, maximum_norms
                    )
                    gradient_scales = torch._foreach_div(
                        maximum_norms, gradient_denominators
                    )
                    gradients = list(torch._foreach_mul(gradients, gradient_scales))

                rms = []
                momentum = []
                for parameter in parameters:
                    state = self.state[parameter]
                    if not state:
                        state["rms"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["momentum"] = torch.zeros_like(
                            parameter, dtype=torch.float32
                        )
                    rms.append(state["rms"])
                    momentum.append(state["momentum"])
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                torch._foreach_mul_(rms, beta2)
                torch._foreach_addcmul_(rms, gradients, gradients, value=1 - beta2)
                rms_hat = torch._foreach_div(rms, 1 - beta2**step)
                rms_denominator = torch._foreach_sqrt(rms_hat)
                torch._foreach_add_(rms_denominator, group["eps"])
                normalized = torch._foreach_div(gradients, rms_denominator)
                torch._foreach_mul_(momentum, beta1)
                torch._foreach_add_(momentum, normalized, alpha=1 - beta1)
                momentum_hat = torch._foreach_div(momentum, 1 - beta1**step)
                if parameters[0].dtype != torch.float32:
                    momentum_hat = [
                        update.to(parameter.dtype)
                        for update, parameter in zip(momentum_hat, parameters)
                    ]
                torch._foreach_add_(parameters, momentum_hat, alpha=-learning_rate)
        return loss


class _DreamerV3CudaGraphLearner:
    """Capture a fixed-shape learner forward/backward while keeping updates eager."""

    def __init__(
        self,
        forward_backward: Callable[
            [TensorDictBase, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        optimizer: _DreamerV3Optimizer,
        target_updater: SoftUpdate,
        parameters: list[torch.nn.Parameter],
        target_parameters: tuple[torch.Tensor, ...],
        mutable_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        self.forward_backward = forward_backward
        self.optimizer = optimizer
        self.target_updater = target_updater
        self.parameters = parameters
        self.target_parameters = target_parameters
        self.mutable_buffers = mutable_buffers
        self.device = parameters[0].device
        if self.device.type != "cuda" or any(
            parameter.device != self.device for parameter in parameters
        ):
            raise RuntimeError(
                "CUDA graph learner parameters must share one CUDA device."
            )
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_sample: TensorDictBase | None = None
        self._outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _snapshot_state(self) -> dict[str, object]:
        optimizer_state = []
        for parameter in self.parameters:
            state = self.optimizer.state[parameter]
            optimizer_state.append(
                {
                    key: value.clone()
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                    for key, value in state.items()
                }
            )
        return {
            "parameters": tuple(
                parameter.detach().clone() for parameter in self.parameters
            ),
            "gradients": tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in self.parameters
            ),
            "target_parameters": tuple(
                parameter.detach().clone() for parameter in self.target_parameters
            ),
            "mutable_buffers": tuple(
                buffer.detach().clone() for buffer in self.mutable_buffers
            ),
            "optimizer_state": optimizer_state,
            "optimizer_steps": tuple(
                group["step"] for group in self.optimizer.param_groups
            ),
            "cpu_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device),
        }

    @torch.no_grad()
    def _restore_state(self, state: dict[str, object]) -> None:
        for parameter, value in zip(self.parameters, state["parameters"]):
            parameter.copy_(value)
        for parameter, gradient in zip(self.parameters, state["gradients"]):
            if gradient is None:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
        for parameter, value in zip(self.target_parameters, state["target_parameters"]):
            parameter.copy_(value)
        for buffer, value in zip(self.mutable_buffers, state["mutable_buffers"]):
            buffer.copy_(value)
        for parameter, saved_state in zip(self.parameters, state["optimizer_state"]):
            current_state = self.optimizer.state[parameter]
            for key, value in saved_state.items():
                if torch.is_tensor(value):
                    current_state[key].copy_(value)
                else:
                    current_state[key] = copy.deepcopy(value)
        for group, step in zip(self.optimizer.param_groups, state["optimizer_steps"]):
            group["step"] = step
        torch.random.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)

    def _capture(self, sample: TensorDictBase) -> None:
        self._static_sample = sample.clone()
        state = self._snapshot_state()
        if any(gradient is None for gradient in state["gradients"]):
            raise RuntimeError(
                "CUDA graph learner capture requires every trainable parameter "
                "to receive a gradient in the initial eager update."
            )

        # Materialize the newly wrapped learner heads and their backward graphs.
        # This extra execution is fully rolled back below and does not advance
        # optimizer, target, normalization, or random state.
        with torch.cuda.device(self.device):
            warm_outputs = self.forward_backward(self._static_sample, False)
            del warm_outputs
            torch.cuda.synchronize(self.device)
            self._restore_state(state)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self.forward_backward(self._static_sample, False)
            torch.cuda.synchronize(self.device)
            self._restore_state(state)
            torch.cuda.synchronize(self.device)
        self._graph = graph
        self._outputs = outputs

    def __call__(
        self, sample: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.device):
            if self._graph is None:
                self._capture(sample)
            else:
                self._static_sample.copy_(sample)
            self._graph.replay()
            self.optimizer.step()
            self.target_updater.step()
        return self._outputs


@torch.no_grad()
def adaptive_grad_clip_(parameters, clip: float, minimum: float = 1e-3) -> None:
    """Clip each parameter gradient relative to its parameter norm."""
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.norm()
        parameter_norm = parameter.detach().norm().clamp_min(minimum)
        scale = (clip * parameter_norm / grad_norm.clamp_min(1e-12)).clamp_max(1)
        parameter.grad.mul_(scale)


def make_env(cfg: DictConfig, seed: int | None = 0) -> TransformedEnv:
    if cfg.env.backend == "gym":
        base_env = GymEnv(cfg.env.name, device="cpu")
    elif cfg.env.backend == "dm_control":
        if not _has_dm_control:
            raise ImportError(
                "The DMC DreamerV3 preset requires dm_control. Install the "
                "optional dm_control dependencies before running it."
            )
        from torchrl.envs.libs.dm_control import DMControlEnv

        # The reference DMC preset calls suite.load() without a task random
        # state. Keep DMC reset randomness unseeded when env.use_seed is false.
        # For explicitly seeded variants, pass the seed at construction because
        # calling set_seed() performs a hidden reset and shifts the task stream.
        base_env = DMControlEnv(
            cfg.env.name,
            cfg.env.task,
            device="cpu",
            _seed=seed if cfg.env.use_seed else None,
        )
    else:
        raise ValueError(f"Unknown environment backend {cfg.env.backend!r}.")

    env = TransformedEnv(base_env)
    if cfg.env.backend == "dm_control":
        env.append_transform(
            CatTensors(
                # The JAX DictConcat encoder sorts observation dictionary keys.
                in_keys=sorted(base_env.observation_spec.keys()),
                out_key="observation",
            )
        )
        # The reference carries and replays the unsquashed Normal sample but
        # clips the control passed to DMC at the environment boundary.
        env.append_transform(ClipTransform(in_keys_inv=["action"], low=-1.0, high=1.0))
    env.append_transform(DoubleToFloat())
    env.append_transform(StepCounter(max_steps=cfg.env.max_episode_steps))
    env.append_transform(InitTracker())
    if cfg.env.backend != "dm_control" and cfg.env.use_seed:
        env.set_seed(seed)
    return env


def build_world_model(*, cfg: DictConfig, obs_dim: int, action_dim: int):
    """MLP encoder + RSSMRolloutV3 + MLP decoder + reward head.

    Returns a TensorDictSequential whose forward consumes a trajectory batch
    and writes every key DreamerV3ModelLoss expects.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    encoder = TensorDictSequential(
        TensorDictModule(
            symlog,
            in_keys=[("next", "observation")],
            out_keys=[("next", "symlog_observation")],
        ),
        TensorDictModule(
            DreamerV3MLP(
                in_features=obs_dim,
                # The reference encoder embedding is its final normalized
                # hidden activation, with no additional output projection.
                out_features=None,
                depth=cfg.networks.encoder_layers,
                num_cells=cfg.networks.hidden_dim,
                norm_eps=cfg.networks.norm_eps,
            ),
            in_keys=[("next", "symlog_observation")],
            out_keys=[("next", "encoded_latents")],
        ),
    )

    prior_net = RSSMPriorV3(
        action_shape=torch.Size([action_dim]),
        hidden_dim=cfg.networks.hidden_dim,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        action_dim=action_dim,
        unimix=cfg.networks.unimix,
        recurrent_model=cfg.networks.recurrent_model,
        num_blocks=cfg.networks.num_blocks,
        num_layers=cfg.networks.dynamics_layers,
        prior_num_layers=cfg.networks.prior_layers,
        norm_eps=cfg.networks.norm_eps,
    )
    rssm_prior = TensorDictModule(
        prior_net,
        in_keys=["state", "belief", "action"],
        out_keys=[
            ("next", "prior_logits"),
            ("next", "state"),
            ("next", "belief"),
        ],
    )

    posterior_net = RSSMPosteriorV3(
        hidden_dim=cfg.networks.hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        obs_embed_dim=cfg.networks.hidden_dim,
        unimix=cfg.networks.unimix,
        use_rms_norm=True,
        num_layers=cfg.networks.posterior_layers,
        norm_eps=cfg.networks.norm_eps,
    )
    rssm_posterior = TensorDictModule(
        posterior_net,
        in_keys=[("next", "belief"), ("next", "encoded_latents")],
        out_keys=[("next", "posterior_logits"), ("next", "state")],
    )

    # Real collector transitions keep their filtered z_t / h_t and action a_t.
    # The replay builder inserts an explicit zero-action terminal-to-reset edge
    # between episodes and marks only that edge is_init. This lets windows cross
    # episode boundaries and trains the reset observation exactly once.
    rollout = RSSMRolloutV3(rssm_prior, rssm_posterior, reset_key="is_init")

    decoder_event_dims = tuple(cfg.networks.decoder_event_dims or (obs_dim,))
    if sum(decoder_event_dims) != obs_dim:
        raise ValueError(
            "Decoder event dimensions must sum to the flattened observation "
            f"size, got {decoder_event_dims} for {obs_dim}."
        )
    decoder = TensorDictModule(
        _DreamerV3Decoder(
            cfg,
            state_dim + cfg.networks.rnn_hidden_dim,
            decoder_event_dims,
        ),
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reco_pixels")],
    )

    reward_net = DreamerV3MLP(
        in_features=state_dim + cfg.networks.rnn_hidden_dim,
        out_features=cfg.networks.num_reward_bins,
        depth=cfg.networks.reward_layers,
        num_cells=cfg.networks.hidden_dim,
        outscale=0.0,
        norm_eps=cfg.networks.norm_eps,
    )
    reward_decoder = SymExpTwoHot(cfg.networks.num_reward_bins)
    reward_head = TensorDictSequential(
        TensorDictModule(
            reward_net,
            in_keys=[("next", "belief"), ("next", "state")],
            out_keys=[("next", "reward_logits")],
        ),
        TensorDictModule(
            _to_float,
            in_keys=[("next", "reward_logits")],
            out_keys=[("next", "reward_logits")],
        ),
        TensorDictModule(
            reward_decoder,
            in_keys=[("next", "reward_logits")],
            out_keys=[("next", "reward")],
        ),
    )

    continuation_net = DreamerV3MLP(
        in_features=state_dim + cfg.networks.rnn_hidden_dim,
        out_features=1,
        depth=cfg.networks.reward_layers,
        num_cells=cfg.networks.hidden_dim,
        norm_eps=cfg.networks.norm_eps,
    )
    continuation_head = TensorDictSequential(
        TensorDictModule(
            continuation_net,
            in_keys=[("next", "belief"), ("next", "state")],
            out_keys=[("next", "continue_pred")],
        ),
        TensorDictModule(
            _to_float,
            in_keys=[("next", "continue_pred")],
            out_keys=[("next", "continue_pred")],
        ),
    )

    world_model = TensorDictSequential(
        encoder, rollout, decoder, reward_head, continuation_head
    )
    return world_model, prior_net, reward_net, reward_decoder, continuation_net


def build_imagination_model(*, prior_net, reward_net, reward_decoder):
    """Build imagination operators backed by the trained world-model heads."""
    transition_model = TensorDictSequential(
        TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=["_", "state", "belief"],
        )
    )
    reward_model = TensorDictSequential(
        TensorDictModule(
            reward_net,
            in_keys=["belief", "state"],
            out_keys=["reward_logits"],
        ),
        TensorDictModule(
            _to_float,
            in_keys=["reward_logits"],
            out_keys=["reward_logits"],
        ),
        TensorDictModule(
            reward_decoder,
            in_keys=["reward_logits"],
            out_keys=["reward"],
        ),
    )
    return WorldModelWrapper(transition_model, reward_model)


def build_continuation_model(*, continuation_net):
    """Build the imagination continuation predictor from the trained head."""
    return TensorDictSequential(
        TensorDictModule(
            continuation_net,
            in_keys=["belief", "state"],
            out_keys=["continue_logits"],
        ),
        TensorDictModule(
            _to_float,
            in_keys=["continue_logits"],
            out_keys=["continue_logits"],
        ),
        TensorDictModule(
            torch.nn.Sigmoid(),
            in_keys=["continue_logits"],
            out_keys=["continuation"],
        ),
    )


def build_actor(*, cfg: DictConfig, action_dim: int):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    actor_mlp = _DreamerV3Actor(cfg, action_dim)
    actor_model = ProbabilisticTensorDictSequential(
        TensorDictModule(
            actor_mlp,
            in_keys=["state", "belief"],
            out_keys=["loc", "scale"],
        ),
        ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            default_interaction_type=InteractionType.RANDOM,
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="action_log_prob",
        ),
    )
    with torch.no_grad():
        actor_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, cfg.networks.rnn_hidden_dim),
                },
                [1],
            )
        )
    return actor_model


def build_real_world_actor(
    *,
    world_model: TensorDictSequential,
    actor_model: ProbabilisticTensorDictSequential,
    mixed_precision: bool = False,
) -> TensorDictModuleBase:
    """Build the observation-conditioned recurrent real-environment policy.

    The policy shares the trained encoder, prior, posterior, and actor. At each
    environment step it implements the reference filtering order: encode the
    current observation, advance the deterministic dynamics with the previous
    action, sample the posterior state, and finally sample the action.
    """
    encoder_net = world_model[0][1].module
    rssm_rollout = world_model[1]
    prior_net = rssm_rollout.rssm_prior.module
    posterior_net = rssm_rollout.rssm_posterior.module
    policy = TensorDictSequential(
        TensorDictModule(
            symlog,
            in_keys=["observation"],
            out_keys=["symlog_observation"],
        ),
        TensorDictModule(
            encoder_net,
            in_keys=["symlog_observation"],
            out_keys=["encoded_latents"],
        ),
        TensorDictModule(
            _DreamerV3PolicyFilter(prior_net, posterior_net),
            in_keys=[
                "state",
                "belief",
                "previous_action",
                "encoded_latents",
                "is_init",
            ],
            out_keys=["state", "belief"],
        ),
        actor_model,
        TensorDictModule(
            _DreamerV3PolicyCarry(),
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "state"),
                ("next", "belief"),
                ("next", "previous_action"),
            ],
        ),
    )
    return _DreamerV3AutocastPolicy(policy, enabled=mixed_precision)


def build_value(*, cfg: DictConfig):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    value_model = TensorDictSequential(
        TensorDictModule(
            DreamerV3MLP(
                in_features=state_dim + cfg.networks.rnn_hidden_dim,
                out_features=cfg.networks.num_value_bins,
                depth=cfg.networks.value_layers,
                num_cells=cfg.networks.hidden_dim,
                outscale=0.0,
                norm_eps=cfg.networks.norm_eps,
            ),
            in_keys=["belief", "state"],
            out_keys=["state_value_logits"],
        ),
        TensorDictModule(
            _to_float,
            in_keys=["state_value_logits"],
            out_keys=["state_value_logits"],
        ),
        TensorDictModule(
            SymExpTwoHot(cfg.networks.num_value_bins),
            in_keys=["state_value_logits"],
            out_keys=["state_value"],
        ),
    )
    with torch.no_grad():
        value_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, cfg.networks.rnn_hidden_dim),
                },
                [1],
            )
        )
    return value_model


def build_mb_env(*, cfg: DictConfig, real_env, imagination_model, device: torch.device):
    """Imagination env backed by the trained prior and reward head."""
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    primer_env = TransformedEnv(
        real_env,
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )
    mb_env = DreamerEnv(
        world_model=imagination_model,
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([cfg.networks.rnn_hidden_dim]),
        device=device,
    )
    mb_env.set_specs_from_env(primer_env)
    with torch.no_grad():
        mb_env.rollout(3)
    return mb_env


@torch.no_grad()
def eval_episode_reward(
    env, actor, num_episodes: int, max_episode_steps: int
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


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)

    device = (
        torch.device(cfg.optimization.device)
        if cfg.optimization.device
        else get_available_device()
    )
    replay_device = (
        torch.device(cfg.replay_buffer.device) if cfg.replay_buffer.device else device
    )
    use_cudagraph_learner = bool(
        cfg.optimization.cudagraph_learner and device.type == "cuda"
    )
    num_envs = cfg.collector.num_envs
    if num_envs <= 0:
        raise ValueError(f"collector.num_envs must be positive, got {num_envs}.")
    if cfg.collector.frames_per_batch % num_envs:
        raise ValueError(
            "collector.frames_per_batch must be divisible by collector.num_envs, "
            f"got {cfg.collector.frames_per_batch} and {num_envs}."
        )
    count_reset_records = bool(cfg.collector.count_reset_records)
    collector_action_frames = (
        _collector_action_budget(
            cfg.collector.total_frames,
            num_envs,
            cfg.env.max_episode_steps,
        )
        if count_reset_records
        else cfg.collector.total_frames
    )
    if collector_action_frames % cfg.collector.frames_per_batch:
        raise ValueError(
            "The action budget derived from collector.total_frames must be "
            "divisible by collector.frames_per_batch, got "
            f"{collector_action_frames} and {cfg.collector.frames_per_batch}."
        )
    real_env = make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    metrics_jsonl_path = (
        Path(cfg.logger.metrics_jsonl).resolve() if cfg.logger.metrics_jsonl else None
    )
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("")
    timeit.reset()
    run_timer = timeit("dreamer_v3/run").start()

    (
        world_model,
        prior_net,
        reward_net,
        reward_decoder,
        continuation_net,
    ) = build_world_model(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim)
    world_model = world_model.to(device)
    if cfg.optimization.compile_rssm:
        if use_cudagraph_learner:
            world_model[1].compile_scan(
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        else:
            world_model[1].compile_scan(
                mode=cfg.optimization.compile_mode,
                fullgraph=True,
            )
    imagination_model = build_imagination_model(
        prior_net=prior_net,
        reward_net=reward_net,
        reward_decoder=reward_decoder,
    ).to(device)
    continuation_model = build_continuation_model(continuation_net=continuation_net).to(
        device
    )
    actor_model = build_actor(cfg=cfg, action_dim=action_dim).to(device)
    value_model = build_value(cfg=cfg).to(device)
    imagination_rollout = _DreamerV3ImaginationRollout(
        prior_model=prior_net,
        actor_model=actor_model[0].module,
        reward_model=reward_net,
        reward_decoder=reward_decoder,
        horizon=cfg.optimization.imagination_horizon,
    )
    if cfg.optimization.compile_imagination:
        if use_cudagraph_learner:
            imagination_rollout.compile_scan(
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        else:
            imagination_rollout.compile_scan(
                mode=cfg.optimization.compile_mode,
                fullgraph=True,
            )
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        imagination_model=imagination_model,
        device=device,
    )

    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_mode="separate",
        lambda_dynamic=cfg.optimization.dynamic_loss_weight,
        lambda_representation=cfg.optimization.representation_loss_weight,
        unimix=cfg.networks.unimix,
        lambda_continue=1.0,
        continue_target_scale=1 - 1 / cfg.optimization.continuation_horizon,
        # DreamerV3 sums observation event dimensions and averages batch/time.
        global_average=False,
        detach_output=False,
    ).to(device)
    model_loss.set_keys(pixels="observation")
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        continuation_model=continuation_model,
        imagination_rollout=imagination_rollout,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        policy_loss_mode="dreamer_v3",
        return_normalization_rate=cfg.optimization.return_normalization_rate,
        return_normalization_min_scale=cfg.optimization.return_normalization_min_scale,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    actor_loss.to(device)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_critic_regularization=cfg.optimization.slow_critic_regularization,
    ).to(device)
    if cfg.optimization.compile_replay_value_loss:
        value_loss.compile_replay_value_loss(fullgraph=True)
    if cfg.optimization.shared_imagination_value:
        actor_loss.__dict__["_shared_value_forward"] = value_loss._shared_value_forward
    value_target_updater = SoftUpdate(value_loss, tau=cfg.optimization.slow_critic_tau)

    trainable_parameters = (
        list(world_model.parameters())
        + list(actor_model.parameters())
        + list(value_loss.parameters())
    )
    optimizer = _DreamerV3Optimizer(
        trainable_parameters,
        lr=cfg.optimization.lr,
        agc=cfg.optimization.adaptive_grad_clip,
        beta1=0.9,
        beta2=0.999,
        eps=cfg.optimization.adam_eps,
        warmup_steps=cfg.optimization.warmup_steps,
    )

    real_world_actor = build_real_world_actor(
        world_model=world_model,
        actor_model=actor_model,
        mixed_precision=cfg.optimization.mixed_precision,
    )
    if cfg.optimization.jax_behavior_policy_sync:
        collector_actor = copy.deepcopy(real_world_actor)
        # JAX's policy-key regex also shadows the decoder. Its carry is empty
        # for proprioception and it is action-dead, but include it for an exact
        # parameter-tree handoff rather than only action-equivalent behavior.
        behavior_decoder = copy.deepcopy(world_model[2])
        learner_policy_tree = torch.nn.ModuleList([real_world_actor, world_model[2]])
        behavior_policy_tree = torch.nn.ModuleList([collector_actor, behavior_decoder])
        behavior_policy_sync = _DreamerV3BehaviorPolicySync(
            learner_policy_tree, behavior_policy_tree
        )
    else:
        collector_actor = real_world_actor
        behavior_policy_sync = None
    collector_policy = (
        _DreamerV3SeededPolicy(collector_actor, cfg.env.seed)
        if cfg.optimization.separate_policy_rng
        else collector_actor
    )

    def make_explore_env(index: int):
        seed = cfg.env.seed + 2 + index if cfg.env.use_seed else None
        return TransformedEnv(
            make_env(cfg, seed),
            TensorDictPrimer(
                random=False,
                default_value=0,
                state=Unbounded(state_dim),
                belief=Unbounded(cfg.networks.rnn_hidden_dim),
                previous_action=Unbounded(action_dim),
            ),
        )

    if num_envs == 1:
        explore_env = make_explore_env(0)
    else:
        explore_env = SerialEnv(
            num_envs,
            [
                (lambda index=index: make_explore_env(index))
                for index in range(num_envs)
            ],
        )

    collector = Collector(
        explore_env,
        collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=collector_action_frames,
        policy_device=device,
        env_device="cpu",
        storing_device="cpu",
        exploration_type=ExplorationType.RANDOM
        if cfg.collector.exploration == "random"
        else ExplorationType.MODE,
    )
    if isinstance(collector_policy, _DreamerV3SeededPolicy):
        # Collector construction probes policy output keys once. JAX initializes
        # those keys without consuming a real-action seed, so restart the
        # isolated policy stream before the first environment transition.
        collector_policy.reset_counter()

    replay_sampler = _DreamerV3ReplaySampler(
        # One extra record is the destination slot for the final refreshed
        # posterior. The stream id remains constant across episode resets.
        slice_len=cfg.replay_buffer.seq_len + 1,
        traj_key=("collector", "replay_stream"),
        cache_values=True,
        online=cfg.replay_buffer.online,
    )
    rb = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.replay_buffer.buffer_size,
            ndim=2 if num_envs > 1 else 1,
            device=replay_device,
            consolidated=True,
        ),
        dim_extend=1 if num_envs > 1 else 0,
        writer=RoundRobinWriter(track_generations=True),
        sampler=replay_sampler,
        batch_size=cfg.replay_buffer.batch_size * (cfg.replay_buffer.seq_len + 1),
        generator=torch.Generator().manual_seed(0),
    )
    replay_record_builder = _DreamerV3ReplayRecordBuilder(num_envs)
    shifted_replay_writer = (
        _DreamerV3ShiftedReplayWriter(num_envs) if count_reset_records else None
    )
    replay_pipeline = _DreamerV3ReplayPipeline()

    action_step = 0
    # JAX's driver emits one initial reset observation from every worker before
    # the first control transition. It counts those records on the curve axis.
    record_step = num_envs if count_reset_records else 0
    environment_step = record_step if count_reset_records else action_step
    update_step = 0
    running_training_return = torch.zeros(num_envs)
    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    history_train_steps: list[int] = []
    history_train_returns: list[float] = []
    loss_history: list[torch.Tensor] = []
    record_loss_history = bool(cfg.logger.output_plot and _has_matplotlib)
    next_eval = 0
    next_train_log = 0
    # Anchors for the interval-rate metrics, matching the reference's
    # elements.FPS, which reports counts over the wall-clock time since its
    # previous read rather than since the start of the run.
    last_log_seconds = run_timer.elapsed()
    last_log_updates = 0

    eval_env = TransformedEnv(
        make_env(cfg, cfg.env.seed + 100),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
            previous_action=Unbounded(action_dim),
        ),
    )

    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )
    warmup = max(warmup, num_envs * (cfg.replay_buffer.seq_len + 1))

    updates_per_batch = cfg.optimization.updates_per_batch
    if cfg.optimization.train_ratio is not None:
        updates_per_batch = max(
            1,
            round(
                cfg.collector.frames_per_batch
                * cfg.optimization.train_ratio
                / (cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len)
            ),
        )

    use_bfloat16 = cfg.optimization.mixed_precision and device.type == "cuda"
    learner_heads_compiled = not cfg.optimization.compile_learner_heads
    first_eager_update_completed = False
    cudagraph_learner = None
    first_training_batch = True

    def forward_backward(
        sample: TensorDictBase, set_to_none: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            model_loss_td, model_out = model_loss(sample)
            model_kl = (
                model_loss_td["loss_model_dynamic"]
                + model_loss_td["loss_model_representation"]
            )
            total_model_loss = (
                model_kl
                + model_loss_td["loss_model_reco"]
                + model_loss_td["loss_model_reward"]
                + model_loss_td["loss_model_continue"]
            ).squeeze()

            post_state = (
                model_out.get(("next", "state")).detach().reshape(-1, state_dim)
            )
            post_belief = (
                model_out.get(("next", "belief"))
                .detach()
                .reshape(-1, cfg.networks.rnn_hidden_dim)
            )
            actor_input = TensorDict(
                {"state": post_state, "belief": post_belief},
                [post_state.shape[0]],
            )
            actor_loss_td, fake_data = actor_loss(actor_input)
            value_loss_td, _ = value_loss(
                fake_data
                if cfg.optimization.shared_imagination_value
                else fake_data.detach()
            )

            replay_features = TensorDict(
                {
                    "state": model_out.get(("next", "state")),
                    "belief": model_out.get(("next", "belief")),
                },
                sample.batch_size,
            )
            bootstrap = fake_data.get("lambda_target")[..., 0, 0].reshape(
                sample.batch_size
            )
            replay_loss = value_loss.replay_value_loss(
                replay_features,
                sample.get(("next", "reward")),
                sample.get(("next", "done")),
                sample.get(("next", "terminated")),
                bootstrap,
                horizon=cfg.optimization.continuation_horizon,
                lmbda=cfg.optimization.lmbda,
            )
            total_loss = (
                total_model_loss
                + actor_loss_td["loss_actor"]
                + value_loss_td["loss_value"]
                + cfg.optimization.replay_value_loss_weight * replay_loss
            )

        optimizer.zero_grad(set_to_none=set_to_none)
        total_loss.backward()
        metrics = torch.stack(
            (
                model_kl.detach().reshape(()),
                model_loss_td["loss_model_reco"].detach().reshape(()),
                model_loss_td["loss_model_reward"].detach().reshape(()),
                actor_loss_td["loss_actor"].detach().reshape(()),
                value_loss_td["loss_value"].detach().reshape(()),
                replay_loss.detach().reshape(()),
            )
        )
        return (
            metrics,
            model_out.get(("next", "state")).detach(),
            model_out.get(("next", "belief")).detach(),
        )

    def train_step(
        sample: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal cudagraph_learner
        nonlocal first_eager_update_completed
        nonlocal learner_heads_compiled
        if use_cudagraph_learner and first_eager_update_completed:
            if cudagraph_learner is None:
                cudagraph_learner = _DreamerV3CudaGraphLearner(
                    forward_backward,
                    optimizer,
                    value_target_updater,
                    trainable_parameters,
                    tuple(value_loss.target_value_model_params.values(True, True)),
                    (actor_loss.return_low, actor_loss.return_high),
                )
            return cudagraph_learner(sample)

        outputs = forward_backward(sample, True)
        optimizer.step()
        value_target_updater.step()
        if not learner_heads_compiled:
            # Compile only after both recurrent scans have materialized their
            # fixed-shape graphs. Wrapping shared heads before that point makes
            # Dynamo trace the compile wrappers into the unrolled recurrences.
            _compile_dreamer_v3_learner_heads(
                world_model,
                reward_net,
                continuation_net,
                actor_model,
            )
            learner_heads_compiled = True
        first_eager_update_completed = True
        return outputs

    if cfg.optimization.separate_policy_rng:
        # Model construction and policy inference use different random streams
        # in JAX. Start the learner stream independently after initialization.
        torch.manual_seed(_jax_torch_seed(cfg.env.seed, 0))

    for data in collector:
        # Collector data is yielded after its action was computed. Applying the
        # pending snapshot here therefore matches JAX's apply-after-action sync.
        if behavior_policy_sync is not None:
            behavior_policy_sync.apply_after_action()
        batch_start_action_step = action_step
        completed_episodes = _training_episode_returns(
            data, running_training_return, num_envs
        )
        for time_index, env_index, score in completed_episodes:
            if count_reset_records:
                action_index = batch_start_action_step // num_envs + time_index + 1
                episode_step = _driver_step_for_action(
                    action_index,
                    env_index,
                    num_envs,
                    cfg.env.max_episode_steps,
                )
            else:
                episode_step = (
                    batch_start_action_step + time_index * num_envs + env_index + 1
                )
            history_train_steps.append(episode_step)
            history_train_returns.append(score)
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "type": "train_episode",
                    "environment_steps": episode_step,
                    "action_steps": batch_start_action_step
                    + (time_index + 1) * num_envs,
                    "score": score,
                    "elapsed_seconds": run_timer.elapsed(),
                },
            )
        replay_data = replay_record_builder(data)
        with timeit("dreamer_v3/replay_extend"):
            if shifted_replay_writer is not None:
                shifted_replay_writer.extend(rb, replay_sampler, replay_data)
            else:
                replay_indices = rb.extend(
                    replay_data if num_envs > 1 else replay_data.reshape(-1)
                )
                replay_sampler.observe_extend(replay_indices, rb.storage)
        if (
            not replay_pipeline.has_prefetched
            and update_step == 0
            and len(rb) >= num_envs * (cfg.replay_buffer.seq_len + 1)
        ):
            # JAX starts a one-batch replay prefetch as soon as the first item
            # exists, well before the learner warmup gate. Cache the equivalent
            # initial batch instead of sampling it only when training begins.
            with timeit("dreamer_v3/replay_sample"):
                replay_pipeline.prefetch(rb)
        action_step += data.numel()
        record_step += replay_data.numel() if count_reset_records else data.numel()
        environment_step = record_step if count_reset_records else action_step

        if len(rb) < warmup:
            continue

        if behavior_policy_sync is not None:
            # JAX stages the pre-update policy tree only once while a snapshot
            # is pending, even if several learner updates follow this record.
            behavior_policy_sync.stage_before_training()

        first_eligible_record = first_training_batch and count_reset_records
        batch_updates = _learner_updates_for_records(
            replay_data.numel(),
            cfg.optimization.train_ratio,
            cfg.replay_buffer.batch_size,
            cfg.replay_buffer.seq_len,
            updates_per_batch,
            first_eligible_record=first_eligible_record,
        )
        if first_eligible_record:
            first_training_batch = False
        batch_losses = torch.empty(
            (batch_updates, 6) if record_loss_history else (6,),
            device=device,
        )
        for update_index in range(batch_updates):
            with timeit("dreamer_v3/replay_sample"):
                replay_sample, sample_info = replay_pipeline.take(rb)
                replay_sample = replay_sample.reshape(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len + 1,
                )
                sample = replay_sample[:, :-1].to(device)
            with timeit("dreamer_v3/replay_update"):
                # The successor is already sampled, so applying the older
                # refresh here preserves JAX's one-sample-ahead/one-output-
                # behind visibility while consuming compiled RSSM outputs
                # before their CUDA-graph buffers are reused.
                replay_pipeline.apply_pending_context(rb)
            with timeit("dreamer_v3/train_update"):
                (
                    update_losses,
                    refreshed_state,
                    refreshed_belief,
                ) = train_step(sample)
                if record_loss_history:
                    batch_losses[update_index].copy_(update_losses)
                else:
                    batch_losses.copy_(update_losses)
            with timeit("dreamer_v3/replay_update"):
                replay_pipeline.stage_context(
                    sample_info,
                    refreshed_state,
                    refreshed_belief,
                )
            update_step += 1

        if record_loss_history:
            loss_history.append(batch_losses.detach().cpu())

        train_log_due = bool(
            metrics_jsonl_path is not None
            and cfg.logger.train_every
            and (
                environment_step >= next_train_log
                or action_step >= collector_action_frames
            )
        )
        eval_due = bool(cfg.logger.eval_every and environment_step >= next_eval)
        latest_losses = (
            (batch_losses[-1] if record_loss_history else batch_losses).detach().cpu()
            if train_log_due or eval_due
            else None
        )
        if train_log_due:
            # Read the clock before the optional diagnostics pass: it runs a
            # full world-model and imagination forward, and charging that to the
            # interval would deflate the very rate it is logged next to.
            elapsed_seconds = run_timer.elapsed()
            diagnostics = (
                _reference_diagnostics(
                    model_loss=model_loss,
                    actor_loss=actor_loss,
                    value_loss=value_loss,
                    sample=sample,
                    state_dim=state_dim,
                    rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
                    use_bfloat16=use_bfloat16,
                    device=device,
                )
                if cfg.logger.get("diagnostics", False)
                else {}
            )
            total_timings = timeit.todict(percall=False, prefix="time")
            batch_elements = cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len
            # ``training_fps`` mirrors the reference's ``fps/train``: batch
            # elements over wall-clock seconds since the previous log, so the
            # two runs' numbers can be compared directly. ``learner_fps``
            # divides by optimisation time alone, which isolates the learner
            # from collection but is NOT comparable with the reference.
            log_seconds = elapsed_seconds - last_log_seconds
            training_fps = (
                (update_step - last_log_updates) * batch_elements / log_seconds
                if log_seconds > 0
                else 0.0
            )
            # Restart the interval after the diagnostics pass rather than at
            # ``elapsed_seconds``, so its cost lands in neither interval. The
            # reference has no counterpart to it.
            last_log_seconds = run_timer.elapsed()
            last_log_updates = update_step
            train_seconds = total_timings.get("time/dreamer_v3/train_update", 0.0)
            learner_fps = (
                update_step * batch_elements / train_seconds if train_seconds else 0.0
            )
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "type": "train",
                    "environment_steps": environment_step,
                    "record_steps": record_step,
                    "action_steps": action_step,
                    "updates": update_step,
                    "loss_dynamic_representation": latest_losses[0].item(),
                    "loss_reconstruction": latest_losses[1].item(),
                    "loss_reward": latest_losses[2].item(),
                    "loss_actor": latest_losses[3].item(),
                    "loss_value": latest_losses[4].item(),
                    "loss_replay_value": latest_losses[5].item(),
                    "training_fps": training_fps,
                    "learner_fps": learner_fps,
                    "elapsed_seconds": elapsed_seconds,
                    "bfloat16": use_bfloat16,
                    "compiled_rssm": bool(cfg.optimization.compile_rssm),
                    "compiled_imagination": bool(cfg.optimization.compile_imagination),
                    "compiled_learner_heads": bool(
                        cfg.optimization.compile_learner_heads
                    ),
                    "compiled_replay_value_loss": bool(
                        cfg.optimization.compile_replay_value_loss
                    ),
                    "shared_imagination_value": bool(
                        cfg.optimization.shared_imagination_value
                    ),
                    **diagnostics,
                    **total_timings,
                },
            )
            next_train_log = environment_step + cfg.logger.train_every

        if eval_due:
            with timeit("dreamer_v3/evaluation"):
                r = eval_episode_reward(
                    eval_env,
                    real_world_actor,
                    cfg.logger.eval_episodes,
                    cfg.env.max_episode_steps,
                )
            history_steps.append(environment_step)
            history_eval.append(r)
            torchrl_logger.info(
                "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
                environment_step,
                r.item(),
                latest_losses[0].item(),
                latest_losses[1].item(),
                latest_losses[2].item(),
                latest_losses[3].item(),
            )
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "type": "evaluation",
                    "environment_steps": environment_step,
                    "record_steps": record_step,
                    "action_steps": action_step,
                    "return": r.item(),
                    "episodes": cfg.logger.eval_episodes,
                    "elapsed_seconds": run_timer.elapsed(),
                },
            )
            next_eval = environment_step + cfg.logger.eval_every

    if cfg.logger.output_plot and _has_matplotlib:
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dep)

        eval_steps = history_steps
        eval_rewards = torch.stack(history_eval).cpu().numpy() if history_eval else []
        loss_curves = torch.cat(loss_history).numpy() if loss_history else None
        kl_vals = loss_curves[:, 0] if loss_curves is not None else []
        reco_vals = loss_curves[:, 1] if loss_curves is not None else []
        reward_vals = loss_curves[:, 2] if loss_curves is not None else []

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(eval_steps, eval_rewards, marker="o")
        axes[0].set_title(f"{cfg.env.name} eval reward (real env)")
        axes[0].set_xlabel("env_step")
        axes[0].set_ylabel("avg episode return")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(reco_vals, label="reco", alpha=0.8)
        axes[1].plot(reward_vals, label="reward", alpha=0.8)
        axes[1].plot(kl_vals, label="kl", alpha=0.8)
        axes[1].set_title("World-model losses (update step)")
        axes[1].set_xlabel("update step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps, "
            f"{updates_per_batch} updates/batch"
        )
        fig.tight_layout()
        fig.savefig(cfg.logger.output_plot, dpi=120)
        torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
    elif cfg.logger.output_plot:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )

    if cfg.logger.metrics_json:
        metrics_path = Path(cfg.logger.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "backend": cfg.env.backend,
                    "environment": cfg.env.name,
                    "task": cfg.env.task,
                    "seed": cfg.env.seed,
                    "environment_seeded": bool(cfg.env.use_seed),
                    "total_environment_steps": environment_step,
                    "total_record_steps": record_step,
                    "total_action_steps": action_step,
                    "environment_steps": history_steps,
                    "evaluation_returns": [value.item() for value in history_eval],
                    "training_episode_steps": history_train_steps,
                    "training_episode_returns": history_train_returns,
                    "updates": update_step,
                    "elapsed_seconds": run_timer.elapsed(),
                    "timings": timeit.todict(percall=False),
                },
                indent=2,
            )
            + "\n"
        )
        torchrl_logger.info("Saved evaluation metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
