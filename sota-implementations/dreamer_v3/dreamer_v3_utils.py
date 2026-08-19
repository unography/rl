# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for the DreamerV3 example.

Environment and model builders, plus the machinery reproducing the reference's
replay, policy and optimizer behaviour. See ``dreamer_v3.py`` for the run loop.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TypeAlias

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
from tensordict.utils import unravel_key

from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import StepCounter, TransformedEnv
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
from torchrl.objectives import symexp, symlog

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


def _imagined_values(value_loss, fake_data: TensorDictBase):
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


def _full_horizon_weight(actor_loss, fake_data: TensorDictBase) -> torch.Tensor:
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


_LEARNER_RNG_STREAM = 1
_REPLAY_RNG_STREAM = 2


def _jax_torch_seed(seed: int, counter: int, stream: int = 0) -> int:
    """Map JAX's two-word per-call seed to one deterministic Torch seed.

    ``stream`` separates independent consumers. JAX derives the learner and the
    policy from different key paths, so they must not share a seed.
    """
    rng = np.random.default_rng(seed=[seed, counter, stream])
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


class _DreamerV3UpdateRatio:
    """JAX's ``elements.when.Ratio``: truncate, and carry the remainder.

    Call with the cumulative driver-record count, as the reference calls its
    scheduler with the global step. The first call, which happens once the
    replay gate opens, returns a single update from the reference's
    uninitialized branch.

    Args:
        ratio (float): Learner updates per driver record.
    """

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self._previous: float | None = None

    def __call__(self, record_count: int) -> int:
        if self.ratio <= 0:
            return 0
        if self._previous is None:
            self._previous = float(record_count)
            return 1
        repeats = int((record_count - self._previous) * self.ratio)
        self._previous += repeats / self.ratio
        return repeats


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
    # Overlapping slices are common and a CUDA advanced-index write has no
    # defined duplicate winner, so keep each coordinate's last occurrence,
    # matching JAX's batch-order application.
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

    After ``N`` learner updates the reference has sampled ``N + 1`` batches and
    applied ``N - 1`` latent refreshes.
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

        Sampling the successor first preserves the reference pipeline: the
        refresh remains invisible to both the current and prefetched batches
        and can first affect the following sample.
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
        self._started = False

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        if self.num_streams == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2 or data.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected collector data with shape [num_streams, time], got "
                f"{tuple(data.shape)} for {self.num_streams} streams."
            )

        records = []
        record_keys = (
            "action",
            "is_init",
            "state",
            "belief",
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
        )
        for time_index in range(data.shape[1]):
            collector_step = data[:, time_index]
            reset = collector_step.get("is_init").reshape(self.num_streams, -1).any(-1)
            insert_reset = reset if self._started else torch.zeros_like(reset)
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
            self._started = True

        return torch.stack(records, 1)


class _DreamerV3ShiftedReplayWriter:
    """Materialize JAX's initial record and mutable final context slot.

    Collector transitions already carry the shifted representation the Torch
    learner consumes. JAX replay additionally holds the initial reset record
    and the newest record, which has no outgoing transition yet: that one is a
    generation-tracked tail slot, finalized by the next collector transition
    without changing its replay identity.
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

    def _drop_stale_online(self, storage: LazyTensorStorage, seq_length: int) -> None:
        """Discard queued starts that the ring buffer has overwritten.

        The reference re-draws when an online chunk was evicted. A queued start
        stays usable while its offset from the oldest live record still leaves
        room for a full window.
        """
        if not self._online_queue or not storage._is_full:
            return
        stored_time = storage.shape[0]
        oldest = (int(storage._last_cursor_index) + 1) % stored_time
        live = stored_time - seq_length + 1
        self._online_queue = [
            start
            for start in self._online_queue
            if (int(start[0]) - oldest) % stored_time < live
        ]

    def sample(self, storage, batch_size: int):
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        self._drop_stale_online(storage, seq_length)
        # The reference draws each sequence of a batch separately and takes an
        # online block whenever the queue holds one, so a batch drains up to
        # num_slices of them.
        num_online = min(num_slices, len(self._online_queue))
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
        return coordinates.reshape(-1, storage.ndim).unbind(-1), {}

    def _empty(self) -> None:
        super()._empty()
        self._stream_lengths = None
        self._online_queue.clear()


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

        # Pass the seed at construction: set_seed() performs a hidden reset,
        # which shifts the task stream. ``env.use_seed=false`` reproduces the
        # reference, whose DMC preset calls suite.load() with no random state.
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
    if cfg.optimization.compile_rssm:
        rollout.compile_rollout(cfg.optimization.compile_rssm)

    decoder_event_dims = tuple(cfg.networks.decoder_event_dims or (obs_dim,))
    if sum(decoder_event_dims) != obs_dim:
        raise ValueError(
            "Decoder event dimensions must sum to the flattened observation "
            f"size, got {decoder_event_dims} for {obs_dim}."
        )
    # One head per observation event, as the reference's DictHead builds: AGC
    # clips each head kernel separately, so a merged kernel would clip
    # differently. The heads predict in symlog space; symexp back in FP32 so
    # the loss's symlog(prediction) recovers the raw head output.
    decoder = TensorDictSequential(
        TensorDictModule(
            _DreamerV3Decoder(
                cfg,
                state_dim + cfg.networks.rnn_hidden_dim,
                decoder_event_dims,
            ),
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reco_symlog_observation")],
        ),
        TensorDictModule(
            _to_float,
            in_keys=[("next", "reco_symlog_observation")],
            out_keys=[("next", "reco_symlog_observation")],
        ),
        TensorDictModule(
            symexp,
            in_keys=[("next", "reco_symlog_observation")],
            out_keys=[("next", "reco_pixels")],
        ),
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


def build_imagination_model(
    *, prior_net, reward_net, reward_decoder, compile_prior: bool = False
):
    """Build imagination operators backed by the trained world-model heads.

    ``compile_prior`` compiles the prior for imagination only, leaving the
    rollout and the acting policy on the eager module they share.
    """
    transition_model = TensorDictSequential(
        TensorDictModule(
            torch.compile(prior_net, dynamic=False) if compile_prior else prior_net,
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
