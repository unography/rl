# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import runpy
from pathlib import Path
from typing import Literal

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl import timeit
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.modules import DreamerV3MLP, SymExpTwoHot
from torchrl.modules.models.model_based_v3 import (
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.dreamer_v3 import _DreamerV3ImaginationRollout
from torchrl.objectives.utils import SoftUpdate, ValueEstimators


RolloutPath = Literal["generic", "tensor", "compiled"]
ImaginationPath = Literal["tensor", "compiled"]
ReplayPath = Literal["host", "device"]
LearnerHeadPath = Literal["eager", "compiled"]
ValuePath = Literal["repeated", "shared"]
ReplayValuePath = Literal["eager", "compiled"]
PriorPath = Literal["recurrence", "full"]
MetricPath = Literal["latest", "all"]
FullLearnerPath = Literal["eager", "outer_graph"]

_BATCH_SIZE = 16
_TIME_STEPS = 64
_ACTION_DIM = 6
_EMBEDDING_DIM = 64
_BELIEF_DIM = 512
_HIDDEN_DIM = 64
_NUM_CATEGORICALS = 32
_NUM_CLASSES = 4
_IMAGINATION_HORIZON = 15
_NUM_REWARD_BINS = 255
_SOTA_EXAMPLE = None


def _get_sota_example():
    global _SOTA_EXAMPLE
    if _SOTA_EXAMPLE is None:
        repo_root = Path(__file__).parents[1]
        _SOTA_EXAMPLE = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_replay_benchmark",
        )
    return _SOTA_EXAMPLE


class _ImaginationActor(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
        self.backbone = DreamerV3MLP(
            state_dim + _BELIEF_DIM,
            2 * _ACTION_DIM,
            depth=3,
            num_cells=_HIDDEN_DIM,
            outscale=0.01,
            device=device,
        )

    def forward(
        self, state: torch.Tensor, belief: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        location, scale = self.backbone(belief, state).chunk(2, -1)
        location = location.tanh()
        scale = 0.9 * torch.sigmoid(scale + 2) + 0.1
        return location, scale


class _LearnerHeads(torch.nn.Module):
    """Stable-shape MLPs used outside the compiled recurrent scans."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
        features = state_dim + _BELIEF_DIM
        self.encoder = DreamerV3MLP(
            24, _EMBEDDING_DIM, depth=3, num_cells=_HIDDEN_DIM, device=device
        )
        self.decoder = DreamerV3MLP(
            features, 24, depth=3, num_cells=_HIDDEN_DIM, device=device
        )
        self.reward = DreamerV3MLP(
            features,
            _NUM_REWARD_BINS,
            depth=1,
            num_cells=_HIDDEN_DIM,
            outscale=0.0,
            device=device,
        )
        self.continuation = DreamerV3MLP(
            features, 1, depth=1, num_cells=_HIDDEN_DIM, device=device
        )
        self.actor = _ImaginationActor(device)

    def compile_heads(self) -> None:
        """Compile each MLP separately, matching the learner setup."""
        for module in (
            self.encoder,
            self.decoder,
            self.reward,
            self.continuation,
            self.actor,
        ):
            module.compile(
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

    def forward(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        belief: torch.Tensor,
        imagined_state: torch.Tensor,
        imagined_belief: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(observation)
        decoded = self.decoder(state, belief)
        reward_logits = self.reward(belief, state).float()
        continuation_logits = self.continuation(imagined_belief, imagined_state).float()
        location, scale = self.actor(imagined_state, imagined_belief)
        return (
            encoded.square().mean()
            + decoded.square().mean()
            + reward_logits.square().mean()
            + continuation_logits.square().mean()
            + location.float().square().mean()
            + scale.float().square().mean()
        )


class _ValueHeads(torch.nn.Module):
    """Online and slow value heads at the imagination training shape."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
        features = state_dim + _BELIEF_DIM
        self.online = DreamerV3MLP(
            features,
            _NUM_REWARD_BINS,
            depth=3,
            num_cells=_HIDDEN_DIM,
            outscale=0.0,
            device=device,
        )
        self.slow = DreamerV3MLP(
            features,
            _NUM_REWARD_BINS,
            depth=3,
            num_cells=_HIDDEN_DIM,
            outscale=0.0,
            device=device,
        ).requires_grad_(False)
        self.decoder = SymExpTwoHot(_NUM_REWARD_BINS).to(device)

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        next_state: torch.Tensor,
        next_belief: torch.Tensor,
        path: ValuePath,
    ) -> torch.Tensor:
        if path == "repeated":
            with torch.no_grad():
                next_value = self.decoder(self.online(next_belief, next_state).float())
                baseline = self.decoder(self.online(belief, state).float())
                slow_value = self.decoder(self.slow(belief, state).float())
            value_logits = self.online(belief, state).float()
        else:
            all_state = torch.cat((state[..., :1, :], next_state), -2)
            all_belief = torch.cat((belief[..., :1, :], next_belief), -2)
            all_value_logits = self.online(all_belief, all_state).float()
            all_value = self.decoder(all_value_logits)
            value_logits = all_value_logits[..., :-1, :]
            next_value = all_value[..., 1:, :].detach()
            baseline = all_value[..., :-1, :].detach()
            with torch.no_grad():
                slow_value = self.decoder(self.slow(all_belief, all_state).float())[
                    ..., :-1, :
                ]
        target_statistic = (
            next_value.square().mean()
            + baseline.square().mean()
            + slow_value.square().mean()
        )
        return value_logits.square().mean() + target_statistic


class _ReplayValueHead(torch.nn.Module):
    """Distribution-valued critic used by the replay-loss benchmark."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
        self.network = DreamerV3MLP(
            state_dim + _BELIEF_DIM,
            _NUM_REWARD_BINS,
            depth=3,
            num_cells=_HIDDEN_DIM,
            outscale=0.0,
            device=device,
        )
        self.decoder = SymExpTwoHot(_NUM_REWARD_BINS).to(device)

    def forward(
        self, belief: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(belief, state).float()
        return logits, self.decoder(logits)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_rollout(device: torch.device, path: RolloutPath) -> RSSMRolloutV3:
    torch.manual_seed(0)
    prior = TensorDictModule(
        RSSMPriorV3(
            action_shape=torch.Size([_ACTION_DIM]),
            hidden_dim=_HIDDEN_DIM,
            rnn_hidden_dim=_BELIEF_DIM,
            num_categoricals=_NUM_CATEGORICALS,
            num_classes=_NUM_CLASSES,
            action_dim=_ACTION_DIM,
            device=device,
            recurrent_model="block_gru",
            num_blocks=8,
            num_layers=1,
            prior_num_layers=2,
            unimix=0.01,
        ),
        in_keys=["state", "belief", "action"],
        out_keys=[
            ("next", "prior_logits"),
            ("next", "state"),
            ("next", "belief"),
        ],
    )
    posterior = TensorDictModule(
        RSSMPosteriorV3(
            hidden_dim=_HIDDEN_DIM,
            num_categoricals=_NUM_CATEGORICALS,
            num_classes=_NUM_CLASSES,
            rnn_hidden_dim=_BELIEF_DIM,
            obs_embed_dim=_EMBEDDING_DIM,
            device=device,
            use_rms_norm=True,
            num_layers=1,
            unimix=0.01,
        ),
        in_keys=[("next", "belief"), ("next", "encoded_latents")],
        out_keys=[("next", "posterior_logits"), ("next", "state")],
    )
    rollout = RSSMRolloutV3(prior, posterior).to(device)
    if path == "generic":
        rollout._fast_path = False
    elif path == "compiled":
        rollout.compile_scan(mode="reduce-overhead", fullgraph=True)
    return rollout


def _make_tensordict(device: torch.device) -> TensorDict:
    generator = torch.Generator(device=device).manual_seed(1)
    state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
    reset = torch.zeros(_BATCH_SIZE, _TIME_STEPS, 1, dtype=torch.bool, device=device)
    reset[0, _TIME_STEPS // 2] = True
    return TensorDict(
        {
            "state": torch.zeros(_BATCH_SIZE, _TIME_STEPS, state_dim, device=device),
            "belief": torch.zeros(_BATCH_SIZE, _TIME_STEPS, _BELIEF_DIM, device=device),
            "action": torch.randn(
                _BATCH_SIZE,
                _TIME_STEPS,
                _ACTION_DIM,
                device=device,
                generator=generator,
            ),
            "is_init": reset,
            "next": {
                "encoded_latents": torch.randn(
                    _BATCH_SIZE,
                    _TIME_STEPS,
                    _EMBEDDING_DIM,
                    device=device,
                    generator=generator,
                )
            },
        },
        [_BATCH_SIZE, _TIME_STEPS],
        device=device,
    )


def _call(
    rollout: RSSMRolloutV3,
    tensordict: TensorDict,
    device: torch.device,
) -> TensorDict:
    with torch.inference_mode():
        output = rollout(tensordict)
    _sync(device)
    return output


def _make_imagination_rollout(
    device: torch.device, path: ImaginationPath
) -> _DreamerV3ImaginationRollout:
    torch.manual_seed(0)
    state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
    rollout = _DreamerV3ImaginationRollout(
        prior_model=RSSMPriorV3(
            action_shape=torch.Size([_ACTION_DIM]),
            hidden_dim=_HIDDEN_DIM,
            rnn_hidden_dim=_BELIEF_DIM,
            num_categoricals=_NUM_CATEGORICALS,
            num_classes=_NUM_CLASSES,
            action_dim=_ACTION_DIM,
            device=device,
            recurrent_model="block_gru",
            num_blocks=8,
            num_layers=1,
            prior_num_layers=2,
            unimix=0.01,
        ),
        actor_model=_ImaginationActor(device),
        reward_model=DreamerV3MLP(
            state_dim + _BELIEF_DIM,
            _NUM_REWARD_BINS,
            depth=1,
            num_cells=_HIDDEN_DIM,
            outscale=0.0,
            device=device,
        ),
        reward_decoder=SymExpTwoHot(_NUM_REWARD_BINS).to(device),
        horizon=_IMAGINATION_HORIZON,
    ).to(device)
    if path == "compiled":
        rollout.compile_scan(mode="reduce-overhead", fullgraph=True)
    return rollout


def _make_imagination_tensordict(device: torch.device) -> TensorDict:
    generator = torch.Generator(device=device).manual_seed(1)
    starts = _BATCH_SIZE * _TIME_STEPS
    state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
    return TensorDict(
        {
            "state": torch.randn(starts, state_dim, device=device, generator=generator),
            "belief": torch.randn(
                starts, _BELIEF_DIM, device=device, generator=generator
            ),
        },
        [starts],
        device=device,
    )


def _call_imagination(
    rollout: _DreamerV3ImaginationRollout,
    tensordict: TensorDict,
    device: torch.device,
) -> TensorDict:
    with torch.inference_mode():
        output = rollout(tensordict)
    _sync(device)
    return output


def _make_learner_head_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1)
    state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
    return (
        torch.randn(
            _BATCH_SIZE,
            _TIME_STEPS,
            24,
            device=device,
            generator=generator,
        ),
        torch.randn(
            _BATCH_SIZE,
            _TIME_STEPS,
            state_dim,
            device=device,
            generator=generator,
        ),
        torch.randn(
            _BATCH_SIZE,
            _TIME_STEPS,
            _BELIEF_DIM,
            device=device,
            generator=generator,
        ),
        torch.randn(
            _BATCH_SIZE * _TIME_STEPS,
            _IMAGINATION_HORIZON,
            state_dim,
            device=device,
            generator=generator,
        ),
        torch.randn(
            _BATCH_SIZE * _TIME_STEPS,
            _IMAGINATION_HORIZON,
            _BELIEF_DIM,
            device=device,
            generator=generator,
        ),
    )


def _call_learner_heads(
    heads: _LearnerHeads,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    heads.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = heads(*inputs)
    loss.backward()
    _sync(device)


def _make_value_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(2)
    starts = _BATCH_SIZE * _TIME_STEPS
    state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
    return (
        torch.randn(
            starts,
            _IMAGINATION_HORIZON,
            state_dim,
            device=device,
            generator=generator,
        ),
        torch.randn(
            starts,
            _IMAGINATION_HORIZON,
            _BELIEF_DIM,
            device=device,
            generator=generator,
        ),
        torch.randn(
            starts,
            _IMAGINATION_HORIZON,
            state_dim,
            device=device,
            generator=generator,
        ),
        torch.randn(
            starts,
            _IMAGINATION_HORIZON,
            _BELIEF_DIM,
            device=device,
            generator=generator,
        ),
    )


def _call_value_heads(
    heads: _ValueHeads,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    path: ValuePath,
    device: torch.device,
) -> None:
    heads.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = heads(*inputs, path)
    loss.backward()
    _sync(device)


def _make_replay_value_loss(
    device: torch.device, path: ReplayValuePath
) -> tuple[DreamerV3ValueLoss, TensorDict, tuple[torch.Tensor, ...]]:
    generator = torch.Generator(device=device).manual_seed(3)
    state = torch.randn(
        _BATCH_SIZE,
        _TIME_STEPS,
        _NUM_CATEGORICALS * _NUM_CLASSES,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    belief = torch.randn(
        _BATCH_SIZE,
        _TIME_STEPS,
        _BELIEF_DIM,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    features = TensorDict(
        {"state": state, "belief": belief},
        [_BATCH_SIZE, _TIME_STEPS],
        device=device,
    )
    value_model = TensorDictModule(
        _ReplayValueHead(device),
        in_keys=["belief", "state"],
        out_keys=["state_value_logits", "state_value"],
    )
    loss_module = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=_NUM_REWARD_BINS,
        slow_critic_regularization=1.0,
    ).to(device)
    if path == "compiled":
        loss_module.compile_replay_value_loss(fullgraph=True)
    reward = torch.randn(
        _BATCH_SIZE,
        _TIME_STEPS,
        1,
        device=device,
        generator=generator,
    )
    done = torch.zeros(_BATCH_SIZE, _TIME_STEPS, 1, dtype=torch.bool, device=device)
    terminated = torch.zeros_like(done)
    bootstrap = torch.randn(
        _BATCH_SIZE,
        _TIME_STEPS,
        device=device,
        generator=generator,
    )
    return loss_module, features, (reward, done, terminated, bootstrap)


def _call_replay_value_loss(
    loss_module: DreamerV3ValueLoss,
    features: TensorDict,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
) -> None:
    loss_module.zero_grad(set_to_none=True)
    features["state"].grad = None
    features["belief"].grad = None
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = loss_module.replay_value_loss(features, *inputs)
    loss.backward()
    _sync(device)


def _make_policy_prior(
    device: torch.device,
) -> tuple[RSSMPriorV3, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    prior = RSSMPriorV3(
        action_shape=(_ACTION_DIM,),
        hidden_dim=_HIDDEN_DIM,
        rnn_hidden_dim=_BELIEF_DIM,
        num_categoricals=_NUM_CATEGORICALS,
        num_classes=_NUM_CLASSES,
        action_dim=_ACTION_DIM,
        recurrent_model="block_gru",
        num_blocks=8,
        device=device,
    )
    inputs = (
        torch.randn(_BATCH_SIZE, _NUM_CATEGORICALS * _NUM_CLASSES, device=device),
        torch.randn(_BATCH_SIZE, _BELIEF_DIM, device=device),
        torch.randn(_BATCH_SIZE, _ACTION_DIM, device=device),
    )
    return prior, inputs


@torch.inference_mode()
def _call_policy_prior(
    prior: RSSMPriorV3,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    path: PriorPath,
    device: torch.device,
) -> None:
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        if path == "recurrence":
            prior._update_belief(*inputs)
        else:
            prior._belief_and_logits(*inputs)
    _sync(device)


def _call_metric_collection(
    metrics: tuple[torch.Tensor, ...],
    path: MetricPath,
    device: torch.device,
) -> None:
    output = torch.empty((len(metrics), 6) if path == "all" else (6,), device=device)
    for index, metric in enumerate(metrics):
        if path == "all":
            output[index].copy_(metric)
        else:
            output.copy_(metric)
    _sync(device)


def _make_replay_buffer(device: torch.device, path: ReplayPath) -> ReplayBuffer:
    storage_device = torch.device("cpu") if path == "host" else device
    sampler = _get_sota_example()["_DreamerV3ReplaySampler"](
        slice_len=_TIME_STEPS + 1,
        traj_key=("collector", "replay_stream"),
        cache_values=True,
        online=True,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=65536,
            ndim=2,
            device=storage_device,
            consolidated=True,
        ),
        dim_extend=1,
        writer=RoundRobinWriter(track_generations=True),
        sampler=sampler,
        batch_size=_BATCH_SIZE * (_TIME_STEPS + 1),
    )
    num_envs = 16
    stored_steps = 256
    written = replay_buffer.extend(
        TensorDict(
            {
                "action": torch.randn(num_envs, stored_steps, _ACTION_DIM),
                "is_init": torch.zeros(num_envs, stored_steps, 1, dtype=torch.bool),
                "state": torch.randn(
                    num_envs,
                    stored_steps,
                    _NUM_CATEGORICALS * _NUM_CLASSES,
                ),
                "belief": torch.randn(num_envs, stored_steps, _BELIEF_DIM),
                ("collector", "traj_ids"): torch.arange(num_envs)[:, None].expand(
                    num_envs, stored_steps
                ),
                ("collector", "replay_stream"): torch.arange(num_envs)[:, None].expand(
                    num_envs, stored_steps
                ),
                ("next", "observation"): torch.randn(num_envs, stored_steps, 24),
                ("next", "reward"): torch.randn(num_envs, stored_steps, 1),
                ("next", "done"): torch.zeros(
                    num_envs, stored_steps, 1, dtype=torch.bool
                ),
                ("next", "terminated"): torch.zeros(
                    num_envs, stored_steps, 1, dtype=torch.bool
                ),
            },
            [num_envs, stored_steps],
        )
    )
    sampler.observe_extend(written, replay_buffer.storage)
    return replay_buffer


def _call_replay(replay_buffer: ReplayBuffer, device: torch.device) -> None:
    replay_sample, sample_info = replay_buffer.sample(return_info=True)
    replay_sample = replay_sample.reshape(_BATCH_SIZE, _TIME_STEPS + 1)
    replay_sample = replay_sample[:, :-1].to(device)

    sample_indices = sample_info["index"]
    if not isinstance(sample_indices, tuple):
        sample_indices = (sample_indices,)
    destination_indices = tuple(
        index.reshape(_BATCH_SIZE, _TIME_STEPS + 1)[:, 1:].reshape(-1)
        for index in sample_indices
    )
    destination_index = (
        torch.stack(destination_indices, -1)
        if len(destination_indices) > 1
        else destination_indices[0]
    )
    destination_generation = (
        sample_info["index_generation"]
        .reshape(_BATCH_SIZE, _TIME_STEPS + 1)[:, 1:]
        .reshape(-1)
    )
    replay_buffer.update_if_present(
        index=destination_index,
        generation=destination_generation,
        patch=replay_sample.select("state", "belief"),
    )
    _sync(device)


class _FullLearnerBenchmark:
    """DMC Walker learner update, including backward and eager parameter updates."""

    def __init__(self, device: torch.device, path: FullLearnerPath) -> None:
        example = _get_sota_example()
        root = Path(__file__).parents[1] / "sota-implementations/dreamer_v3"
        self.cfg = OmegaConf.merge(
            OmegaConf.load(root / "config.yaml"),
            OmegaConf.load(root / "config_dmc_walker.yaml"),
        )
        self.device = device
        self.path = path
        torch.manual_seed(0)
        (
            self.world_model,
            prior,
            reward_model,
            reward_decoder,
            continuation_model,
        ) = example["build_world_model"](
            cfg=self.cfg,
            obs_dim=24,
            action_dim=_ACTION_DIM,
        )
        self.world_model = self.world_model.to(device)
        scan_options = (
            {
                "fullgraph": True,
                "dynamic": False,
                "options": {"triton.cudagraphs": False},
            }
            if path == "outer_graph"
            else {"mode": "reduce-overhead", "fullgraph": True}
        )
        self.world_model[1].compile_scan(**scan_options)
        imagination_model = example["build_imagination_model"](
            prior_net=prior,
            reward_net=reward_model,
            reward_decoder=reward_decoder,
        ).to(device)
        continuation = example["build_continuation_model"](
            continuation_net=continuation_model
        ).to(device)
        self.actor_model = example["build_actor"](
            cfg=self.cfg, action_dim=_ACTION_DIM
        ).to(device)
        value_model = example["build_value"](cfg=self.cfg).to(device)
        imagination_rollout = _DreamerV3ImaginationRollout(
            prior_model=prior,
            actor_model=self.actor_model[0].module,
            reward_model=reward_model,
            reward_decoder=reward_decoder,
            horizon=_IMAGINATION_HORIZON,
        )
        imagination_rollout.compile_scan(**scan_options)
        real_env = example["make_env"](self.cfg, self.cfg.env.seed + 1)
        model_env = example["build_mb_env"](
            cfg=self.cfg,
            real_env=real_env,
            imagination_model=imagination_model,
            device=device,
        )
        real_env.close()
        self.model_loss = DreamerV3ModelLoss(
            self.world_model,
            num_reward_bins=_NUM_REWARD_BINS,
            free_bits=self.cfg.optimization.free_bits,
            kl_mode="separate",
            lambda_dynamic=self.cfg.optimization.dynamic_loss_weight,
            lambda_representation=self.cfg.optimization.representation_loss_weight,
            unimix=self.cfg.networks.unimix,
            lambda_continue=1.0,
            continue_target_scale=(1 - 1 / self.cfg.optimization.continuation_horizon),
            global_average=False,
            detach_output=False,
        ).to(device)
        self.model_loss.set_keys(pixels="observation")
        self.actor_loss = DreamerV3ActorLoss(
            self.actor_model,
            value_model,
            model_env,
            continuation_model=continuation,
            imagination_rollout=imagination_rollout,
            imagination_horizon=_IMAGINATION_HORIZON,
            use_reinforce=True,
            return_normalization_rate=(self.cfg.optimization.return_normalization_rate),
            return_normalization_min_scale=(
                self.cfg.optimization.return_normalization_min_scale
            ),
        )
        self.actor_loss.make_value_estimator(
            ValueEstimators.TDLambda,
            gamma=self.cfg.optimization.gamma,
            lmbda=self.cfg.optimization.lmbda,
        )
        self.actor_loss = self.actor_loss.to(device)
        self.value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            num_value_bins=_NUM_REWARD_BINS,
            actor_loss=self.actor_loss,
            slow_critic_regularization=(
                self.cfg.optimization.slow_critic_regularization
            ),
        ).to(device)
        self.value_loss.compile_replay_value_loss(fullgraph=True)
        self.actor_loss.set_shared_value_forward(self.value_loss)
        self.target_updater = SoftUpdate(
            self.value_loss, tau=self.cfg.optimization.slow_critic_tau
        )
        self.parameters = (
            list(self.world_model.parameters())
            + list(self.actor_model.parameters())
            + list(self.value_loss.parameters())
        )
        self.optimizer = example["_DreamerV3Optimizer"](
            self.parameters,
            lr=self.cfg.optimization.lr,
            agc=self.cfg.optimization.adaptive_grad_clip,
            beta1=0.9,
            beta2=0.999,
            eps=self.cfg.optimization.adam_eps,
            warmup_steps=self.cfg.optimization.warmup_steps,
        )
        self.sample = self._make_sample()
        self.graph_learner = None

        self._forward_backward_fn = self._build_forward_backward()
        # Match production: the first update materializes parameters and
        # optimizer state eagerly, then stable-shape heads are compiled.
        self._eager_step()
        example["_compile_dreamer_v3_learner_heads"](
            self.world_model,
            reward_model,
            continuation,
            self.actor_model,
        )
        if path == "outer_graph":
            self.graph_learner = example["_DreamerV3CudaGraphLearner"](
                self._forward_backward_fn,
                self.optimizer,
                self.target_updater,
                self.parameters,
                tuple(self.value_loss.target_value_model_params.values(True, True)),
                (self.actor_loss.return_low, self.actor_loss.return_high),
            )
        # Compile and capture are intentionally excluded from the timed region.
        setup_timer = timeit("dreamer_v3/full_learner_setup").start()
        self()
        self.setup_seconds = setup_timer.elapsed()
        _sync(device)

    def _make_sample(self) -> TensorDict:
        generator = torch.Generator(device=self.device).manual_seed(4)
        state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
        sample = TensorDict(
            {
                "state": torch.randn(
                    _BATCH_SIZE,
                    _TIME_STEPS,
                    state_dim,
                    device=self.device,
                    generator=generator,
                ),
                "belief": torch.randn(
                    _BATCH_SIZE,
                    _TIME_STEPS,
                    _BELIEF_DIM,
                    device=self.device,
                    generator=generator,
                ),
                "action": torch.randn(
                    _BATCH_SIZE,
                    _TIME_STEPS,
                    _ACTION_DIM,
                    device=self.device,
                    generator=generator,
                ).tanh(),
                "is_init": torch.zeros(
                    _BATCH_SIZE,
                    _TIME_STEPS,
                    1,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "next": {
                    "observation": torch.randn(
                        _BATCH_SIZE,
                        _TIME_STEPS,
                        24,
                        device=self.device,
                        generator=generator,
                    ),
                    "reward": torch.randn(
                        _BATCH_SIZE,
                        _TIME_STEPS,
                        1,
                        device=self.device,
                        generator=generator,
                    ),
                    "done": torch.zeros(
                        _BATCH_SIZE,
                        _TIME_STEPS,
                        1,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "terminated": torch.zeros(
                        _BATCH_SIZE,
                        _TIME_STEPS,
                        1,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                },
            },
            [_BATCH_SIZE, _TIME_STEPS],
            device=self.device,
        )
        sample["is_init"][:, 0] = True
        return sample

    def _build_forward_backward(self):
        """Reuse the example's learner step rather than re-implementing it.

        The benchmark must measure exactly the computation the training script
        runs; a local copy silently drifts the moment the loss mix changes.
        """
        return _get_sota_example()["build_forward_backward"](
            model_loss=self.model_loss,
            actor_loss=self.actor_loss,
            value_loss=self.value_loss,
            optimizer=self.optimizer,
            state_dim=_NUM_CATEGORICALS * _NUM_CLASSES,
            rnn_hidden_dim=_BELIEF_DIM,
            device=torch.device("cuda"),
            use_bfloat16=True,
            shared_imagination_value=True,
            continuation_horizon=self.cfg.optimization.continuation_horizon,
            lmbda=self.cfg.optimization.lmbda,
            replay_value_loss_weight=(self.cfg.optimization.replay_value_loss_weight),
        )

    def _eager_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self._forward_backward_fn(self.sample, True)
        self.optimizer.step()
        self.target_updater.step()
        return outputs

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph_learner is None:
            outputs = self._eager_step()
        else:
            outputs = self.graph_learner(self.sample)
        _sync(self.device)
        return outputs


@pytest.mark.parametrize("path", ["generic", "tensor", "compiled"])
def test_dreamer_v3_rssm_rollout(benchmark, path: RolloutPath) -> None:
    """Compare RSSM rollout paths at the DMC Walker training shape."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    rollout = _make_rollout(device, path)
    tensordict = _make_tensordict(device)

    # Compilation and allocator setup are intentionally outside the timed region.
    prep_iterations = 2 if path == "compiled" else 1
    for _ in range(prep_iterations):
        _call(rollout, tensordict, device)

    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _TIME_STEPS,
            "transitions_per_call": _BATCH_SIZE * _TIME_STEPS,
            "device": str(device),
        }
    )
    benchmark(_call, rollout, tensordict, device)


@pytest.mark.parametrize("path", ["tensor", "compiled"])
def test_dreamer_v3_imagination_rollout(benchmark, path: ImaginationPath) -> None:
    """Compare tensor and compiled imagination at the Walker training shape."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    rollout = _make_imagination_rollout(device, path)
    tensordict = _make_imagination_tensordict(device)

    prep_iterations = 2 if path == "compiled" else 1
    for _ in range(prep_iterations):
        _call_imagination(rollout, tensordict, device)

    benchmark.extra_info.update(
        {
            "imagination_starts": _BATCH_SIZE * _TIME_STEPS,
            "imagination_horizon": _IMAGINATION_HORIZON,
            "transitions_per_call": (_BATCH_SIZE * _TIME_STEPS * _IMAGINATION_HORIZON),
            "device": str(device),
        }
    )
    benchmark(_call_imagination, rollout, tensordict, device)


@pytest.mark.parametrize("path", ["eager", "compiled"])
def test_dreamer_v3_learner_heads(benchmark, path: LearnerHeadPath) -> None:
    """Compare learner-head forward/backward at the Walker training shape."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    heads = _LearnerHeads(device).to(device)
    inputs = _make_learner_head_inputs(device)
    if path == "compiled":
        heads.compile_heads()

    prep_iterations = 2 if path == "compiled" else 1
    for _ in range(prep_iterations):
        _call_learner_heads(heads, inputs, device)

    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _TIME_STEPS,
            "imagination_horizon": _IMAGINATION_HORIZON,
            "mixed_precision": device.type == "cuda",
            "device": str(device),
        }
    )
    benchmark(_call_learner_heads, heads, inputs, device)


@pytest.mark.parametrize("path", ["repeated", "shared"])
def test_dreamer_v3_imagination_value_heads(benchmark, path: ValuePath) -> None:
    """Compare repeated and shared H+1 value forwards at the Walker shape."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    heads = _ValueHeads(device).to(device)
    inputs = _make_value_inputs(device)
    _call_value_heads(heads, inputs, path, device)

    benchmark.extra_info.update(
        {
            "imagination_starts": _BATCH_SIZE * _TIME_STEPS,
            "imagination_horizon": _IMAGINATION_HORIZON,
            "mixed_precision": device.type == "cuda",
            "device": str(device),
        }
    )
    benchmark(_call_value_heads, heads, inputs, path, device)


@pytest.mark.parametrize("path", ["eager", "compiled"])
def test_dreamer_v3_replay_value_loss(benchmark, path: ReplayValuePath) -> None:
    """Compare replay critic loss forward/backward at the Walker shape."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    loss_module, features, inputs = _make_replay_value_loss(device, path)
    prep_iterations = 2 if path == "compiled" else 1
    for _ in range(prep_iterations):
        _call_replay_value_loss(loss_module, features, inputs, device)

    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _TIME_STEPS,
            "mixed_precision": device.type == "cuda",
            "device": str(device),
        }
    )
    benchmark(_call_replay_value_loss, loss_module, features, inputs, device)


@pytest.mark.parametrize("path", ["recurrence", "full"])
def test_dreamer_v3_policy_prior(benchmark, path: PriorPath) -> None:
    """Measure collection filtering with and without the unused prior head."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    prior, inputs = _make_policy_prior(device)
    _call_policy_prior(prior, inputs, path, device)
    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "belief_dim": _BELIEF_DIM,
            "path": path,
            "device": str(device),
        }
    )
    benchmark(_call_policy_prior, prior, inputs, path, device)


@pytest.mark.parametrize("path", ["latest", "all"])
def test_dreamer_v3_metric_collection(benchmark, path: MetricPath) -> None:
    """Compare latest-only and full per-vector learner metric collection."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    metrics = tuple(torch.randn(6, device=device) for _ in range(16))
    _call_metric_collection(metrics, path, device)
    benchmark.extra_info.update(
        {"updates_per_vector_step": 16, "path": path, "device": str(device)}
    )
    benchmark(_call_metric_collection, metrics, path, device)


@pytest.mark.parametrize("path", ["host", "device"])
def test_dreamer_v3_replay_context(benchmark, path: ReplayPath) -> None:
    """Compare host and learner-device replay at the Walker training shape."""
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    replay_buffer = _make_replay_buffer(device, path)

    _call_replay(replay_buffer, device)
    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _TIME_STEPS,
            "transitions_per_call": _BATCH_SIZE * _TIME_STEPS,
            "learner_device": str(device),
            "storage_device": str(replay_buffer.storage.device),
            "online_replay": True,
        }
    )
    benchmark(_call_replay, replay_buffer, device)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("path", ["eager", "outer_graph"])
def test_dreamer_v3_full_learner(benchmark, path: FullLearnerPath) -> None:
    """Compare the complete eager and outer-CUDA-graph Walker learner update."""
    device = torch.device("cuda:0")
    learner = _FullLearnerBenchmark(device, path)
    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _TIME_STEPS,
            "imagination_horizon": _IMAGINATION_HORIZON,
            "mixed_precision": True,
            "path": path,
            "device": str(device),
            "setup_seconds": learner.setup_seconds,
        }
    )
    benchmark(learner)


if __name__ == "__main__":
    _, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
