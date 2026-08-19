# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the DreamerV3 example builders and replay stream."""
from __future__ import annotations

import argparse

import pytest
import torch
from _dreamer_v3_common import (
    _DreamerV3Rig,
    _EXAMPLE_DIR,
    _has_hydra,
    _has_omegaconf,
    _load_example,
)
from tensordict import TensorDict
from torch import nn

from torchrl.data import LazyTensorStorage, ReplayBuffer, RoundRobinWriter
from torchrl.modules.distributions.continuous import IndependentNormal
from torchrl.objectives.dreamer_v3 import symlog
from torchrl.testing import get_default_devices


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ExampleBuilders(_DreamerV3Rig):
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_shares_imagination_parameters(self, device):
        from omegaconf import OmegaConf

        example = _load_example()
        cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
        cfg.networks.num_reward_bins = self.num_reward_bins
        (world_model, prior, reward_head, reward_decoder, continuation_head,) = example[
            "build_world_model"
        ](cfg=cfg, obs_dim=3, action_dim=self.action_dim)
        imagination_model = example["build_imagination_model"](
            prior_net=prior,
            reward_net=reward_head,
            reward_decoder=reward_decoder,
        ).to(device)
        continuation_model = example["build_continuation_model"](
            continuation_net=continuation_head
        ).to(device)
        world_model = world_model.to(device)
        # Only the terminal-to-reset replay edge keeps this marker set.
        assert world_model[1].reset_key == "is_init"
        observation = torch.tensor(
            [[[0.0, 1.0, -3.0], [2.0, -1.0, 0.5]]], device=device
        )
        world_input = TensorDict(
            {
                "state": torch.zeros(1, 2, self.state_dim, device=device),
                "belief": torch.zeros(1, 2, cfg.networks.rnn_hidden_dim, device=device),
                "action": torch.zeros(1, 2, self.action_dim, device=device),
                "next": {"observation": observation},
            },
            [1, 2],
        )
        world_model(world_input)
        torch.testing.assert_close(
            world_input["next", "symlog_observation"], symlog(observation)
        )
        shared_parameters = tuple(prior.parameters()) + tuple(reward_head.parameters())
        world_parameters = tuple(world_model.parameters())
        imagination_parameters = tuple(imagination_model.parameters())
        assert all(
            any(parameter is candidate for candidate in world_parameters)
            and any(parameter is candidate for candidate in imagination_parameters)
            for parameter in shared_parameters
        )
        assert all(
            any(parameter is candidate for candidate in world_parameters)
            and any(
                parameter is candidate for candidate in continuation_model.parameters()
            )
            for parameter in continuation_head.parameters()
        )

        reward_td = TensorDict(
            {
                "state": torch.randn(2, self.state_dim, device=device),
                "belief": torch.randn(2, cfg.networks.rnn_hidden_dim, device=device),
            },
            [2],
        )
        imagination_model.get_reward_operator()(reward_td)
        assert reward_td["reward_logits"].shape == (2, self.num_reward_bins)
        assert reward_td["reward"].shape == (2, 1)
        continuation_model(reward_td)
        assert reward_td["continuation"].shape == (2, 1)

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_optimizer_warmup_and_per_parameter_statistics(self, device):
        example = _load_example()
        parameter = nn.Parameter(torch.tensor([2.0, -1.0], device=device))
        optimizer = example["DreamerV3Optimizer"](
            [parameter],
            lr=0.1,
            agc=0.3,
            beta1=0.9,
            beta2=0.999,
            eps=0.0,
            warmup_steps=2,
        )
        initial = parameter.detach().clone()
        parameter.grad = torch.tensor([4.0, 3.0], device=device)
        optimizer.step()
        # The linear warmup gives the first update a rate of zero.
        torch.testing.assert_close(parameter, initial)
        parameter.grad = torch.tensor([4.0, 3.0], device=device)
        optimizer.step()
        torch.testing.assert_close(parameter, initial - 0.05)
        assert optimizer.state[parameter]["rms"].dtype == torch.float32

        # Shapes share a foreach bucket, but each keeps its own statistics.
        parameters = [
            nn.Parameter(torch.tensor([2.0, -1.0], device=device)),
            nn.Parameter(torch.tensor([[0.5, -3.0, 1.5]], device=device)),
        ]
        gradients = [
            torch.tensor([4.0, -3.0], device=device),
            torch.tensor([[-2.0, 1.0, 5.0]], device=device),
        ]
        initials = [parameter.detach().clone() for parameter in parameters]
        optimizer = example["DreamerV3Optimizer"](
            parameters,
            lr=0.1,
            agc=0.3,
            beta1=0.9,
            beta2=0.999,
            eps=0.0,
            warmup_steps=2,
        )
        for _ in range(2):
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient.clone()
            optimizer.step()
        for parameter, initial, gradient in zip(parameters, initials, gradients):
            torch.testing.assert_close(
                parameter,
                initial - 0.05 * gradient.sign(),
            )

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_real_world_actor(self, device):
        from omegaconf import OmegaConf

        example = _load_example()
        cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
        world_model, _, _, _, _ = example["build_world_model"](
            cfg=cfg, obs_dim=3, action_dim=self.action_dim
        )
        world_model = world_model.to(device)
        actor = example["build_actor"](cfg=cfg, action_dim=self.action_dim).to(device)
        policy = example["build_real_world_actor"](
            world_model=world_model,
            actor_model=actor,
        )

        observation = torch.tensor(
            [[0.0, 1.0, -3.0], [20.0, -10.0, 5.0]], device=device
        )
        incoming_state = torch.randn(2, self.state_dim, device=device)
        incoming_belief = torch.randn(2, cfg.networks.rnn_hidden_dim, device=device)
        previous_action = torch.randn(2, self.action_dim, device=device)
        is_init = torch.tensor([[True], [False]], device=device)

        encoder = world_model[0][1].module
        prior = world_model[1].rssm_prior.module
        posterior = world_model[1].rssm_posterior.module
        encoded = encoder(symlog(observation))
        reset = is_init.expand_as(incoming_state)
        expected_state_input = torch.where(reset, 0, incoming_state)
        expected_belief_input = torch.where(
            is_init.expand_as(incoming_belief), 0, incoming_belief
        )
        expected_action_input = torch.where(
            is_init.expand_as(previous_action), 0, previous_action
        )
        torch.manual_seed(0)
        expected_belief = prior._update_belief(
            expected_state_input,
            expected_belief_input,
            expected_action_input,
        )
        expected_logits, expected_state = posterior(expected_belief, encoded)

        policy_input = TensorDict(
            {
                "observation": observation,
                "state": incoming_state,
                "belief": incoming_belief,
                "previous_action": previous_action,
                "is_init": is_init,
            },
            [2],
        )
        prior_projector_calls = 0

        def count_prior_projector_calls(*_):
            nonlocal prior_projector_calls
            prior_projector_calls += 1

        prior_hook = prior.rnn_to_prior_projector.register_forward_hook(
            count_prior_projector_calls
        )
        torch.manual_seed(0)
        policy(policy_input)
        prior_hook.remove()

        torch.testing.assert_close(policy_input["state"], expected_state)
        torch.testing.assert_close(policy_input["belief"], expected_belief)
        torch.testing.assert_close(policy_input["next", "state"], expected_state)
        torch.testing.assert_close(policy_input["next", "belief"], expected_belief)
        torch.testing.assert_close(
            policy_input["next", "previous_action"], policy_input["action"]
        )
        assert not torch.equal(expected_logits[0], expected_logits[1])
        assert prior_projector_calls == 0

        distribution = actor.get_dist(policy_input)
        assert isinstance(distribution, IndependentNormal)
        assert policy_input["loc"].dtype == torch.float32
        assert policy_input["scale"].dtype == torch.float32
        assert (policy_input["loc"].abs() <= 1).all()
        assert (policy_input["scale"] >= cfg.networks.policy_min_std).all()
        assert (policy_input["scale"] <= cfg.networks.policy_max_std).all()

        world_parameters = tuple(world_model.parameters())
        policy_parameters = tuple(policy.parameters())
        shared_parameters = (
            tuple(encoder.parameters())
            + tuple(prior.parameters())
            + tuple(posterior.parameters())
        )
        assert all(
            any(parameter is candidate for candidate in world_parameters)
            and any(parameter is candidate for candidate in policy_parameters)
            for parameter in shared_parameters
        )
        assert all(
            any(parameter is candidate for candidate in policy_parameters)
            for parameter in actor.parameters()
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_policy_outputs_are_stackable(self, device):
        from omegaconf import OmegaConf

        example = _load_example()
        cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
        world_model, _, reward, _, continuation = example["build_world_model"](
            cfg=cfg, obs_dim=3, action_dim=self.action_dim
        )
        world_model = world_model.to(device)
        actor = example["build_actor"](cfg=cfg, action_dim=self.action_dim).to(device)
        policy = example["build_real_world_actor"](
            world_model=world_model,
            actor_model=actor,
            mixed_precision=True,
        )

        def policy_input(
            state: torch.Tensor,
            belief: torch.Tensor,
            previous_action: torch.Tensor,
        ) -> TensorDict:
            return TensorDict(
                {
                    "observation": torch.randn(2, 3, device=device),
                    "state": state,
                    "belief": belief,
                    "previous_action": previous_action,
                    "is_init": torch.zeros(2, 1, dtype=torch.bool, device=device),
                },
                [2],
            )

        state = torch.zeros(2, self.state_dim, device=device)
        belief = torch.zeros(2, cfg.networks.rnn_hidden_dim, device=device)
        previous_action = torch.zeros(2, self.action_dim, device=device)
        with torch.inference_mode():
            first = policy_input(state, belief, previous_action)
            policy(first)
            assert first["state"].dtype == torch.float32
            assert first["belief"].dtype == torch.float32
            second = policy_input(
                first["next", "state"],
                first["next", "belief"],
                first["next", "previous_action"],
            )
            policy(second)
            # The collector keeps every policy output until it stacks them.
            assert (
                first["next", "state"].data_ptr() != second["next", "state"].data_ptr()
            )
            assert torch.stack([first, second], 0).shape == (2, 2)


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ExampleReplay(_DreamerV3Rig):
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_replay_prefetch_delay_and_overlap(self, device):
        example = _load_example()

        class CountingReplay:
            def __init__(self):
                self.samples = 0

            def sample(self, return_info):
                assert return_info
                sample_id = self.samples
                self.samples += 1
                return TensorDict({"sample_id": torch.tensor(sample_id)}, []), {
                    "index": torch.tensor([sample_id]),
                    "index_generation": torch.tensor([0]),
                }

        counting_replay = CountingReplay()
        pipeline = example["DreamerV3ReplayPipeline"]()
        pipeline.prefetch(counting_replay)
        assert counting_replay.samples == 1
        first, _ = pipeline.take(counting_replay)
        assert first["sample_id"] == 0
        assert counting_replay.samples == 2
        second, _ = pipeline.take(counting_replay)
        assert second["sample_id"] == 1
        assert counting_replay.samples == 3

        replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=8, device=device),
            writer=RoundRobinWriter(track_generations=True),
        )
        replay.extend(
            TensorDict(
                {
                    "state": torch.zeros(5, 1, device=device),
                    "belief": torch.zeros(5, 2, device=device),
                    "collector": {
                        "context_valid": torch.zeros(
                            5, 1, dtype=torch.bool, device=device
                        )
                    },
                },
                [5],
            )
        )
        overlapping_info = {
            "index": torch.tensor([0, 1, 2, 3, 1, 2, 3, 4]),
            "index_generation": torch.zeros(8, dtype=torch.int64),
        }
        overlapping_state = torch.tensor(
            [[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]],
            device=device,
        )
        overlapping_belief = overlapping_state.expand(2, 3, 2) + 100
        example["_refresh_replay_context"](
            replay,
            overlapping_info["index"],
            overlapping_info["index_generation"],
            overlapping_state,
            overlapping_belief,
        )
        torch.testing.assert_close(
            replay.storage[:]["state"],
            torch.tensor([[0.0], [10.0], [20.0], [21.0], [22.0]], device=device),
        )
        torch.testing.assert_close(
            replay.storage[:]["belief"],
            torch.tensor(
                [
                    [0.0, 0.0],
                    [110.0, 110.0],
                    [120.0, 120.0],
                    [121.0, 121.0],
                    [122.0, 122.0],
                ],
                device=device,
            ),
        )

        replay.storage[:]["state"].zero_()
        replay.storage[:]["belief"].zero_()
        delayed = example["DreamerV3ReplayPipeline"]()
        first_info = {
            "index": torch.tensor([0, 1, 2, 3]),
            "index_generation": torch.zeros(4, dtype=torch.int64),
        }
        second_info = {
            "index": torch.tensor([1, 2, 3, 4]),
            "index_generation": torch.zeros(4, dtype=torch.int64),
        }
        first_state = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
        first_belief = first_state.expand(1, 3, 2) + 10
        delayed.stage_context(first_info, first_state, first_belief)
        assert delayed.has_pending_context
        assert not replay.storage[:]["state"].any()
        delayed.apply_pending_context(replay)
        assert not delayed.has_pending_context
        torch.testing.assert_close(
            replay.storage[:]["state"],
            torch.tensor([[0.0], [1.0], [2.0], [3.0], [0.0]], device=device),
        )
        delayed.stage_context(
            second_info,
            first_state + 100,
            first_belief + 100,
        )
        assert delayed.has_pending_context
        torch.testing.assert_close(
            replay.storage[:]["state"],
            torch.tensor([[0.0], [1.0], [2.0], [3.0], [0.0]], device=device),
        )

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_tail_placeholder_stays_out_of_the_learner(self, device):
        """The newest record is a placeholder that the learner's trim removes."""
        example = _load_example()

        num_streams, time_steps, slice_len = 2, 6, 3
        observation = torch.arange(
            num_streams * time_steps, dtype=torch.float32
        ).reshape(num_streams, time_steps, 1)
        collector_data = TensorDict(
            {
                "observation": observation,
                "action": observation + 10,
                "is_init": torch.zeros(num_streams, time_steps, 1, dtype=torch.bool),
                "state": observation + 20,
                "belief": observation.expand(num_streams, time_steps, 2) + 30,
                "collector": {
                    "traj_ids": torch.zeros(num_streams, time_steps, dtype=torch.long)
                },
                "next": {
                    "observation": observation + 100,
                    "reward": observation + 40,
                    "done": torch.zeros(num_streams, time_steps, 1, dtype=torch.bool),
                    "terminated": torch.zeros(
                        num_streams, time_steps, 1, dtype=torch.bool
                    ),
                },
            },
            [num_streams, time_steps],
        )
        records = example["DreamerV3ReplayRecordBuilder"](num_streams)(collector_data)

        sampler = example["DreamerV3ReplaySampler"](
            slice_len=slice_len,
            online=False,
        )
        replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=64, ndim=2, device=device),
            dim_extend=1,
            writer=RoundRobinWriter(track_generations=True),
            sampler=sampler,
            batch_size=8 * slice_len,
            generator=torch.Generator().manual_seed(0),
        )
        example["DreamerV3ShiftedRecordExtender"](num_streams).extend(
            replay, sampler, records.to(device)
        )

        assert not replay.storage[-1]["collector", "context_valid"].any()

        reached_tail = False
        for _ in range(32):
            sample = replay.sample().reshape(8, slice_len)
            valid = sample["collector", "context_valid"]
            assert valid[:, :-1].all(), "placeholder leaked into the learner slice"
            reached_tail = reached_tail or not valid[:, -1].all()
        # The assertion above means something only if a window reaches the tail.
        assert reached_tail, "no sampled window ended at the placeholder"

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_continuous_online_replay(self, device):
        example = _load_example()

        num_streams, time_steps = 2, 3
        observation = torch.arange(
            num_streams * time_steps, dtype=torch.float32
        ).reshape(num_streams, time_steps, 1)
        next_observation = observation + 100
        is_init = torch.tensor([[[True], [False], [True]], [[True], [False], [True]]])
        collector_data = TensorDict(
            {
                "observation": observation,
                "action": observation + 10,
                "is_init": is_init,
                "state": observation + 20,
                "belief": observation.expand(num_streams, time_steps, 2) + 30,
                "collector": {"traj_ids": torch.tensor([[0, 0, 1], [2, 2, 3]])},
                "next": {
                    "observation": next_observation,
                    "reward": observation + 40,
                    "done": torch.tensor(
                        [[[False], [True], [False]], [[False], [True], [False]]]
                    ),
                    "terminated": torch.zeros(
                        num_streams, time_steps, 1, dtype=torch.bool
                    ),
                },
            },
            [num_streams, time_steps],
        )
        builder = example["DreamerV3ReplayRecordBuilder"](num_streams)
        records = builder(collector_data)

        # The first reset gives the context record only. Each later reset adds
        # one zero-action edge that trains the reset observation.
        assert records.shape == (num_streams, 4)
        torch.testing.assert_close(
            records["is_init"],
            torch.tensor(
                [
                    [[False], [False], [True], [False]],
                    [[False], [False], [True], [False]],
                ]
            ),
        )
        torch.testing.assert_close(
            records["next", "observation"],
            torch.stack(
                [
                    next_observation[:, 0],
                    next_observation[:, 1],
                    observation[:, 2],
                    next_observation[:, 2],
                ],
                1,
            ),
        )
        torch.testing.assert_close(
            records["action"][:, 2],
            torch.zeros(num_streams, 1),
        )
        assert not records["next", "done"][:, 2].any()
        sampler = example["DreamerV3ReplaySampler"](
            slice_len=3,
            online=True,
        )
        replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=40, ndim=2, device=device),
            dim_extend=1,
            writer=RoundRobinWriter(track_generations=True),
            sampler=sampler,
            batch_size=6,
        )
        shifted_writer = example["DreamerV3ShiftedRecordExtender"](num_streams)
        written = shifted_writer.extend(replay, sampler, records.to(device))
        assert written.shape == (10, 2)
        assert replay.storage.shape == torch.Size([5, num_streams])
        # 4 shifted edges plus the tail give 3 starts of a 3-record window.
        assert replay.storage.shape[0] - sampler.slice_len + 1 == 3
        torch.testing.assert_close(
            replay.storage[:]["action"].transpose(0, 1),
            torch.cat([records["action"], torch.zeros(num_streams, 1, 1)], dim=1).to(
                device
            ),
        )
        assert not replay.storage[-1]["collector", "context_valid"].any()
        assert sampler.online_queue_size == 2

        # One batch takes up to num_slices blocks from the queue.
        sample, info = replay.sample(return_info=True)
        sample = sample.reshape(2, 3)
        sampled_index = torch.stack(info["index"], -1).reshape(2, 3, 2)
        torch.testing.assert_close(
            sampled_index[0, :, 0],
            torch.tensor([1, 2, 3], device=sampled_index.device),
        )
        torch.testing.assert_close(
            sampled_index[1, :, 0],
            torch.tensor([1, 2, 3], device=sampled_index.device),
        )
        assert (sampled_index[0, :, 1] == 0).all()
        assert (sampled_index[1, :, 1] == 1).all()
        assert sampler.online_queue_size == 0

        # The empty queue makes the next batch draw uniform starts.
        replay.sample(return_info=True)
        assert sampler.online_queue_size == 0

        # A window that ends at the tail infers a posterior there, so the
        # finalization must keep that context and the tail generation.
        tail_index = shifted_writer._tail_index.clone()
        tail_generation = shifted_writer._tail_generation.clone()
        refresh_index = (
            torch.tensor([2, 3, 4, 2, 3, 4], device=device),
            torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        )
        refresh_generation = replay.writer.generations_of(
            torch.stack(refresh_index, -1)
        )
        refreshed_state = torch.tensor(
            [[[11.0], [12.0]], [[21.0], [22.0]]], device=device
        )
        refreshed_belief = refreshed_state.expand(2, 2, 2) + 100
        example["_refresh_replay_context"](
            replay,
            refresh_index,
            refresh_generation,
            refreshed_state,
            refreshed_belief,
        )
        assert replay.storage[-1]["collector", "context_valid"].all()

        # One update follows each worker add, so a batch admits one block.
        newer_indices = shifted_writer.extend(
            replay, sampler, records[:, :3].to(device)
        )
        assert newer_indices.shape == (6, 2)
        torch.testing.assert_close(
            replay.writer.generations_of(tail_index), tail_generation
        )
        torch.testing.assert_close(
            replay.storage[4]["action"], records[:, 0]["action"].to(device)
        )
        torch.testing.assert_close(replay.storage[4]["state"], refreshed_state[:, -1])
        torch.testing.assert_close(replay.storage[4]["belief"], refreshed_belief[:, -1])
        assert replay.storage[4]["collector", "context_valid"].all()
        assert not replay.storage[-1]["collector", "context_valid"].any()
        assert sampler.online_queue_size == 2

        # The batch serves both queued blocks, one for each stream.
        _, second_info = replay.sample(return_info=True)
        second_index = torch.stack(second_info["index"], -1).reshape(2, 3, 2)
        torch.testing.assert_close(
            second_index[0, :, 0],
            torch.tensor([4, 5, 6], device=second_index.device),
        )
        torch.testing.assert_close(
            second_index[1, :, 0],
            torch.tensor([4, 5, 6], device=second_index.device),
        )
        assert (second_index[0, :, 1] == 0).all()
        assert (second_index[1, :, 1] == 1).all()
        assert sampler.online_queue_size == 0

        uniform_sample, _ = replay.sample(return_info=True)
        assert uniform_sample.shape == (6,)

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_initial_context_cardinality(self, device):
        example = _load_example()

        records = TensorDict(
            {
                "action": torch.tensor([[[1.0], [2.0]]], device=device),
                "is_init": torch.zeros(1, 2, 1, dtype=torch.bool, device=device),
                "state": torch.tensor([[[3.0], [4.0]]], device=device),
                "belief": torch.tensor([[[5.0], [6.0]]], device=device),
                "collector": {
                    "traj_ids": torch.zeros(1, 2, dtype=torch.long, device=device),
                    "context_valid": torch.ones(
                        1, 2, 1, dtype=torch.bool, device=device
                    ),
                },
                "next": {
                    "observation": torch.tensor([[[7.0], [8.0]]], device=device),
                    "reward": torch.zeros(1, 2, 1, device=device),
                    "done": torch.zeros(1, 2, 1, dtype=torch.bool, device=device),
                    "terminated": torch.zeros(1, 2, 1, dtype=torch.bool, device=device),
                },
            },
            [1, 2],
            device=device,
        )
        sampler = example["DreamerV3ReplaySampler"](
            slice_len=3,
            online=False,
        )
        replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=10, device=device),
            writer=RoundRobinWriter(track_generations=True),
            sampler=sampler,
            batch_size=3,
            generator=torch.Generator().manual_seed(0),
        )
        shifted_writer = example["DreamerV3ShiftedRecordExtender"](1)

        written = shifted_writer.extend(replay, sampler, records)
        assert written.shape == (3,)
        assert len(replay) == 3
        assert replay.storage.shape[0] - sampler.slice_len + 1 == 1
        _, info = replay.sample(return_info=True)
        sampled_index = info["index"]
        if isinstance(sampled_index, tuple):
            sampled_index = sampled_index[0]
        torch.testing.assert_close(
            sampled_index, torch.tensor([0, 1, 2], device=sampled_index.device)
        )

        tail_index = shifted_writer._tail_index.clone()
        tail_generation = shifted_writer._tail_generation.clone()
        shifted_writer.extend(replay, sampler, records[:, 1:])
        assert len(replay) == 4
        assert replay.storage.shape[0] - sampler.slice_len + 1 == 2
        torch.testing.assert_close(
            replay.writer.generations_of(tail_index), tail_generation
        )
        torch.testing.assert_close(replay.storage[2]["action"], records[0, 1]["action"])
        torch.testing.assert_close(replay.storage[2]["state"], records[0, 1]["state"])
        assert replay.storage[2]["collector", "context_valid"].all()
        assert not replay.storage[3]["collector", "context_valid"].any()
        assert sampler.online_queue_size == 0


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
