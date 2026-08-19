# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the DreamerV3 walker protocol and its benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import math
import runpy

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
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.envs.libs.gym import _has_gym
from torchrl.testing import get_default_devices


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ExampleProtocolParity(_DreamerV3Rig):
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_deferred_policy_sync(self, device):
        del device
        example = _load_example()
        learner = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            learner.weight.zero_()
        behavior = copy.deepcopy(learner)
        sync = example["DreamerV3BehaviorPolicySync"](learner, behavior)

        used_versions = []
        for _ in range(5):
            # The behavior policy acts before the sync installs a snapshot.
            used_versions.append(behavior.weight.item())
            sync.apply_after_action()
            sync.stage_before_training()
            for _ in range(16):
                with torch.no_grad():
                    learner.weight.add_(1)
                # A later update cannot replace a pending snapshot.
                sync.stage_before_training()

        assert used_versions == [0.0, 0.0, 0.0, 16.0, 32.0]
        assert sync.has_pending

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_seeded_policy_stream(self, device):
        del device
        example = _load_example()
        stochastic = TensorDictModule(
            lambda value: torch.rand_like(value),
            in_keys=["input"],
            out_keys=["sample"],
        )
        seeded_policy = example["DreamerV3SeededPolicy"](stochastic, seed=3)
        torch.manual_seed(123)
        global_state = torch.random.get_rng_state().clone()
        first = TensorDict({"input": torch.zeros(4)}, [])
        second = TensorDict({"input": torch.zeros(4)}, [])
        seeded_policy(first)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)
        seeded_policy(second)
        assert not torch.equal(first["sample"], second["sample"])
        repeated_policy = example["DreamerV3SeededPolicy"](stochastic, seed=3)
        repeated = TensorDict({"input": torch.zeros(4)}, [])
        repeated_policy(repeated)
        torch.testing.assert_close(first["sample"], repeated["sample"])
        assert seeded_policy.counter == 2
        seeded_policy.reset_counter()
        restarted = TensorDict({"input": torch.zeros(4)}, [])
        seeded_policy(restarted)
        assert seeded_policy.counter == 1
        torch.testing.assert_close(first["sample"], restarted["sample"])

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_record_counter_and_update_cadence(self, device):
        del device
        example = _load_example()
        action_budget = example["collector_action_budget"]
        driver_step = example["driver_step_for_action"]
        update_ratio_type = example["DreamerV3UpdateRatio"]

        assert action_budget(1_000_000, 16, 1000) == 998_992
        assert action_budget(1_100_000, 16, 1000) == 1_098_896
        assert [driver_step(1000, index, 16, 1000) for index in range(16)] == list(
            range(16_001, 16_017)
        )
        assert [driver_step(2000, index, 16, 1000) for index in range(16)] == list(
            range(32_017, 32_033)
        )
        # 1024 over 16x64 replay elements is one update for each record, and
        # the first eligible record trains once.
        walker_ratio = update_ratio_type(1024 / (16 * 64))
        walker_updates = [walker_ratio(16 * (step + 1)) for step in range(5)]
        assert walker_updates == [1, 16, 16, 16, 16]

        # 16 records at a ratio of 32 give half an update, so a carried
        # remainder makes the batches alternate.
        fractional = update_ratio_type(32 / (16 * 64))
        fractional_updates = [fractional(16 * (step + 1)) for step in range(9)]
        assert fractional_updates == [1, 0, 1, 0, 1, 0, 1, 0, 1]

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf and _has_gym),
        reason="requires hydra, omegaconf and gym",
    )
    def test_dreamer_v3_sota_protocol_end_to_end(self, device, tmp_path):
        """The run turns on every protocol knob that the default config omits.

        126 records over 2 environments is 63 vector records, 3 of which are
        resets, which leaves 120 actions.
        """
        del device
        from omegaconf import OmegaConf

        example = _load_example()
        cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
        cfg.env.max_episode_steps = 20
        cfg.collector.num_envs = 2
        cfg.collector.count_reset_records = True
        cfg.collector.total_frames = 126
        cfg.collector.frames_per_batch = 4
        cfg.optimization.deferred_policy_sync = True
        cfg.optimization.separate_policy_rng = True
        cfg.optimization.updates_per_batch = 1
        cfg.replay_buffer.batch_size = 2
        cfg.replay_buffer.seq_len = 4
        cfg.replay_buffer.warmup_factor = 1
        cfg.networks.hidden_dim = 8
        cfg.networks.rnn_hidden_dim = 8
        cfg.networks.num_categoricals = 2
        cfg.networks.num_classes = 2
        cfg.networks.num_reward_bins = 11
        cfg.networks.num_value_bins = 11
        for layers in (
            "encoder_layers",
            "decoder_layers",
            "reward_layers",
            "actor_layers",
            "value_layers",
        ):
            cfg.networks[layers] = 1
        cfg.logger.eval_every = 0
        cfg.logger.train_every = 60
        cfg.logger.output_plot = None
        cfg.logger.metrics_jsonl = str(tmp_path / "metrics.jsonl")

        # ``main`` seeds the global generator.
        with torch.random.fork_rng():
            example["main"].__wrapped__(cfg)

        records = [
            json.loads(line)
            for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
            if line
        ]
        assert records[-1]["type"] == "summary", "summary must close the stream"
        metrics = records[-1]
        # The record count also holds the reset records, so it is the larger.
        assert metrics["total_action_steps"] == 120
        assert metrics["total_environment_steps"] == 126
        assert metrics["updates"] > 0
        # Each of the 2 environments ran 3 episodes to the step limit.
        episodes = [r for r in records if r["type"] == "train_episode"]
        assert len(episodes) == 6

        # ``output_plot = None`` must still leave the windowed loss means.
        train_records = [r for r in records if r["type"] == "train"]
        assert train_records, "no train records were written"
        loss_keys = [
            "loss_dynamic_representation",
            "loss_reconstruction",
            "loss_reward",
            "loss_actor",
            "loss_value",
            "loss_replay_value",
        ]
        for record in train_records:
            assert record["updates_in_window"] > 0
            for key in loss_keys:
                assert key in record, f"{key} missing with output_plot disabled"
                assert math.isfinite(record[key])
        # The windows must cover the full run.
        assert sum(r["updates_in_window"] for r in train_records) == metrics["updates"]


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ExampleWalkerConfig(_DreamerV3Rig):
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_dmc_parameter_parity(self, device):
        from omegaconf import OmegaConf

        del device
        example = _load_example()
        cfg = OmegaConf.merge(
            OmegaConf.load(_EXAMPLE_DIR / "config.yaml"),
            OmegaConf.load(_EXAMPLE_DIR / "config_dmc_walker.yaml"),
        )
        world_model, _, _, _, _ = example["build_world_model"](
            cfg=cfg, obs_dim=24, action_dim=6
        )
        actor = example["build_actor"](cfg=cfg, action_dim=6)
        value = example["build_value"](cfg=cfg)

        def parameter_count(module):
            return sum(parameter.numel() for parameter in module.parameters())

        counts = {
            "enc": parameter_count(world_model[0]),
            "dyn": parameter_count(world_model[1]),
            "dec": parameter_count(world_model[2]),
            "rew": parameter_count(world_model[3]),
            "con": parameter_count(world_model[4]),
            "pol": parameter_count(actor),
            "val": parameter_count(value),
        }
        assert counts == {
            "enc": 10_112,
            "dyn": 364_416,
            "dec": 51_096,
            "rew": 57_663,
            "con": 41_153,
            "pol": 50_316,
            "val": 66_111,
        }
        decoder_core = world_model[2][0].module
        assert [tuple(head.weight.shape) for head in decoder_core.output_heads] == [
            (1, 64),
            (14, 64),
            (9, 64),
        ]
        assert world_model[2].in_keys == [
            ("next", "state"),
            ("next", "belief"),
        ]
        assert world_model[3][0].in_keys == [
            ("next", "belief"),
            ("next", "state"),
        ]
        assert world_model[4][0].in_keys == [
            ("next", "belief"),
            ("next", "state"),
        ]
        assert value[0].in_keys == ["belief", "state"]

        class CaptureFeatures(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = None

            def forward(self, *inputs):
                self.inputs = inputs
                return torch.zeros(*inputs[0].shape[:-1], cfg.networks.hidden_dim)

        capture = CaptureFeatures()
        actor_core = actor[0].module
        actor_core.backbone = capture
        state = torch.randn(3, 128)
        belief = torch.randn(3, 512)
        actor_core(state, belief)
        assert capture.inputs[0] is belief
        assert capture.inputs[1] is state
        assert actor_core.mean_head.weight.shape == (6, 64)
        assert actor_core.std_head.weight.shape == (6, 64)

    @pytest.mark.skipif(not _has_omegaconf, reason="requires omegaconf")
    def test_dreamer_v3_dmc_benchmark_aggregation(self, device, tmp_path):
        from omegaconf import OmegaConf

        del device
        benchmark = runpy.run_path(
            _EXAMPLE_DIR / "benchmark.py",
            run_name="dreamer_v3_benchmark_test",
        )
        paths = []
        for seed, returns in enumerate(([1.0, 4.0], [3.0, 6.0], [2.0, 5.0])):
            path = tmp_path / f"seed_{seed}.jsonl"
            lines = [
                json.dumps(
                    {"type": "train_episode", "environment_steps": step, "score": score}
                )
                for step, score in zip((50, 150), returns)
            ]
            lines.append(
                json.dumps(
                    {
                        "type": "summary",
                        "seed": seed,
                        "total_environment_steps": 200,
                    }
                )
            )
            path.write_text("\n".join(lines) + "\n")
            paths.append(path)

        summary = benchmark["aggregate_runs"](paths, window_size=100)
        assert summary["environment_steps"] == [100, 200]
        assert summary["median_return"] == [2.0, 5.0]
        assert summary["lower_quartile_return"] == [1.5, 4.5]
        assert summary["upper_quartile_return"] == [2.5, 5.5]
        assert summary["window_size"] == 100

        # An even seed count has no middle element, so a median and its
        # quartiles must share one quantile method to stay ordered.
        even = benchmark["aggregate_runs"](paths[:2], window_size=100)
        for median, low, high in zip(
            even["median_return"],
            even["lower_quartile_return"],
            even["upper_quartile_return"],
        ):
            assert low <= median <= high
        assert even["median_return"] == [2.0, 5.0]

        # An override of these would report one trajectory as several seeds.
        for reserved in ("env.seed=7", "++logger.metrics_jsonl=/tmp/x.jsonl"):
            with pytest.raises(ValueError, match="cannot be overridden"):
                benchmark["reject_reserved_overrides"]([reserved])
        benchmark["reject_reserved_overrides"](
            ["collector.total_frames=1000", "benchmark.window_size=1000"]
        )

        # The episodes finish together, one cycle of 16016 = (1000+1)*16 steps
        # apart, so a smaller window is empty for most of the run.
        config = benchmark["effective_config"]()
        assert (
            benchmark["episode_cycle"](config)
            == (config.env.max_episode_steps + 1) * config.collector.num_envs
        )
        benchmark["validate_window_size"](50_000)
        with pytest.raises(ValueError, match="below the 16016-step episode cycle"):
            benchmark["validate_window_size"](1000)
        # The cycle takes the same overrides that the runs receive.
        with pytest.raises(ValueError, match="below the 4004-step episode cycle"):
            benchmark["validate_window_size"](2000, ["collector.num_envs=4"])
        benchmark["validate_window_size"](5000, ["env.max_episode_steps=100"])

        # A run that stops early has no summary, so its step count is unknown.
        truncated = tmp_path / "seed_truncated.jsonl"
        truncated.write_text(
            json.dumps({"type": "train_episode", "environment_steps": 50, "score": 1.0})
            + "\n"
        )
        with pytest.raises(ValueError, match="no summary record"):
            benchmark["aggregate_runs"]([truncated], window_size=100)

        settings = benchmark["benchmark_settings"]()
        assert settings == {
            "seeds": [0, 1, 2],
            "minimum_final_median_return": 900.0,
            "window_size": 50_000,
        }
        overridden = benchmark["benchmark_settings"](
            ["collector.total_frames=1000", "benchmark.window_size=1000"]
        )
        assert overridden["window_size"] == 1000
        assert overridden["seeds"] == [0, 1, 2]

        # The preset runs the reference path, not the smoke defaults.
        config = OmegaConf.load(_EXAMPLE_DIR / "config_dmc_walker.yaml")
        assert config.optimization.train_ratio == 1024
        assert config.optimization.deferred_policy_sync
        assert config.optimization.separate_policy_rng
        assert config.collector.count_reset_records
        assert config.replay_buffer.online


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
