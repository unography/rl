# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the DreamerV3 walker protocol: JAX parity and the benchmark.

The builders and the replay stream are covered in test_dreamer_v3_example.py,
the losses in test_dreamer_v3.py.
"""
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
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import ValueEstimators
from torchrl.testing import get_default_devices


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ExampleJaxParity(_DreamerV3Rig):
    """Step accounting, RNG streams and diagnostics against the JAX reference."""

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_jax_behavior_policy_sync(self, device):
        del device
        example = _load_example()
        learner = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            learner.weight.zero_()
        behavior = copy.deepcopy(learner)
        sync = example["DreamerV3BehaviorPolicySync"](learner, behavior)

        used_versions = []
        for _ in range(5):
            # The action is produced before the pending snapshot is installed.
            used_versions.append(behavior.weight.item())
            sync.apply_after_action()
            sync.stage_before_training()
            for _ in range(16):
                with torch.no_grad():
                    learner.weight.add_(1)
                # Later updates cannot replace the first pending snapshot.
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
    def test_dreamer_v3_jax_record_counter_and_update_cadence(self, device):
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
        # The walker cadence: train_ratio 1024 over 16x64 replay elements is
        # exactly one update per driver record. The first eligible record
        # trains once, as the reference's uninitialized branch does.
        walker_ratio = update_ratio_type(1024 / (16 * 64))
        walker_updates = [walker_ratio(16 * (step + 1)) for step in range(5)]
        assert walker_updates == [1, 16, 16, 16, 16]

        # A ratio that does not divide the collector batch must carry its
        # remainder rather than rounding each batch independently: 16 records
        # at train_ratio 32 is half an update, so batches alternate.
        fractional = update_ratio_type(32 / (16 * 64))
        fractional_updates = [fractional(16 * (step + 1)) for step in range(9)]
        assert fractional_updates == [1, 0, 1, 0, 1, 0, 1, 0, 1]

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf and _has_gym),
        reason="requires hydra, omegaconf and gym",
    )
    def test_dreamer_v3_sota_jax_protocol_end_to_end(self, device, tmp_path):
        """Run the example under the reference collection protocol.

        The default config leaves ``count_reset_records`` off, which bypasses
        the shifted replay writer, the mutable tail, the behavior-policy sync
        and the seeded policy stream. Drive a short multi-environment run with
        them all enabled so the reference protocol is covered end to end.

        A 20-step episode limit forces several resets, so the synthetic
        terminal-to-reset record and the tail finalization that follows it are
        exercised repeatedly. 126 records over 2 environments is 63 vector
        records, 3 of which are resets, leaving 120 actions.
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
        cfg.optimization.jax_behavior_policy_sync = True
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

        # ``main`` seeds the global generator, so keep it out of other tests.
        with torch.random.fork_rng():
            example["main"].__wrapped__(cfg)

        records = [
            json.loads(line)
            for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
            if line
        ]
        assert records[-1]["type"] == "summary", "summary must close the stream"
        metrics = records[-1]
        # The record axis counts the initial reset observations and every
        # synthetic reset record, so it exceeds the control-action count.
        assert metrics["total_action_steps"] == 120
        assert metrics["total_environment_steps"] == 126
        assert metrics["updates"] > 0
        # Every episode ran to the step limit, so each environment finished
        # exactly three of them.
        episodes = [r for r in records if r["type"] == "train_episode"]
        assert len(episodes) == 6

        # This run sets ``output_plot = None``: the windowed loss means must
        # be logged anyway.
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
        # The windows must tile the run, leaving no update unaccounted for.
        assert sum(r["updates_in_window"] for r in train_records) == metrics["updates"]

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_reference_diagnostics(self, device):
        """The reference diagnostics report unweighted world-model losses.

        The JAX reference logs ``train/loss/{dyn,rep,con}`` before applying the
        loss coefficients, and ``dyn``/``rep`` therefore share a value (the two
        KL terms differ only in where the gradient is stopped). The Torch loss
        module returns the terms already weighted, so the diagnostics pass has
        to undo the coefficients for a term-by-term comparison to hold.
        """
        example = _load_example()
        reference_diagnostics = example["reference_diagnostics"]

        class _StubWithLatents(nn.Module):
            """Adds the continue head and the next-step latents to the stub."""

            def __init__(self_, base):
                super().__init__()
                self_.base = base
                self_.continue_head = nn.Linear(self.state_dim + self.rnn_hidden_dim, 1)

            def forward(self_, td):
                td = self_.base(td)
                features = torch.cat([td["state"], td["belief"]], dim=-1)
                td.set(("next", "continue_pred"), self_.continue_head(features))
                td.set(("next", "state"), td["state"])
                td.set(("next", "belief"), td["belief"])
                return td

        lambda_representation = 0.1
        lambda_continue = 2.0
        world_model = _StubWithLatents(self._create_world_model()).to(device)
        model_loss = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
            kl_mode="separate",
            lambda_dynamic=1.0,
            lambda_representation=lambda_representation,
            lambda_continue=lambda_continue,
        ).to(device)
        actor_loss = DreamerV3ActorLoss(
            # The reference policy is a Normal, whose analytic entropy the
            # diagnostics report; TanhNormal has no closed-form entropy.
            self._create_normal_actor_model_with_log_prob().to(device),
            self._create_value_model(out_features=self.num_reward_bins).to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
        )
        actor_loss.make_value_estimator(ValueEstimators.TDLambda, gamma=1.0, lmbda=0.95)
        actor_loss = actor_loss.to(device)
        value_loss = DreamerV3ValueLoss(
            actor_loss.value_model,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
            actor_loss=actor_loss,
        ).to(device)

        sample = self._create_world_model_data().to(device)
        sample["state"] = torch.randn_like(sample["state"])
        sample["belief"] = torch.randn_like(sample["belief"])
        return_state = (
            actor_loss.return_low.clone(),
            actor_loss.return_high.clone(),
        )
        diagnostics = reference_diagnostics(
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            sample=sample,
            state_dim=self.state_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            use_bfloat16=False,
            device=torch.device(device),
        )

        # The two KL terms have identical forward values, so reporting them
        # unweighted must make them agree despite the 0.1 representation weight.
        assert diagnostics["loss_dynamic"] == pytest.approx(
            diagnostics["loss_representation"], rel=1e-5
        )
        weighted, _ = model_loss(sample)
        assert diagnostics["loss_representation"] == pytest.approx(
            weighted["loss_model_representation"].item() / lambda_representation,
            rel=1e-5,
        )
        assert diagnostics["loss_continue"] == pytest.approx(
            weighted["loss_model_continue"].item() / lambda_continue, rel=1e-5
        )

        assert actor_loss.training
        torch.testing.assert_close(actor_loss.return_low, return_state[0])
        torch.testing.assert_close(actor_loss.return_high, return_state[1])
        for key in ("val", "ret", "adv", "adv_mag", "ent_action", "weight", "con"):
            assert key in diagnostics


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ExampleWalkerConfig(_DreamerV3Rig):
    """The walker config and the benchmark script that aggregates its runs."""

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

        # A run killed before it finished writes no summary record, so its
        # total step count is unknown and the aggregation must refuse rather
        # than silently average a short run against complete ones.
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

        # The preset must run the reference path, not the smoke defaults.
        config = OmegaConf.load(_EXAMPLE_DIR / "config_dmc_walker.yaml")
        assert config.optimization.train_ratio == 1024
        assert config.optimization.jax_behavior_policy_sync
        assert config.optimization.separate_policy_rng
        assert config.collector.count_reset_records
        assert config.replay_buffer.online


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
