# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DreamerV3 loss modules and RSSM components.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _objectives_common import LossModuleTestBase
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl.data import (
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    RoundRobinWriter,
    Unbounded,
)
from torchrl.envs import EnvBase
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.model_based import (
    DreamerActor,
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.modules.models.models import MLP
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.dreamer_v3 import (
    _default_bins,
    _match_trailing_dim,
    _replay_value_target,
    categorical_kl_balanced,
    categorical_kl_terms,
    symexp,
    symlog,
    two_hot_cross_entropy,
    two_hot_decode,
    two_hot_encode,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv

_has_hydra = importlib.util.find_spec("hydra") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None
_has_dm_control = importlib.util.find_spec("dm_control") is not None

_EXAMPLE_DIR = Path(__file__).parents[2] / "sota-implementations/dreamer_v3"


def _load_example(monkeypatch, name: str) -> dict:
    """Run one module of the DreamerV3 example and return its globals."""
    pytest.importorskip("omegaconf")
    pytest.importorskip("hydra")
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    return runpy.run_path(_EXAMPLE_DIR / f"{name}.py", run_name=f"{name}_test")


def _small_image_config(height: int = 16, width: int = 16):
    """A tiny image configuration: two stages, 4x4 smallest map."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
    cfg.env.backend = "dm_control"
    cfg.env.observation_mode = "image"
    cfg.env.image_size = [height, width]
    cfg.networks.rnn_hidden_dim = 16
    cfg.networks.num_categoricals = 4
    cfg.networks.num_classes = 4
    cfg.networks.hidden_dim = 8
    cfg.networks.num_reward_bins = 16
    cfg.networks.image_depth = 2
    cfg.networks.image_depth_multipliers = [1, 2]
    cfg.networks.image_kernel_size = 3
    cfg.networks.decoder_spatial_blocks = 2
    return cfg, Unbounded((height, width, 3), dtype=torch.uint8)


@functools.cache
def _dmc_renders() -> bool:
    """Whether dm_control can render here: pixel tests need a GL context.

    Called inside test bodies, not at collection: a broken GL backend can
    take the process down.
    """
    if not _has_dm_control:
        return False
    try:
        from dm_control import suite

        suite.load("walker", "walk").physics.render(8, 8, camera_id=0)
    except Exception:
        return False
    return True


_requires_presets = pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf), reason="requires hydra and omegaconf"
)
_requires_dm_control = pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf and _has_dm_control),
    reason="requires hydra, omegaconf and dm_control",
)


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3(LossModuleTestBase):  # type: ignore[misc]
    img_size = (64, 64)
    # Compact sizes to keep tests fast
    num_cats = 4
    num_classes = 4
    state_dim = num_cats * num_classes  # 16
    rnn_hidden_dim = 8
    action_dim = 3
    num_reward_bins = 16  # small for tests; paper uses 255

    def _create_world_model_data(self):
        B, T = 2, 3
        return TensorDict(
            {
                "state": torch.zeros(B, T, self.state_dim),
                "belief": torch.zeros(B, T, self.rnn_hidden_dim),
                "pixels": torch.rand(B, T, 3, *self.img_size),
                "action": torch.randn(B, T, self.action_dim),
                "next": {
                    "pixels": torch.rand(B, T, 3, *self.img_size),
                    "reward": torch.randn(B, T, 1),
                    "done": torch.zeros(B, T, dtype=torch.bool),
                    "terminated": torch.zeros(B, T, dtype=torch.bool),
                },
            },
            [B, T],
        )

    def _create_actor_data(self):
        B, T = 2, 3
        return TensorDict(
            {
                "state": torch.randn(B, T, self.state_dim),
                "belief": torch.randn(B, T, self.rnn_hidden_dim),
                "reward": torch.randn(B, T, 1),
            },
            [B, T],
        )

    def _create_value_data(self):
        N = 6  # 2 * 3
        return TensorDict(
            {
                "state": torch.randn(N, self.state_dim),
                "belief": torch.randn(N, self.rnn_hidden_dim),
                "lambda_target": torch.randn(N, 1),
            },
            [N],
        )

    def _create_world_model(self, reward_two_hot=True):
        """Minimal stub world model that produces all keys DreamerV3ModelLoss expects."""

        class _StubWorldModel(nn.Module):
            def __init__(
                self_,
                num_cats,
                num_classes,
                rnn_hidden_dim,
                num_reward_bins,
                reward_two_hot,
            ):
                super().__init__()
                state_dim = num_cats * num_classes
                # pixel encoder → reco
                self_.encoder = nn.LazyConv2d(8, 4, stride=2)
                self_.decoder = nn.LazyConvTranspose2d(3, 4, stride=2)
                # prior / posterior MLP stubs
                self_.prior_net = nn.Linear(
                    state_dim + rnn_hidden_dim, num_cats * num_classes
                )
                self_.posterior_net = nn.LazyLinear(num_cats * num_classes)
                # reward head
                out_r = num_reward_bins if reward_two_hot else 1
                self_.reward_net = nn.LazyLinear(out_r)
                self_.reward_decoder = SymExpTwoHot(num_reward_bins)
                self_.num_cats = num_cats
                self_.num_classes = num_classes
                self_.reward_two_hot = reward_two_hot

            def forward(self_, tensordict):
                B, T = tensordict.shape
                state = tensordict["state"]  # [B, T, state_dim]
                belief = tensordict["belief"]  # [B, T, rnn_hidden]

                # prior logits
                prior_in = torch.cat([state, belief], dim=-1)
                prior_flat = self_.prior_net(prior_in)
                prior_logits = prior_flat.view(B, T, self_.num_cats, self_.num_classes)

                # posterior logits (lazy — accepts anything)
                post_flat = self_.posterior_net(prior_in)
                posterior_logits = post_flat.view(
                    B, T, self_.num_cats, self_.num_classes
                )

                # reco pixels (tiny decode — just needs right shape)
                next_pixels = tensordict["next", "pixels"]  # [B, T, 3, H, W]
                flat_pix = next_pixels.flatten(0, 1)  # [B*T, 3, H, W]
                enc = torch.relu(self_.encoder(flat_pix))
                reco_flat = torch.sigmoid(self_.decoder(enc))
                _, C, H, W = reco_flat.shape
                reco_pixels = reco_flat.view(B, T, C, H, W)

                # reward prediction
                reward_in = torch.cat([state, belief], dim=-1)
                reward_pred = self_.reward_net(reward_in)  # [B, T, out_r]

                tensordict.set(("next", "prior_logits"), prior_logits)
                tensordict.set(("next", "posterior_logits"), posterior_logits)
                tensordict.set(("next", "reco_pixels"), reco_pixels)
                if self_.reward_two_hot:
                    tensordict.set(("next", "reward_logits"), reward_pred)
                    reward_pred = self_.reward_decoder(reward_pred)
                tensordict.set(("next", "reward"), reward_pred)
                return tensordict

        stub = _StubWorldModel(
            self.num_cats,
            self.num_classes,
            self.rnn_hidden_dim,
            self.num_reward_bins,
            reward_two_hot,
        )
        # warm-up lazy layers
        with torch.no_grad():
            stub(self._create_world_model_data())
        return stub

    def _create_mb_env(self):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        default_dict = {
            "state": Unbounded(self.state_dim),
            "belief": Unbounded(self.rnn_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )
        rssm_prior = RSSMPriorV3(
            action_spec=mock_env.action_spec,
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=mock_env.action_spec.shape[0],
        )
        transition_model = SafeSequential(
            TensorDictModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=["_", "state", "belief"],
            )
        )
        reward_model = TensorDictModule(
            MLP(out_features=1, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["reward"],
        )
        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(transition_model, reward_model),
            prior_shape=torch.Size([self.state_dim]),
            belief_shape=torch.Size([self.rnn_hidden_dim]),
        )
        model_based_env.set_specs_from_env(mock_env)
        with torch.no_grad():
            model_based_env.rollout(3)
        return model_based_env

    def _create_actor_model(self):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        actor_module = DreamerActor(
            out_features=mock_env.action_spec.shape[0],
            depth=1,
            num_cells=8,
        )
        actor_model = ProbabilisticTensorDictSequential(
            TensorDictModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
            ),
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TanhNormal,
            ),
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, self.state_dim),
                    "belief": torch.randn(1, 2, self.rnn_hidden_dim),
                },
                batch_size=[1],
            )
            actor_model(td)
        return actor_model

    def _create_value_model(self, out_features=1):
        value_head = TensorDictModule(
            MLP(out_features=out_features, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["state_value" if out_features == 1 else "state_value_logits"],
        )
        if out_features == 1:
            value_model = value_head
        else:
            value_model = TensorDictSequential(
                value_head,
                TensorDictModule(
                    SymExpTwoHot(out_features),
                    in_keys=["state_value_logits"],
                    out_keys=["state_value"],
                ),
            )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, self.state_dim),
                    "belief": torch.randn(1, 2, self.rnn_hidden_dim),
                },
                batch_size=[1],
            )
            value_model(td)
        return value_model

    # ------------------------------------------------------------------ #
    # Required by LossModuleTestBase
    # ------------------------------------------------------------------ #

    def test_reset_parameters_recursive(self, device):
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        self.reset_parameters_recursive_test(loss_fn)

    # ------------------------------------------------------------------ #
    # Utility tests
    # ------------------------------------------------------------------ #

    def test_dreamer_v3_symlog_invertibility(self, device):
        x = torch.tensor([-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0], device=device)
        reconstructed = symexp(symlog(x))
        assert torch.allclose(
            reconstructed, x, atol=1e-4
        ), f"symexp(symlog(x)) ≠ x: {reconstructed}"

    def test_dreamer_v3_two_hot_roundtrip(self, device):
        bins = _default_bins(self.num_reward_bins).to(device)
        vals = torch.linspace(-15.0, 15.0, 9, device=device)
        encoded = two_hot_encode(vals, bins)
        # Each row must be a valid probability distribution
        assert torch.allclose(encoded.sum(-1), torch.ones(9, device=device), atol=1e-5)
        decoded = two_hot_decode(torch.log(encoded + 1e-8), bins)
        assert torch.allclose(
            decoded, vals, atol=0.5
        ), f"two_hot round-trip error too large: {(decoded - vals).abs().max()}"

    def test_dreamer_v3_two_hot_official_support(self, device):
        bins = _default_bins(5, device=device)
        expected = torch.tensor(
            [-485165184.0, -22025.4648, 0.0, 22025.4648, 485165184.0],
            device=device,
        )
        torch.testing.assert_close(bins, expected, rtol=1e-6, atol=1e-4)
        assert torch.equal(bins, -bins.flip(0))

        even_bins = _default_bins(4, device=device)
        assert torch.equal(even_bins, -even_bins.flip(0))
        expected_even = symexp(torch.linspace(-20, 20, 4, device=device))
        torch.testing.assert_close(even_bins, expected_even)

    def test_dreamer_v3_two_hot_golden_encode_loss(self, device):
        two_hot = SymExpTwoHot(5).to(device)
        midpoint = (two_hot.bins[1] + two_hot.bins[2]) / 2
        target = torch.stack(
            (
                two_hot.bins[0] - 1,
                midpoint,
                two_hot.bins[-1] + 1,
            )
        )
        encoded = two_hot.encode(target)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        torch.testing.assert_close(encoded, expected)

        logits = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], device=device)
        loss = two_hot_cross_entropy(logits, midpoint.reshape(1), two_hot.bins)
        torch.testing.assert_close(
            loss, torch.tensor([2.4519143], device=device), rtol=1e-6, atol=1e-6
        )

    def test_dreamer_v3_two_hot_golden_decode(self, device):
        two_hot = SymExpTwoHot(5).to(device)
        uniform = torch.zeros(3, 5, device=device)
        assert torch.equal(two_hot.decode(uniform), torch.zeros(3, device=device))

        logits = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], device=device)
        decoded = two_hot.decode(logits)
        torch.testing.assert_close(
            decoded,
            torch.tensor([-36122512.0], device=device),
            rtol=2e-6,
            atol=2.0,
        )

    def test_dreamer_v3_two_hot_module_state_and_compile(self, device):
        two_hot = SymExpTwoHot(5).to(device)
        logits = torch.linspace(-0.5, 0.5, 20, device=device).reshape(4, 5)
        expected = two_hot(logits)
        restored = SymExpTwoHot(5).to(device)
        restored.load_state_dict(two_hot.state_dict())
        torch.testing.assert_close(restored(logits), expected)

        compiled = torch.compile(restored, fullgraph=True)
        torch.testing.assert_close(compiled(logits), expected, rtol=1e-5, atol=1e-5)

    # ------------------------------------------------------------------ #
    # World model loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("reward_two_hot", [True, False])
    @pytest.mark.parametrize(
        "lambda_kl,lambda_reco,lambda_reward", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]
    )
    def test_dreamer_v3_model_loss_output_keys(
        self, device, reward_two_hot, lambda_kl, lambda_reco, lambda_reward
    ):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=reward_two_hot).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            lambda_kl=lambda_kl,
            lambda_reco=lambda_reco,
            lambda_reward=lambda_reward,
            reward_two_hot=reward_two_hot,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        for key in ("loss_model_kl", "loss_model_reco", "loss_model_reward"):
            assert key in loss_td.keys(), f"Missing {key}"
            assert loss_td[key].shape == torch.Size([1])

    def test_dreamer_v3_model_loss_backward(self, device):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        total_loss = sum(
            loss_td[k]
            for k in ("loss_model_kl", "loss_model_reco", "loss_model_reward")
        )
        total_loss.backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert grad_total > 0, "All gradients are zero after backward"
        for name, p in loss_module.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"

    def test_dreamer_v3_model_loss_sums_only_event_dims(self, device):
        batch_size, event_size = (2, 3), 4
        target = torch.ones(*batch_size, event_size, device=device)
        logits = torch.zeros(
            *batch_size, self.num_cats, self.num_classes, device=device
        )
        tensordict = TensorDict(
            {
                "next": {
                    "pixels": target,
                    "reco_pixels": torch.zeros_like(target),
                    "prior_logits": logits,
                    "posterior_logits": logits.clone(),
                    "reward": torch.zeros(*batch_size, 1, device=device),
                }
            },
            batch_size,
        )
        world_model = TensorDictModule(
            torch.zeros_like,
            in_keys=[("next", "true_reward")],
            out_keys=[("next", "reward")],
        )
        loss_td, _ = DreamerV3ModelLoss(
            world_model, reward_two_hot=False, free_bits=0.0, global_average=False
        )(tensordict)
        expected = event_size * symlog(torch.tensor(1.0, device=device)).square()
        torch.testing.assert_close(loss_td["loss_model_reco"].squeeze(), expected)

    @pytest.mark.parametrize("reco_loss", ["l2", "l1"])
    def test_dreamer_v3_model_loss_unit_interval_reco(self, device, reco_loss):
        batch_size = (2, 3)
        # One 2x2 RGB image per step, with the extreme and middle byte values.
        values = torch.tensor([0, 127, 255, 128], dtype=torch.uint8, device=device)
        pixels = values.reshape(1, 1, 2, 2, 1).expand(*batch_size, 2, 2, 3)
        reco = torch.full((*batch_size, 2, 2, 3), 0.25, device=device)
        reco.requires_grad_(True)
        logits = torch.zeros(
            *batch_size, self.num_cats, self.num_classes, device=device
        )
        tensordict = TensorDict(
            {
                "next": {
                    "pixels": pixels,
                    "reco_pixels": reco,
                    "prior_logits": logits,
                    "posterior_logits": logits.clone(),
                    "reward": torch.zeros(*batch_size, 1, device=device),
                }
            },
            batch_size,
        )
        world_model = TensorDictModule(
            torch.zeros_like,
            in_keys=[("next", "true_reward")],
            out_keys=[("next", "reward")],
        )
        loss_module = DreamerV3ModelLoss(
            world_model,
            reward_two_hot=False,
            free_bits=0.0,
            global_average=False,
            reco_loss=reco_loss,
            reco_space="unit_interval",
        )
        loss_td, _ = loss_module(tensordict)
        # Independent formula: no symlog, sum over H, W and C, mean over B, T.
        error = values.float() / 255 - 0.25
        per_pixel = error.square() if reco_loss == "l2" else error.abs()
        expected = per_pixel.sum() * 3
        torch.testing.assert_close(loss_td["loss_model_reco"].squeeze(), expected)
        # The target stays uint8 in the tensordict: no FP32 copy is stored.
        assert tensordict["next", "pixels"].dtype == torch.uint8
        loss_td["loss_model_reco"].sum().backward()
        expected_gradient = (
            2 * (0.25 - values.float() / 255) if reco_loss == "l2" else -error.sign()
        ) / (batch_size[0] * batch_size[1])
        torch.testing.assert_close(
            reco.grad[0, 0, :, :, 0].flatten(), expected_gradient
        )

        # A float target on [0, 1] is accepted as is.
        float_target = tensordict.clone()
        float_target.set(("next", "pixels"), pixels.float() / 255)
        float_loss, _ = loss_module(float_target)
        torch.testing.assert_close(
            float_loss["loss_model_reco"], loss_td["loss_model_reco"]
        )
        with pytest.raises(TypeError, match="uint8 or floating-point"):
            bad = tensordict.clone()
            bad.set(("next", "pixels"), pixels.int())
            loss_module(bad)
        with pytest.raises(TypeError, match="floating-point prediction"):
            bad = tensordict.clone()
            bad.set(("next", "reco_pixels"), pixels.clone())
            loss_module(bad)
        with pytest.raises(ValueError, match="same shape"):
            bad = tensordict.clone()
            bad.set(("next", "reco_pixels"), reco.detach()[..., :1])
            loss_module(bad)
        with pytest.raises(ValueError, match="reco_space"):
            DreamerV3ModelLoss(world_model, reco_space="pixels")

    @pytest.mark.parametrize("free_bits", [0.0, 0.5])
    def test_dreamer_v3_kl_balanced_gradients(self, device, free_bits):
        """Both prior_logits and posterior_logits must receive gradients (KL balancing).

        Run with free_bits=0 (no clamp) and free_bits=0.5 (typical) to confirm
        that gradient flow survives the per-categorical free-bits clamp.
        """
        # Larger logits make per-categorical KL exceed any modest free_bits,
        # ensuring the clamp does not zero out the gradient on every element.
        prior_logits = (
            torch.randn(2, 3, self.num_cats, self.num_classes, device=device) * 2.0
        ).requires_grad_(True)
        posterior_logits = (
            torch.randn(2, 3, self.num_cats, self.num_classes, device=device) * 2.0
        ).requires_grad_(True)
        kl = categorical_kl_balanced(
            posterior_logits, prior_logits, alpha=0.8, free_bits=free_bits
        )
        kl.backward()
        assert (
            prior_logits.grad is not None and prior_logits.grad.norm() > 0
        ), "prior_logits has no gradient - KL balancing broken"
        assert (
            posterior_logits.grad is not None and posterior_logits.grad.norm() > 0
        ), "posterior_logits has no gradient - KL balancing broken"

    def test_dreamer_v3_kl_balanced_free_bits_clamp(self, device):
        """When the per-categorical KL is below ``free_bits``, the loss is the
        clamp value and its gradient is zero. When most categoricals are above,
        the gradient must still flow (per-categorical clamp, not mean clamp)."""
        # Two near-identical distributions: KL is essentially zero and gets
        # clamped to free_bits => gradient must be exactly zero everywhere.
        base = torch.randn(2, 3, self.num_cats, self.num_classes, device=device)
        prior_logits = base.clone().requires_grad_(True)
        posterior_logits = base.clone().requires_grad_(True)
        free_bits = 0.5
        kl = categorical_kl_balanced(
            posterior_logits, prior_logits, alpha=0.8, free_bits=free_bits
        )
        # Loss equals the clamp floor: alpha * fb + (1 - alpha) * fb = fb.
        assert kl.item() == pytest.approx(free_bits, abs=1e-5)
        kl.backward()
        assert prior_logits.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)
        assert posterior_logits.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_dreamer_v3_reference_kl_fixture_and_gradients(self, device):
        posterior_logits = torch.tensor(
            [[[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]]],
            device=device,
            requires_grad=True,
        )
        prior_logits = torch.tensor(
            [[[0.0, 1.0, -1.0], [1.0, -0.5, 0.5]]],
            device=device,
            requires_grad=True,
        )

        dynamics, representation = categorical_kl_terms(
            posterior_logits,
            prior_logits,
            free_nats=0.0,
            unimix=0.01,
        )
        assert dynamics.item() == pytest.approx(1.9163513, abs=1e-6)
        assert representation.item() == pytest.approx(1.9163513, abs=1e-6)

        dynamics.backward(retain_graph=True)
        assert posterior_logits.grad is None
        assert prior_logits.grad is not None and prior_logits.grad.norm() > 0
        prior_logits.grad = None
        representation.backward()
        assert posterior_logits.grad is not None and posterior_logits.grad.norm() > 0
        assert prior_logits.grad is None

    def test_dreamer_v3_reference_kl_aggregates_before_free_nats(self, device):
        logits = torch.randn(3, 4, 8, device=device, requires_grad=True)
        dynamics, representation = categorical_kl_terms(
            logits,
            logits,
            free_nats=1.0,
            unimix=0.01,
        )
        assert dynamics.item() == pytest.approx(1.0)
        assert representation.item() == pytest.approx(1.0)

    def test_dreamer_v3_model_loss_reference_kl_keys(self, device):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            kl_mode="separate",
            lambda_dynamic=1.0,
            lambda_representation=0.1,
            unimix=0.01,
            free_bits=0.0,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_model_kl" not in loss_td.keys()
        assert "loss_model_dynamic" in loss_td.keys()
        assert "loss_model_representation" in loss_td.keys()
        dynamic = loss_td["loss_model_dynamic"]
        representation = loss_td["loss_model_representation"]
        assert dynamic.shape == torch.Size([1])
        assert representation.shape == torch.Size([1])
        (dynamic + representation).backward()

    def test_dreamer_v3_model_tensor_keys(self, device):
        world_model = self._create_world_model()
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        default_keys = {
            "reward": "reward",
            "reward_logits": "reward_logits",
            "true_reward": "true_reward",
            "prior_logits": "prior_logits",
            "posterior_logits": "posterior_logits",
            "pixels": "pixels",
            "reco_pixels": "reco_pixels",
        }
        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

    @pytest.mark.parametrize("detach_output", [True, False])
    def test_dreamer_v3_model_loss_detach_output(self, device, detach_output):
        world_model = self._create_world_model().to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
            detach_output=detach_output,
        )
        _, features = loss_module(self._create_world_model_data().to(device))
        posterior = features["next", "posterior_logits"]
        assert posterior.requires_grad is not detach_output

    # ------------------------------------------------------------------ #
    # Actor loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("imagination_horizon", [3, 5])
    @pytest.mark.parametrize("discount_loss", [True, False])
    @pytest.mark.parametrize(
        "td_est",
        [ValueEstimators.TD0, ValueEstimators.TD1, ValueEstimators.TDLambda, None],
    )
    def test_dreamer_v3_actor_loss(
        self, device, imagination_horizon, discount_loss, td_est
    ):
        tensordict = self._create_actor_data().to(device)
        mb_env = self._create_mb_env().to(device)
        actor_model = self._create_actor_model().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            mb_env,
            imagination_horizon=imagination_horizon,
            discount_loss=discount_loss,
        )
        if td_est is not None:
            loss_module.make_value_estimator(td_est)
        loss_td, fake_data = loss_module(tensordict.reshape(-1))
        assert "loss_actor" in loss_td.keys()
        assert loss_td["loss_actor"].ndim == 0 or loss_td["loss_actor"].numel() == 1
        loss_td["loss_actor"].backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert grad_total > 0, "All gradients are zero after actor backward"

    def test_dreamer_v3_continuation_lambda_and_weights(self, device):
        class _ConstantContinuation(nn.Module):
            def forward(self_, state, belief):
                return state[..., :1] * 0 + 0.5

        actor_model = self._create_actor_model_with_log_prob().to(device)
        continuation_model = TensorDictModule(
            _ConstantContinuation(),
            in_keys=["state", "belief"],
            out_keys=["continuation"],
        ).to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            continuation_model=continuation_model,
            imagination_horizon=3,
            discount_loss=True,
            entropy_bonus=0.0,
            use_reinforce=True,
            return_normalization=False,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=1.0, lmbda=0.5)

        reward = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
        value = torch.tensor([[[10.0], [20.0], [30.0]]], device=device)
        continuation = torch.full_like(reward, 0.5)
        torch.testing.assert_close(
            loss_module.lambda_target(reward, value, continuation),
            torch.tensor([[[6.375], [11.5], [18.0]]], device=device),
        )

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        # The initial state is weighted by its own continuation probability.
        expected_weight = torch.tensor([0.5, 0.25, 0.125], device=device)
        torch.testing.assert_close(
            fake_data["discount_weight"][0, :, 0], expected_weight
        )
        torch.testing.assert_close(
            fake_data["next", "continuation"],
            torch.full_like(fake_data["next", "continuation"], 0.5),
        )
        assert not fake_data["discount_weight"].requires_grad
        actor_parameters = tuple(actor_model.parameters())
        actual_gradients = torch.autograd.grad(
            loss_td["loss_actor"], actor_parameters, retain_graph=True
        )

        actor_inputs = fake_data.select(*actor_model.in_keys, strict=False).detach()
        distribution = actor_model.get_dist(actor_inputs)
        log_prob = distribution.log_prob(fake_data["action"].detach())
        log_prob = _match_trailing_dim(log_prob, fake_data["lambda_target"])
        baseline_td = fake_data.select(*value_model.in_keys, strict=False)
        value_model(baseline_td)
        advantage = (fake_data["lambda_target"] - baseline_td["state_value"]).detach()
        expected_loss = -(fake_data["discount_weight"] * log_prob * advantage).mean()
        expected_gradients = torch.autograd.grad(expected_loss, actor_parameters)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

        value_loss = DreamerV3ValueLoss(
            value_model,
            discount_loss=True,
            actor_loss=loss_module,
        )
        value_loss(fake_data.detach())

    # ------------------------------------------------------------------ #
    # Value loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("compiled", [False, True])
    def test_dreamer_v3_replay_value_target(self, device, compiled):
        reward = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=device)
        bootstrap = torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device)
        done = torch.zeros_like(reward, dtype=torch.bool)
        terminated = torch.zeros_like(done)

        target_fn = _replay_value_target
        if compiled:
            target_fn = torch.compile(target_fn, backend="eager", fullgraph=True)
        target = target_fn(
            reward,
            done,
            terminated,
            bootstrap,
            horizon=2.0,
            lmbda=0.5,
        )
        torch.testing.assert_close(
            target, torch.tensor([[9.8125, 15.25, 23.0]], device=device)
        )

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dreamer_v3_replay_value_loss_nested_keys(self, device, reduction):
        batch, time_steps = 2, 4
        state = torch.randn(
            batch,
            time_steps,
            self.state_dim,
            device=device,
            requires_grad=True,
        )
        replay = TensorDict(
            {
                "state": state,
                "belief": torch.randn(
                    batch, time_steps, self.rnn_hidden_dim, device=device
                ),
                "first_return": torch.randn(batch, time_steps, device=device),
                "next": {
                    "replay": {
                        "reward": torch.randn(batch, time_steps, device=device),
                        "done": torch.zeros(
                            batch, time_steps, dtype=torch.bool, device=device
                        ),
                        "terminated": torch.zeros(
                            batch, time_steps, dtype=torch.bool, device=device
                        ),
                    }
                },
            },
            [batch, time_steps],
        )
        value_loss = DreamerV3ValueLoss(
            self._create_value_model().to(device), reduction=reduction
        ).to(device)
        value_loss.set_keys(
            reward=("replay", "reward"),
            done=("replay", "done"),
            terminated=("replay", "terminated"),
            bootstrap="first_return",
        )

        loss = value_loss.replay_value_loss(replay)["loss_replay_value"]
        expected_shape = (batch, time_steps - 1) if reduction == "none" else ()
        assert loss.shape == expected_shape
        loss.sum().backward()
        assert state.grad is not None and state.grad.abs().sum() > 0

    @pytest.mark.parametrize("discount_loss", [True, False])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dreamer_v3_value_loss_symlog_mse(self, device, discount_loss, reduction):
        tensordict = self._create_value_data().to(device)
        value_model = self._create_value_model(out_features=1).to(device)
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            discount_loss=discount_loss,
            reduction=reduction,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_value" in loss_td.keys()
        expected_shape = tensordict.batch_size if reduction == "none" else ()
        assert loss_td["loss_value"].shape == expected_shape
        loss_td["loss_value"].sum().backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert (
            grad_total > 0
        ), "All gradients are zero after value (symlog_mse) backward"

    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_v3_value_loss_two_hot(self, device, discount_loss):
        tensordict = self._create_value_data().to(device)
        # Value model must output logits over bins
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            discount_loss=discount_loss,
            num_value_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_value" in loss_td.keys()
        loss_td["loss_value"].backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert grad_total > 0, "All gradients are zero after value (two_hot) backward"

    def test_dreamer_v3_categorical_value_exposes_decoded_value(self, device):
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        tensordict = self._create_value_data().to(device)
        value_model(tensordict)
        assert tensordict["state_value_logits"].shape[-1] == self.num_reward_bins
        assert tensordict["state_value"].shape[-1] == 1

        actor_loss = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
        )
        actor_loss.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, fake_data = actor_loss(
            self._create_actor_data().to(device).reshape(-1)
        )
        assert loss_td["loss_actor"].ndim == 0
        assert fake_data["lambda_target"].shape[-1] == 1

    def test_dreamer_v3_legacy_logits_keys_warn(self, device):
        class LegacyWorldModel(nn.Module):
            def __init__(self_, world_model):
                super().__init__()
                self_.world_model = world_model

            def forward(self_, tensordict):
                tensordict = self_.world_model(tensordict)
                logits = tensordict.pop(("next", "reward_logits"))
                tensordict.set(("next", "reward"), logits)
                return tensordict

        world_model = LegacyWorldModel(self._create_world_model()).to(device)
        model_loss = DreamerV3ModelLoss(
            world_model, num_reward_bins=self.num_reward_bins
        )
        with pytest.warns(DeprecationWarning, match="removed in v0.16"):
            model_loss(self._create_world_model_data().to(device))

        legacy_value = TensorDictModule(
            MLP(out_features=self.num_reward_bins, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["state_value"],
        ).to(device)
        value_loss = DreamerV3ValueLoss(
            legacy_value,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
        )
        with pytest.warns(DeprecationWarning, match="removed in v0.16"):
            value_loss(self._create_value_data().to(device))

    def test_dreamer_v3_nested_logits_keys(self, device):
        class NestedWorldModel(nn.Module):
            def __init__(self_, world_model):
                super().__init__()
                self_.world_model = world_model

            def forward(self_, tensordict):
                tensordict = self_.world_model(tensordict)
                tensordict.rename_key_(
                    ("next", "reward_logits"),
                    ("next", "predictions", "reward_logits"),
                )
                return tensordict

        model_loss = DreamerV3ModelLoss(
            NestedWorldModel(self._create_world_model()).to(device),
            num_reward_bins=self.num_reward_bins,
        )
        model_loss.set_keys(reward_logits=("predictions", "reward_logits"))
        model_loss(self._create_world_model_data().to(device))

        value_model = TensorDictSequential(
            TensorDictModule(
                MLP(out_features=self.num_reward_bins, depth=1, num_cells=8),
                in_keys=["state", "belief"],
                out_keys=[("predictions", "value_logits")],
            ),
            TensorDictModule(
                SymExpTwoHot(self.num_reward_bins),
                in_keys=[("predictions", "value_logits")],
                out_keys=[("predictions", "value")],
            ),
        ).to(device)
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
        )
        value_loss.set_keys(
            value=("predictions", "value"),
            value_logits=("predictions", "value_logits"),
        )
        value_loss(self._create_value_data().to(device))

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_shares_imagination_parameters(self, device, monkeypatch):
        from omegaconf import OmegaConf

        example = _load_example(monkeypatch, "train")
        cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
        cfg.networks.num_reward_bins = self.num_reward_bins
        (world_model, prior, reward_net, reward_decoder, continuation_net,) = example[
            "build_world_model"
        ](cfg=cfg, observation_spec=Unbounded(3), action_dim=self.action_dim)
        posterior = world_model[1].rssm_posterior.module
        imagination_model = example["build_imagination_model"](
            prior_net=prior,
            reward_net=reward_net,
            reward_decoder=reward_decoder,
        ).to(device)
        continuation_model = example["build_continuation_model"](
            continuation_net=continuation_net
        ).to(device)
        actor_model = example["build_actor"](cfg=cfg, action_dim=self.action_dim).to(
            device
        )
        real_actor = example["build_real_world_actor"](
            cfg=cfg,
            world_model=world_model,
            actor_model=actor_model,
        ).to(device)
        world_model = world_model.to(device)
        assert (
            prior.rnn_to_prior_projector[0].out_features
            == posterior.obs_rnn_to_post_projector[0].out_features
            == cfg.networks.hidden_dim
        )
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
        world_input = world_model(world_input)
        torch.testing.assert_close(
            world_input["next", "symlog_observation"], symlog(observation)
        )
        torch.testing.assert_close(
            symlog(world_input["next", "reco_pixels"]),
            world_input["next", "reco_symlog_observation"],
        )
        shared_parameters = tuple(prior.parameters()) + tuple(reward_net.parameters())
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
            for parameter in continuation_net.parameters()
        )

        observation = torch.tensor(
            [[0.25, -0.75, 1.5]], device=device, requires_grad=True
        )
        real_input = TensorDict(
            {
                "observation": observation,
                "state": torch.zeros(1, self.state_dim, device=device),
                "belief": torch.zeros(1, cfg.networks.rnn_hidden_dim, device=device),
                "previous_action": torch.zeros(1, self.action_dim, device=device),
                "is_init": torch.zeros(1, 1, dtype=torch.bool, device=device),
            },
            [1],
        )
        real_actor(real_input)
        observation_gradient = torch.autograd.grad(
            real_input["loc"].sum(), observation
        )[0]
        assert observation_gradient.abs().sum() > 0
        assert ("next", "belief") in real_input.keys(include_nested=True)

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

    def test_dreamer_v3_value_invalid_loss_type(self, device):
        value_model = self._create_value_model()
        with pytest.raises(ValueError, match="symlog_mse.*two_hot"):
            DreamerV3ValueLoss(value_model, value_loss="bad_loss_type")

    def test_dreamer_v3_slow_critic_regularization_and_update(self, device):
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            discount_loss=False,
            num_value_bins=self.num_reward_bins,
            slow_critic_regularization=1.0,
        ).to(device)
        updater = SoftUpdate(loss_module, tau=0.02)
        tensordict = self._create_value_data().to(device)

        online_td = tensordict.select(*value_model.in_keys, strict=False)
        with loss_module.value_model_params.to_module(
            loss_module.value_model, preserve_module_state=False
        ):
            loss_module.value_model(online_td)
        target_td = tensordict.select(*value_model.in_keys, strict=False)
        with torch.no_grad(), loss_module.target_value_model_params.to_module(
            loss_module.value_model, preserve_module_state=False
        ):
            loss_module.value_model(target_td)
        expected_slow_loss = two_hot_cross_entropy(
            online_td["state_value_logits"],
            target_td["state_value"].squeeze(-1),
            loss_module.value_bins,
        ).mean()

        loss_td, _ = loss_module(tensordict)
        torch.testing.assert_close(loss_td["value_slow_loss"], expected_slow_loss)
        loss_td["loss_value"].backward()
        assert any(
            parameter.grad is not None
            for parameter in loss_module.value_model_params.values(True, True)
            if parameter.requires_grad
        )
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in loss_module.target_value_model_params.values(True, True)
        )

        source = next(
            parameter
            for parameter in loss_module.value_model_params.values(True, True)
            if parameter.requires_grad
        )
        target = next(
            parameter
            for parameter in loss_module.target_value_model_params.values(True, True)
            if parameter.shape == source.shape
        )
        target_before = target.clone()
        with torch.no_grad():
            source.add_(1.0)
        updater.step()
        torch.testing.assert_close(target, target_before.lerp(source.detach(), 0.02))

    def test_dreamer_v3_slow_critic_checkpoint_and_online_bootstrap(self, device):
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        actor_loss = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            value_model,
            self._create_mb_env().to(device),
        )
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
            actor_loss=actor_loss,
            slow_critic_regularization=1.0,
        ).to(device)
        SoftUpdate(value_loss, tau=0.02)

        actor_parameters = tuple(actor_loss.__dict__["value_model"].parameters())
        online_parameters = tuple(value_loss.value_model_params.values(True, True))
        target_parameters = tuple(
            value_loss.target_value_model_params.values(True, True)
        )
        assert all(
            any(parameter is online for online in online_parameters)
            for parameter in actor_parameters
        )
        assert all(
            all(parameter is not target for target in target_parameters)
            for parameter in actor_parameters
        )

        checkpoint = {
            key: value.detach().clone()
            for key, value in value_loss.state_dict().items()
        }
        target_keys = [key for key in checkpoint if key.startswith("target_value")]
        assert target_keys
        expected_target = tuple(parameter.clone() for parameter in target_parameters)
        with torch.no_grad():
            for parameter in target_parameters:
                parameter.add_(10.0)
        value_loss.load_state_dict(checkpoint)
        for actual, expected in zip(
            value_loss.target_value_model_params.values(True, True),
            expected_target,
        ):
            torch.testing.assert_close(actual, expected)

    # ------------------------------------------------------------------ #
    # RSSM component tests
    # ------------------------------------------------------------------ #

    def test_rssm_posterior_v3_forward_shapes_and_grads(self, device):
        B = 4
        obs_embed_dim = 16
        posterior = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        belief = torch.randn(B, self.rnn_hidden_dim, device=device, requires_grad=True)
        obs_embed = torch.randn(B, obs_embed_dim, device=device, requires_grad=True)

        logits, state = posterior(belief, obs_embed)
        assert logits.shape == (B, self.num_cats, self.num_classes)
        assert state.shape == (B, self.state_dim)
        # one-hot forward: each categorical sums to 1
        state_grid = state.view(B, self.num_cats, self.num_classes)
        assert torch.allclose(
            state_grid.sum(-1), torch.ones(B, self.num_cats, device=device), atol=1e-5
        )

        # Straight-through: gradients must flow back through logits to belief/obs.
        # NOTE: ``state.sum()`` is mathematically constant w.r.t. the logits — every
        # row of the softmax inside the STE sums to 1, so any sum-reduction over
        # the full ``state`` has zero gradient through softmax (uniform incoming
        # gradient cancels exactly in the softmax Jacobian). Whether the resulting
        # belief/obs grads are exactly 0.0 or a tiny float-roundoff residue depends
        # on the runtime — leading to flakiness across Python/torch versions.
        # Use random per-element weights so the gradient signal through softmax
        # is non-degenerate.
        torch.manual_seed(0)
        weights = torch.randn_like(state)
        (state * weights).sum().backward()
        assert belief.grad is not None and belief.grad.abs().sum() > 0
        assert obs_embed.grad is not None and obs_embed.grad.abs().sum() > 0

    def test_rssm_rollout_v3_forward(self, device):
        B, T = 2, 4
        obs_embed_dim = 12
        action_dim = self.action_dim

        prior_net = RSSMPriorV3(
            action_shape=torch.Size([action_dim]),
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=action_dim,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        rssm_prior = TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "prior_logits"),
                ("next", "state"),
                ("next", "belief"),
            ],
        )
        rssm_posterior = TensorDictModule(
            posterior_net,
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[("next", "posterior_logits"), ("next", "state")],
        )
        rollout = RSSMRolloutV3(rssm_prior, rssm_posterior)

        td = TensorDict(
            {
                "state": torch.zeros(B, T, self.state_dim, device=device),
                "belief": torch.zeros(B, T, self.rnn_hidden_dim, device=device),
                "action": torch.randn(B, T, action_dim, device=device),
                "next": {
                    "encoded_latents": torch.randn(B, T, obs_embed_dim, device=device),
                },
            },
            [B, T],
        )
        out = rollout(td)
        assert out.shape == (B, T)
        prior_logits = out.get(("next", "prior_logits"))
        post_logits = out.get(("next", "posterior_logits"))
        assert prior_logits.shape == (B, T, self.num_cats, self.num_classes)
        assert post_logits.shape == (B, T, self.num_cats, self.num_classes)

        reset = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        reset[:, 2] = True
        td_a = td.clone().set("is_init", reset)
        td_b = td.clone().set("is_init", reset)
        td_b["action"][:, :2] = torch.randn_like(td_b["action"][:, :2])
        td_b["next", "encoded_latents"][:, :2] = torch.randn_like(
            td_b["next", "encoded_latents"][:, :2]
        )
        torch.manual_seed(0)
        out_a = rollout(td_a)
        torch.manual_seed(0)
        out_b = rollout(td_b)
        for key in (
            ("next", "prior_logits"),
            ("next", "posterior_logits"),
            ("next", "state"),
            ("next", "belief"),
        ):
            torch.testing.assert_close(out_a[key][:, 2:], out_b[key][:, 2:])

        td_c = td_a.clone()
        td_d = td_a.clone()
        td_c["action"][:, 2].zero_()
        td_d["action"][:, 2].fill_(1.0)
        torch.manual_seed(0)
        out_c = rollout(td_c)
        torch.manual_seed(0)
        out_d = rollout(td_d)
        torch.testing.assert_close(
            out_c["next", "prior_logits"][:, 2],
            out_d["next", "prior_logits"][:, 2],
        )

    # ------------------------------------------------------------------ #
    # Coverage for previously untested branches
    # ------------------------------------------------------------------ #

    def test_dreamer_v3_model_loss_reco_l1(self, device):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            reco_loss="l1",
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_model_reco" in loss_td.keys()
        loss_td["loss_model_reco"].backward()

    def test_dreamer_v3_model_loss_no_continue_default(self, device):
        """With ``lambda_continue=0`` (default), no ``loss_model_continue`` key is emitted."""
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_model_continue" not in loss_td.keys()

    def test_dreamer_v3_model_loss_continue(self, device):
        """Exercises the lambda_continue > 0 branch with a continue head."""
        B, T = 2, 3
        base_td = self._create_world_model_data().to(device)

        class _StubWithContinue(nn.Module):
            def __init__(self_, base):
                super().__init__()
                self_.base = base
                self_.continue_head = nn.Linear(
                    self.state_dim + self.rnn_hidden_dim, 1
                ).to(device)

            def forward(self_, td):
                td = self_.base(td)
                cat_in = torch.cat([td["state"], td["belief"]], dim=-1)
                td.set(
                    ("next", "continue_pred"),
                    self_.continue_head(cat_in).squeeze(-1),
                )
                return td

        world_model = _StubWithContinue(self._create_world_model()).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            lambda_continue=1.0,
            continue_target_scale=0.75,
            num_reward_bins=self.num_reward_bins,
        )
        # state/belief in the default data are zeros, so the continue_head
        # weight gradient is always zero (W*0 = 0). Use non-zero inputs so
        # the BCE gradient reaches both weight and bias.
        base_td["state"] = torch.randn_like(base_td["state"])
        base_td["belief"] = torch.randn_like(base_td["belief"])
        # seed a mix of done / not-done so the BCE target is non-degenerate
        base_td["next", "done"][0, 0] = True
        loss_td, model_out = loss_module(base_td)
        assert "loss_model_continue" in loss_td.keys()
        target = (~base_td["next", "terminated"]).float() * 0.75
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            model_out["next", "continue_pred"], target
        )
        torch.testing.assert_close(loss_td["loss_model_continue"].squeeze(), expected)
        loss_td["loss_model_continue"].backward()
        assert world_model.continue_head.weight.grad.abs().sum() > 0
        assert base_td.shape == (B, T)

    def _create_actor_model_with_log_prob(self):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        actor_module = DreamerActor(
            out_features=mock_env.action_spec.shape[0],
            depth=1,
            num_cells=8,
        )
        actor_model = ProbabilisticTensorDictSequential(
            TensorDictModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
            ),
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TanhNormal,
                return_log_prob=True,
                log_prob_key="action_log_prob",
            ),
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, self.state_dim),
                    "belief": torch.randn(1, 2, self.rnn_hidden_dim),
                },
                batch_size=[1],
            )
            actor_model(td)
        return actor_model

    def test_dreamer_v3_actor_loss_reinforce(self, device):
        """REINFORCE branch: log_prob * sg(advantage) path must be exercised."""
        tensordict = self._create_actor_data().to(device)
        mb_env = self._create_mb_env().to(device)
        actor_model = self._create_actor_model_with_log_prob().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            mb_env,
            imagination_horizon=3,
            use_reinforce=True,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, _ = loss_module(tensordict.reshape(-1))
        assert "loss_actor" in loss_td.keys()
        loss_td["loss_actor"].backward()
        actor_grad = sum(
            p.grad.pow(2).sum().item()
            for p in actor_model.parameters()
            if p.grad is not None
        )
        assert actor_grad > 0, "REINFORCE path produced no actor gradients"

    def test_dreamer_v3_reinforce_return_normalization(self, device):
        actor_model = self._create_actor_model_with_log_prob().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
            discount_loss=False,
            entropy_bonus=0.0,
            use_reinforce=True,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_module.return_low.fill_(-2.0)
        loss_module.return_high.fill_(8.0)
        loss_module.eval()

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        baseline_td = fake_data.select(*value_model.in_keys, strict=False)
        value_model(baseline_td)
        advantage = (fake_data["lambda_target"] - baseline_td["state_value"]).detach()
        log_prob = _match_trailing_dim(
            fake_data["action_log_prob"], fake_data["lambda_target"]
        )
        expected = -(log_prob * advantage / 10.0).mean()
        torch.testing.assert_close(loss_td["loss_actor"], expected)
        torch.testing.assert_close(
            loss_td["return_scale"], torch.tensor(10.0, device=device)
        )

        compiled_scale = torch.compile(loss_module._return_scale, fullgraph=True)
        torch.testing.assert_close(
            compiled_scale(fake_data["lambda_target"]),
            torch.tensor(10.0, device=device),
        )

    def test_dreamer_v3_reparam_return_normalization(self, device):
        """The reparameterization branch must divide the objective by the
        EMA return-percentile span, like the REINFORCE branch."""
        actor_model = self._create_actor_model().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
            discount_loss=False,
            entropy_bonus=0.0,
            use_reinforce=False,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_module.return_low.fill_(-2.0)
        loss_module.return_high.fill_(8.0)
        loss_module.eval()

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        expected = -(fake_data["lambda_target"] / 10.0).mean()
        torch.testing.assert_close(loss_td["loss_actor"], expected)
        torch.testing.assert_close(
            loss_td["return_scale"], torch.tensor(10.0, device=device)
        )

    def test_dreamer_v3_reparam_return_statistics_update(self, device):
        """Training-mode forward in the reparameterization branch must update
        the EMA return statistics."""
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            entropy_bonus=0.0,
            use_reinforce=False,
            return_normalization_rate=0.01,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        expected_low, expected_high = torch.quantile(
            fake_data["lambda_target"].detach(),
            torch.tensor([0.05, 0.95], device=device),
        )
        torch.testing.assert_close(loss_module.return_low, 0.01 * expected_low)
        torch.testing.assert_close(loss_module.return_high, 0.01 * expected_high)
        torch.testing.assert_close(
            loss_td["return_scale"],
            (loss_module.return_high - loss_module.return_low).clamp_min(1.0),
        )

    def test_dreamer_v3_return_statistics_checkpoint(self, device):
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model_with_log_prob().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            entropy_bonus=0.0,
            use_reinforce=True,
            return_normalization_rate=0.01,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        expected_low, expected_high = torch.quantile(
            fake_data["lambda_target"].detach(),
            torch.tensor([0.05, 0.95], device=device),
        )
        torch.testing.assert_close(loss_module.return_low, 0.01 * expected_low)
        torch.testing.assert_close(loss_module.return_high, 0.01 * expected_high)
        torch.testing.assert_close(loss_td["return_low"], loss_module.return_low)
        torch.testing.assert_close(loss_td["return_high"], loss_module.return_high)
        torch.testing.assert_close(
            loss_td["return_scale"],
            (loss_module.return_high - loss_module.return_low).clamp_min(1.0),
        )

        checkpoint = {
            key: value.detach().clone()
            for key, value in loss_module.state_dict().items()
        }
        expected_statistics = (
            loss_module.return_low.clone(),
            loss_module.return_high.clone(),
        )
        loss_module.return_low.zero_()
        loss_module.return_high.zero_()
        loss_module.load_state_dict(checkpoint)
        torch.testing.assert_close(loss_module.return_low, expected_statistics[0])
        torch.testing.assert_close(loss_module.return_high, expected_statistics[1])

        loss_module.eval()
        loss_module(self._create_actor_data().to(device).reshape(-1))
        torch.testing.assert_close(loss_module.return_low, expected_statistics[0])
        torch.testing.assert_close(loss_module.return_high, expected_statistics[1])

    def test_dreamer_v3_legacy_retnorm_checkpoint_migrates(self, device):
        """Checkpoints written before the retnorm refactor stored 0-dim
        ``return_low`` / ``return_high`` buffers; loading them must fill the
        ``retnorm`` statistics without strict-mode key errors."""
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model_with_log_prob().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            use_reinforce=True,
        ).to(device)
        legacy_checkpoint = {
            key: value.detach().clone()
            for key, value in loss_module.state_dict().items()
            if key not in ("retnorm.low", "retnorm.high")
        }
        legacy_checkpoint["return_low"] = torch.tensor(-2.5, device=device)
        legacy_checkpoint["return_high"] = torch.tensor(7.5, device=device)
        loss_module.load_state_dict(legacy_checkpoint)
        torch.testing.assert_close(
            loss_module.retnorm.low, torch.tensor([-2.5], device=device)
        )
        torch.testing.assert_close(
            loss_module.retnorm.high, torch.tensor([7.5], device=device)
        )

    def test_dreamer_v3_value_loss_sync_gamma(self, device):
        """sync_gamma_with_actor_loss must pull gamma from the actor's value estimator."""
        mb_env = self._create_mb_env().to(device)
        actor_model = self._create_actor_model().to(device)
        value_model = self._create_value_model().to(device)
        actor_loss = DreamerV3ActorLoss(actor_model, value_model, mb_env)
        actor_loss.make_value_estimator(ValueEstimators.TDLambda, gamma=0.95, lmbda=0.9)

        value_loss = DreamerV3ValueLoss(value_model, gamma=0.99)
        assert value_loss.gamma == 0.99
        value_loss.sync_gamma_with_actor_loss(actor_loss)
        assert value_loss.gamma == pytest.approx(0.95)

    # ------------------------------------------------------------------ #
    # End-to-end model-loss test with the real RSSM pair (no stub)
    # ------------------------------------------------------------------ #

    def test_dreamer_v3_model_loss_real_rssm(self, device):
        """DreamerV3ModelLoss against the real RSSMPriorV3 + RSSMPosteriorV3 wiring."""
        B, T = 2, 3
        obs_embed_dim = 16

        prior_net = RSSMPriorV3(
            action_shape=torch.Size([self.action_dim]),
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=self.action_dim,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        class _EndToEndWorldModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.encoder = nn.Sequential(
                    nn.LazyConv2d(8, 4, stride=2),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.LazyLinear(obs_embed_dim),
                )
                self_.decoder = nn.Sequential(
                    nn.LazyLinear(3 * 64 * 64),
                    nn.Unflatten(-1, (3, 64, 64)),
                )
                self_.reward_head = nn.LazyLinear(self.num_reward_bins)
                self_.reward_decoder = SymExpTwoHot(self.num_reward_bins)
                self_.prior = prior_net
                self_.posterior = posterior_net
                self_.num_cats = self.num_cats
                self_.num_classes = self.num_classes

            def forward(self_, td):
                B_, T_ = td.shape
                state = td["state"]
                belief = td["belief"]
                action = td["action"]

                prior_logits, _, next_belief = self_.prior(
                    state.flatten(0, 1), belief.flatten(0, 1), action.flatten(0, 1)
                )
                prior_logits = prior_logits.view(
                    B_, T_, self_.num_cats, self_.num_classes
                )
                next_belief = next_belief.view(B_, T_, -1)

                next_pixels = td["next", "pixels"]
                pix_flat = next_pixels.flatten(0, 1)
                obs_embed = self_.encoder(pix_flat)

                post_logits, post_state = self_.posterior(
                    next_belief.flatten(0, 1), obs_embed
                )
                post_logits = post_logits.view(
                    B_, T_, self_.num_cats, self_.num_classes
                )

                reco_flat = self_.decoder(post_state)
                reco_pixels = reco_flat.view(B_, T_, 3, 64, 64)

                reward_pred = self_.reward_head(post_state).view(
                    B_, T_, self.num_reward_bins
                )

                td.set(("next", "prior_logits"), prior_logits)
                td.set(("next", "posterior_logits"), post_logits)
                td.set(("next", "reco_pixels"), reco_pixels)
                td.set(("next", "reward_logits"), reward_pred)
                td.set(("next", "reward"), self_.reward_decoder(reward_pred))
                return td

        world_model = _EndToEndWorldModel().to(device)
        tensordict = self._create_world_model_data().to(device)
        # warm-up lazy layers
        with torch.no_grad():
            world_model(tensordict.clone())

        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        total = (
            loss_td["loss_model_kl"]
            + loss_td["loss_model_reco"]
            + loss_td["loss_model_reward"]
        )
        total.backward()
        # both the real prior and posterior nets must receive gradients
        prior_grad = sum(
            p.grad.pow(2).sum().item()
            for p in prior_net.parameters()
            if p.grad is not None
        )
        posterior_grad = sum(
            p.grad.pow(2).sum().item()
            for p in posterior_net.parameters()
            if p.grad is not None
        )
        assert prior_grad > 0, "Real prior received no gradient"
        assert posterior_grad > 0, "Real posterior received no gradient"
        assert B == 2 and T == 3

    def test_dreamer_v3_image_layouts(self, device, monkeypatch):
        """The data layouts a port of the reference networks could get wrong."""
        agent = _load_example(monkeypatch, "dreamer_v3_agent")
        cfg, spec = _small_image_config()
        torch.manual_seed(0)
        encoder = agent["_DreamerV3ImageEncoder"](cfg, spec.shape).to(device)
        decoder = agent["_DreamerV3ImageDecoder"](cfg, spec.shape, 16).to(device)

        # Channel RMS normalization at one pixel: channels [3, 4] have mean
        # square 12.5, so the outputs are [3, 4] / sqrt(12.5) times the scale.
        norm = agent["_DreamerV3ChannelRMSNorm"](2, 0.0).to(device)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 2.0]))
        value = torch.tensor([3.0, 4.0], device=device).reshape(1, 2, 1, 1)
        torch.testing.assert_close(
            norm(value).flatten(),
            torch.tensor([0.848528, 2.262742], device=device),
        )

        # Upsampling repeats each pixel twice along height and width.
        small = torch.arange(8.0, device=device).reshape(1, 2, 2, 2)
        torch.testing.assert_close(
            agent["_upsample_nearest"](small),
            torch.nn.functional.interpolate(small, scale_factor=2, mode="nearest"),
        )

        # The encoder flattens the last 4x4x4 map in HWC order: the flat index
        # of (h, w, c) is (h * 4 + w) * 4 + c. With identity-like weights the
        # activation stays where its input pixel was, so put one bright pixel
        # in the input and find it in the embedding.
        lit = {1}
        with torch.no_grad():
            for convolution in encoder.convolutions:
                convolution.weight.zero_()
                convolution.bias.zero_()
                center = convolution.kernel_size[0] // 2
                for channel in range(convolution.out_channels):
                    source = channel % convolution.in_channels
                    convolution.weight[channel, source, center, center] = 1.0
                lit = {
                    channel
                    for channel in range(convolution.out_channels)
                    if channel % convolution.in_channels in lit
                }
        pixels = torch.zeros(16, 16, 3, dtype=torch.uint8, device=device)
        pixels[9, 13, 1] = 255  # (h, w) = (2, 3) after two 2x2 poolings
        embedding = encoder(pixels)
        bright = (embedding > 0).nonzero().flatten().tolist()
        assert bright == sorted((2 * 4 + 3) * 4 + channel for channel in lit)

        # The decoder places block k of the deterministic projection at
        # channels [k * c, (k + 1) * c) of every (h, w) position.
        with torch.no_grad():
            decoder.stoch_projection.weight.zero_()
            decoder.stoch_projection.bias.zero_()
        block_size = 16 // decoder.blocks
        for block in range(decoder.blocks):
            belief = torch.zeros(1, 16, device=device)
            belief[0, block * block_size] = 1.0
            state = torch.zeros(1, 16, device=device)
            spatial = decoder.spatial_map(state, belief)
            projected = decoder.deter_projection(belief)[0]
            channels = decoder.space_channels // decoder.blocks
            chunk_size = 64 // decoder.blocks
            chunk = projected[block * chunk_size : (block + 1) * chunk_size]
            chunk = chunk.reshape(4, 4, channels).permute(2, 0, 1)
            torch.testing.assert_close(
                spatial[0, block * channels : (block + 1) * channels], chunk
            )

    def test_dreamer_v3_image_modules_dtypes_and_compile(self, device, monkeypatch):
        agent = _load_example(monkeypatch, "dreamer_v3_agent")
        cfg, spec = _small_image_config()
        torch.manual_seed(0)
        encoder = agent["_DreamerV3ImageEncoder"](cfg, spec.shape).to(device)
        decoder = agent["_DreamerV3ImageDecoder"](cfg, spec.shape, 16).to(device)
        assert encoder.out_features == 64
        # The convolutions use the reference initialization: zero biases and
        # weights truncated at two standard deviations of the fan-in scale.
        for convolution in (*encoder.convolutions, *decoder.convolutions):
            nominal = 1.1368 / convolution.weight[0].numel() ** 0.5
            assert convolution.weight.abs().max() <= 2 * nominal
            assert convolution.bias.abs().sum() == 0
        pixels = torch.randint(
            0, 256, (2, 3, 16, 16, 3), dtype=torch.uint8, device=device
        )
        embedding = encoder(pixels)
        assert embedding.shape == (2, 3, 64)
        assert embedding.dtype == torch.float32
        assert encoder(pixels[0, 0]).shape == (64,)
        state = torch.randn(2, 3, 16, device=device)
        belief = torch.randn(2, 3, 16, device=device)
        reco = decoder(state, belief)
        assert reco.shape == (2, 3, 16, 16, 3)
        assert reco.dtype == torch.float32
        assert reco.min() >= 0 and reco.max() <= 1
        assert decoder(state[0, 0], belief[0, 0]).shape == (16, 16, 3)

        # BF16 autocast keeps the shapes; the decoder output stays FP32.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            embedding_bf16 = encoder(pixels)
            reco_bf16 = decoder(state, belief)
        assert embedding_bf16.shape == embedding.shape
        assert embedding_bf16.dtype == torch.bfloat16
        assert reco_bf16.shape == reco.shape
        assert reco_bf16.dtype == torch.float32

        # No graph break in either module.
        compiled_encoder = torch.compile(encoder, fullgraph=True)
        compiled_decoder = torch.compile(decoder, fullgraph=True)
        torch.testing.assert_close(compiled_encoder(pixels), embedding)
        torch.testing.assert_close(compiled_decoder(state, belief), reco)

    def test_dreamer_v3_image_world_model_update(self, device, monkeypatch):
        agent = _load_example(monkeypatch, "dreamer_v3_agent")
        cfg, spec = _small_image_config()
        torch.manual_seed(0)
        world_model, *_ = agent["build_world_model"](
            cfg=cfg, observation_spec=spec, action_dim=2
        )
        world_model = world_model.to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=16,
            kl_mode="separate",
            free_bits=1.0,
            reco_space="unit_interval",
            detach_output=False,
        ).to(device)
        optimizer = agent["DreamerV3Optimizer"](
            world_model.parameters(), lr=1e-3, warmup_steps=0
        )
        batch_size, time = 2, 3
        sample = TensorDict(
            {
                "state": torch.zeros(batch_size, time, 16, device=device),
                "belief": torch.zeros(batch_size, time, 16, device=device),
                "action": torch.randn(batch_size, time, 2, device=device),
                "is_init": torch.zeros(
                    batch_size, time, 1, dtype=torch.bool, device=device
                ),
                "next": {
                    "pixels": torch.randint(
                        0,
                        256,
                        (batch_size, time, 16, 16, 3),
                        dtype=torch.uint8,
                        device=device,
                    ),
                    "reward": torch.randn(batch_size, time, 1, device=device),
                    "done": torch.zeros(
                        batch_size, time, 1, dtype=torch.bool, device=device
                    ),
                    "terminated": torch.zeros(
                        batch_size, time, 1, dtype=torch.bool, device=device
                    ),
                },
            },
            [batch_size, time],
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in world_model.named_parameters()
        }
        loss_td, model_out = loss_module(sample)
        assert model_out["next", "reco_pixels"].shape == (batch_size, time, 16, 16, 3)
        # The sample keeps its raw bytes: no FP32 copy of the target.
        assert sample["next", "pixels"].dtype == torch.uint8
        assert model_out["next", "pixels"].dtype == torch.uint8
        losses = torch.stack(
            [loss_td[key] for key in loss_td.keys() if key.startswith("loss_")]
        )
        assert torch.isfinite(losses).all()
        # An L2 error on [0, 1] summed over the image cannot exceed its size.
        assert 0 < loss_td["loss_model_reco"].item() <= 16 * 16 * 3
        total = (
            loss_td["loss_model_dynamic"]
            + loss_td["loss_model_representation"]
            + loss_td["loss_model_reco"]
            + loss_td["loss_model_reward"]
        )
        total.backward()
        optimizer.step()
        changed = {
            name
            for name, parameter in world_model.named_parameters()
            if not torch.equal(parameter, before[name])
        }
        # The world model is a sequence: encoder first, decoder third.
        encoder_names = {
            name
            for name in before
            if name.startswith("module.0.") and "convolutions" in name
        }
        decoder_names = {name for name in before if name.startswith("module.2.")}
        assert encoder_names and encoder_names <= changed
        assert decoder_names and decoder_names <= changed

    def test_dreamer_v3_image_policy_inference(self, device, monkeypatch):
        agent = _load_example(monkeypatch, "dreamer_v3_agent")
        cfg, spec = _small_image_config()
        torch.manual_seed(0)
        world_model, *_ = agent["build_world_model"](
            cfg=cfg, observation_spec=spec, action_dim=2
        )
        actor_model = agent["build_actor"](cfg=cfg, action_dim=2)
        policy = agent["build_real_world_actor"](
            cfg=cfg, world_model=world_model, actor_model=actor_model
        ).to(device)
        assert "pixels" in policy.in_keys
        assert "observation" not in policy.in_keys
        for batch in ([1], [4]):
            td = TensorDict(
                {
                    "pixels": torch.randint(
                        0, 256, (*batch, 16, 16, 3), dtype=torch.uint8, device=device
                    ),
                    "state": torch.zeros(*batch, 16, device=device),
                    "belief": torch.zeros(*batch, 16, device=device),
                    "previous_action": torch.zeros(*batch, 2, device=device),
                    "is_init": torch.ones(*batch, 1, dtype=torch.bool, device=device),
                },
                batch,
            )
            with torch.no_grad():
                policy(td)
            assert td["action"].shape == (*batch, 2)
            assert td["next", "belief"].shape == (*batch, 16)
            assert td["next", "state"].shape == (*batch, 16)
            assert td["pixels"].dtype == torch.uint8

    @pytest.mark.parametrize(
        "observation_key", ["observation", ("sensors", "proprio")]
    )
    def test_dreamer_v3_replay_record_builder_observation_key(
        self, device, monkeypatch, observation_key
    ):
        replay = _load_example(monkeypatch, "dreamer_v3_replay")
        builder = replay["DreamerV3ReplayRecordBuilder"](2, observation_key)
        num_streams, time = 2, 3
        # Integer-valued observations make the copies checkable byte for byte.
        observation = torch.arange(num_streams * time * 4, dtype=torch.uint8).reshape(
            num_streams, time, 4
        )
        next_observation = observation + 100
        is_init = torch.zeros(num_streams, time, 1, dtype=torch.bool, device=device)
        is_init[:, 1] = True
        data = TensorDict(
            {
                observation_key: observation.to(device),
                "action": torch.ones(num_streams, time, 2, device=device),
                "is_init": is_init,
                "state": torch.ones(num_streams, time, 4, device=device),
                "belief": torch.ones(num_streams, time, 3, device=device),
                "next": {
                    observation_key: next_observation.to(device),
                    "reward": torch.ones(num_streams, time, 1, device=device),
                    "done": torch.zeros(num_streams, time, 1, dtype=torch.bool),
                    "terminated": torch.zeros(num_streams, time, 1, dtype=torch.bool),
                },
            },
            [num_streams, time],
        ).to(device)
        records = builder(data)
        # The first batch drops its leading reset; the mid-batch one is kept.
        assert records.shape == (num_streams, time + 1)
        next_key = replay["unravel_key"](("next", observation_key))
        stored = records.get(next_key)
        assert stored.dtype == torch.uint8
        assert records.get("is_init").squeeze(-1).tolist() == [
            [False, True, False, False]
        ] * num_streams
        torch.testing.assert_close(stored[:, 0], next_observation[:, 0].to(device))
        # The reset record carries the observation the episode starts from.
        torch.testing.assert_close(stored[:, 1], observation[:, 1].to(device))
        torch.testing.assert_close(stored[:, 2], next_observation[:, 1].to(device))
        assert records.get("action")[:, 1].abs().sum() == 0
        assert records.get("action")[:, 2].sum() == num_streams * 2
        assert set(records.keys(include_nested=True, leaves_only=True)) == {
            "action",
            "is_init",
            "state",
            "belief",
            next_key,
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
            ("collector", "context_valid"),
        }

    def test_dreamer_v3_replay_keeps_uint8_images(self, device, monkeypatch):
        replay = _load_example(monkeypatch, "dreamer_v3_replay")
        num_streams, time, height, width = 2, 4, 8, 8
        builder = replay["DreamerV3ReplayRecordBuilder"](num_streams, "pixels")
        sampler = replay["DreamerV3ReplaySampler"](slice_len=3, online=False)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(max_size=64, ndim=2, device=device),
            dim_extend=1,
            writer=RoundRobinWriter(track_generations=True),
            sampler=sampler,
            batch_size=2 * 3,
        )

        def images(identifiers: torch.Tensor) -> torch.Tensor:
            # Every record gets one constant image, whose byte is its id.
            return (
                identifiers.to(torch.uint8)
                .reshape(num_streams, time, 1, 1, 1)
                .expand(num_streams, time, height, width, 3)
                .clone()
            )

        identifiers = torch.arange(1, num_streams * time + 1).reshape(
            num_streams, time
        )
        data = TensorDict(
            {
                "pixels": images(identifiers + 100),
                "action": identifiers.float().reshape(num_streams, time, 1),
                "is_init": torch.zeros(num_streams, time, 1, dtype=torch.bool),
                "state": torch.zeros(num_streams, time, 4),
                "belief": torch.zeros(num_streams, time, 3),
                "next": {
                    "pixels": images(identifiers),
                    "reward": torch.ones(num_streams, time, 1),
                    "done": torch.zeros(num_streams, time, 1, dtype=torch.bool),
                    "terminated": torch.zeros(num_streams, time, 1, dtype=torch.bool),
                },
            },
            [num_streams, time],
        ).to(device)
        records = builder(data)
        replay_indices = rb.extend(records)
        sampler.observe_extend(replay_indices, rb.storage)
        stored = rb[:]
        stored_keys = set(stored.keys(include_nested=True, leaves_only=True))
        assert ("next", "pixels") in stored_keys
        assert not any(
            "pixels" in key for key in stored_keys if key != ("next", "pixels")
        )
        assert stored["next", "pixels"].dtype == torch.uint8
        # One record: the image, 4 + 3 FP32 latents, one FP32 action, one
        # FP32 reward and four boolean flags. No leading reset was inserted.
        record_bytes = height * width * 3 + 7 * 4 + 4 + 4 + 4
        assert stored.bytes() == num_streams * time * record_bytes

        pipeline = replay["DreamerV3ReplayPipeline"]()
        sample, info = pipeline.take(rb)
        pixels = sample["next", "pixels"]
        assert pixels.dtype == torch.uint8
        assert pixels.shape == (6, height, width, 3)
        # Every sampled image is the constant image of its own record.
        for image, action in zip(pixels, sample["action"].flatten()):
            assert image.unique().tolist() == [int(action)]

        # The latent-context refresh changes only the context entries.
        before = rb[:].clone()
        pipeline.stage_context(
            info, torch.ones(2, 2, 4, device=device), torch.ones(2, 2, 3, device=device)
        )
        pipeline.apply_pending_context(rb)
        after = rb[:]
        assert torch.equal(after["next", "pixels"], before["next", "pixels"])
        assert torch.equal(after["action"], before["action"])
        assert not torch.equal(after["state"], before["state"])
        assert not torch.equal(after["belief"], before["belief"])


class _ConstantPixelsEnv(EnvBase):
    """An image environment whose renderer draws nothing, or fails."""

    def __init__(self, fail: bool = False):
        super().__init__(device="cpu")
        self.fail = fail
        self.observation_spec = Composite(
            pixels=Unbounded((16, 16, 3), dtype=torch.uint8)
        )
        self.action_spec = Unbounded(1)
        self.reward_spec = Unbounded(1)

    def _reset(self, tensordict=None, **kwargs):
        if self.fail:
            raise OSError("no GL context")
        return TensorDict({"pixels": torch.zeros(16, 16, 3, dtype=torch.uint8)}, [])

    def _step(self, tensordict):
        return TensorDict(
            {
                "pixels": torch.zeros(16, 16, 3, dtype=torch.uint8),
                "reward": torch.zeros(1),
                "done": torch.zeros(1, dtype=torch.bool),
            },
            [],
        )

    def _set_seed(self, seed):
        return seed


def test_dreamer_v3_rendered_frame_check(monkeypatch):
    agent = _load_example(monkeypatch, "dreamer_v3_agent")
    with pytest.raises(RuntimeError, match="constant frame"):
        agent["check_rendered_frame"](_ConstantPixelsEnv(), "pixels")
    with pytest.raises(RuntimeError, match="MUJOCO_GL"):
        agent["check_rendered_frame"](_ConstantPixelsEnv(fail=True), "pixels")


def test_dreamer_v3_env_and_image_config_validation(monkeypatch):
    from omegaconf import OmegaConf

    agent = _load_example(monkeypatch, "dreamer_v3_agent")
    cfg = OmegaConf.load(_EXAMPLE_DIR / "config.yaml")
    cfg.env.observation_mode = "audio"
    with pytest.raises(ValueError, match="observation_mode"):
        agent["make_env"](cfg, 0)
    cfg.env.observation_mode = "image"
    with pytest.raises(ValueError, match="backend=dm_control"):
        agent["make_env"](cfg, 0)
    cfg.env.backend = "dm_control"
    cfg.env.image_size = [64]
    with pytest.raises(ValueError, match="image_size"):
        agent["make_env"](cfg, 0)

    cfg, spec = _small_image_config()
    encoder = agent["_DreamerV3ImageEncoder"]
    with pytest.raises(ValueError, match="divisible by 4"):
        encoder(cfg, torch.Size([18, 16, 3]))
    with pytest.raises(ValueError, match="height, width"):
        encoder(cfg, torch.Size([3, 16, 16]))
    cfg.networks.decoder_spatial_blocks = 3
    with pytest.raises(ValueError, match="must divide"):
        encoder(cfg, spec.shape)
    cfg.networks.decoder_spatial_blocks = 2
    cfg.networks.image_kernel_size = 4
    with pytest.raises(ValueError, match="odd"):
        encoder(cfg, spec.shape)
    cfg.networks.image_kernel_size = 3
    cfg.networks.image_depth_multipliers = []
    with pytest.raises(ValueError, match="positive"):
        encoder(cfg, spec.shape)
    cfg.networks.image_depth_multipliers = [1, 2]
    cfg.env.observation_mode = "vector"
    cfg.networks.decoder_event_dims = [1, 2]
    with pytest.raises(ValueError, match="must sum"):
        agent["build_world_model"](cfg=cfg, observation_spec=Unbounded(4), action_dim=1)


@_requires_presets
@pytest.mark.parametrize(
    "config_name,protocol,model_size,task,threshold,observation,parameter_count",
    [
        (
            "config_dmc_walker",
            "dmc_proprio",
            "size1m",
            "walker/walk",
            900.0,
            (24,),
            640_867,
        ),
        (
            "config_dmc_cheetah",
            "dmc_proprio",
            "size1m",
            "cheetah/run",
            None,
            (17,),
            639_964,
        ),
        (
            "config_dmc_walker_vision",
            "dmc_vision",
            "size12m",
            "walker/walk",
            None,
            (64, 64, 3),
            10_494_158,
        ),
    ],
)
def test_dreamer_v3_presets(
    monkeypatch,
    config_name,
    protocol,
    model_size,
    task,
    threshold,
    observation,
    parameter_count,
):
    """Each preset composes, names its protocol and task, and builds its model."""
    benchmark = _load_example(monkeypatch, "benchmark")
    agent = _load_example(monkeypatch, "dreamer_v3_agent")
    cfg = benchmark["effective_config"](config_name)
    assert (cfg.protocol, cfg.model_size) == (protocol, model_size)
    assert benchmark["task_name"](cfg) == task
    assert cfg.collector.total_frames == 1_100_000
    settings = benchmark["benchmark_settings"](config_name)
    assert settings["minimum_final_median_return"] == threshold
    image = cfg.env.observation_mode == "image"
    assert agent["observation_key"](cfg) == ("pixels" if image else "observation")
    if image:
        # Nothing is evicted: the capacity exceeds the run plus the tail records.
        assert (
            cfg.replay_buffer.buffer_size
            >= cfg.collector.total_frames + cfg.collector.num_envs
        )
        assert cfg.replay_buffer.device == "cpu"
        assert cfg.optimization.train_ratio == 256
    else:
        assert cfg.optimization.train_ratio == 1024
    spec = Unbounded(observation, dtype=torch.uint8 if image else torch.float32)
    world_model, *_ = agent["build_world_model"](
        cfg=cfg, observation_spec=spec, action_dim=6
    )
    modules = (
        world_model,
        agent["build_actor"](cfg=cfg, action_dim=6),
        agent["build_value"](cfg=cfg),
    )
    # The count pins every model dimension of the selected size bundle.
    total = sum(
        parameter.numel() for module in modules for parameter in module.parameters()
    )
    assert total == parameter_count


@_requires_presets
def test_dreamer_v3_walker_preset_unchanged_by_refactor(monkeypatch):
    """The load-bearing walker values before the config groups, still effective."""
    from omegaconf import OmegaConf

    benchmark = _load_example(monkeypatch, "benchmark")
    walker = benchmark["effective_config"]("config_dmc_walker")
    expected = {
        "env": {
            "backend": "dm_control",
            "name": "walker",
            "task": "walk",
            "max_episode_steps": 1000,
        },
        "collector": {
            "num_envs": 16,
            "frames_per_batch": 16,
            "count_reset_records": True,
        },
        "replay_buffer": {
            "device": None,
            "buffer_size": 5_000_000,
            "batch_size": 16,
            "seq_len": 64,
        },
        "networks": {
            "rnn_hidden_dim": 512,
            "num_categoricals": 32,
            "num_classes": 4,
            "hidden_dim": 64,
            "decoder_event_dims": [1, 14, 9],
        },
        "optimization": {
            "deferred_policy_sync": True,
            "separate_policy_rng": True,
            "mixed_precision": True,
            "train_ratio": 1024,
        },
        "logger": {"train_every": 4096, "eval_every": 10000, "eval_episodes": 5},
        "benchmark": {"seeds": [0, 1, 2], "window_size": 50000},
    }
    for block, values in expected.items():
        for key, value in values.items():
            assert OmegaConf.to_container(walker[block])[key] == value, (block, key)
    # The cheetah preset shares the whole schedule through the protocol group.
    cheetah = benchmark["effective_config"]("config_dmc_cheetah")
    for block in ("collector", "replay_buffer", "optimization"):
        assert OmegaConf.to_container(cheetah[block]) == OmegaConf.to_container(
            walker[block]
        )
    # The model-size group is overridable from the command line.
    larger = benchmark["effective_config"]("config_dmc_walker", ["model_size=size12m"])
    assert (larger.networks.rnn_hidden_dim, larger.networks.image_depth) == (2048, 16)


@_requires_presets
def test_dreamer_v3_dmc_benchmark_aggregation(tmp_path, monkeypatch):
    benchmark = _load_example(monkeypatch, "benchmark")
    paths = []
    for seed, returns in enumerate(([1.0, 4.0], [3.0, 6.0], [2.0, 5.0])):
        path = tmp_path / f"seed_{seed}.jsonl"
        records = [
            {
                "type": "train_episode",
                "environment_steps": step,
                "score": score,
            }
            for step, score in zip((100, 200), returns)
        ]
        records.append(
            {
                "type": "summary",
                "seed": seed,
                "total_environment_steps": 200,
            }
        )
        path.write_text("\n".join(map(json.dumps, records)) + "\n")
        paths.append(path)

    summary = benchmark["aggregate_runs"](
        paths, window_size=100, config_name="config_test", task="walker/walk"
    )
    assert summary["environment_steps"] == [100, 200]
    assert summary["median_return"] == [2.0, 5.0]
    assert summary["lower_quartile_return"] == [1.5, 4.5]
    assert summary["upper_quartile_return"] == [2.5, 5.5]
    assert summary["config_name"] == "config_test"
    assert summary["task"] == "walker/walk"

    config = benchmark["effective_config"]("config_dmc_walker")
    assert config.env.name == "walker"
    assert config.env.task == "walk"
    assert config.collector.total_frames == 1_100_000
    assert config.optimization.train_ratio == 1024
    assert benchmark["task_name"](config) == "walker/walk"
    assert benchmark["default_output_dir"]("config_dmc_walker") == Path(
        "dmc_walker_runs"
    )


@_requires_presets
def test_dreamer_v3_benchmark_overrides(monkeypatch):
    benchmark = _load_example(monkeypatch, "benchmark")
    # A null threshold disables the check instead of falling back to walker's.
    settings = benchmark["benchmark_settings"](
        "config_dmc_walker", ["benchmark.minimum_final_median_return=null"]
    )
    assert settings["minimum_final_median_return"] is None
    with pytest.raises(ValueError, match="cannot be overridden"):
        benchmark["reject_reserved_overrides"](["env.seed=3"])
    assert benchmark["default_output_dir"]("config_dmc_cheetah") == Path(
        "dmc_cheetah_runs"
    )


def test_dreamer_v3_replay_memory_check(monkeypatch, tmp_path):
    replay = _load_example(monkeypatch, "dreamer_v3_replay")
    physical = replay["host_memory_bytes"](cgroup_files=())
    assert physical is not None and physical > 0
    # A cgroup file that holds a number bounds the limit; "max" does not.
    limit_file = tmp_path / "memory.max"
    limit_file.write_text("123456789\n")
    assert replay["host_memory_bytes"](cgroup_files=[limit_file]) == 123456789
    limit_file.write_text("max\n")
    assert replay["host_memory_bytes"](cgroup_files=[limit_file]) == physical
    record_bytes = 22568
    limit = 1000 * record_bytes
    assert replay["check_replay_capacity"](1000, record_bytes, None) == limit
    assert replay["check_replay_capacity"](800, record_bytes, limit) == (
        800 * record_bytes
    )
    with pytest.raises(ValueError, match="above 90%"):
        replay["check_replay_capacity"](1000, record_bytes, limit)


@_requires_dm_control
def test_dreamer_v3_cheetah_env(monkeypatch):
    benchmark = _load_example(monkeypatch, "benchmark")
    agent = _load_example(monkeypatch, "dreamer_v3_agent")
    cfg = benchmark["effective_config"]("config_dmc_cheetah")
    env = agent["make_env"](cfg, 0)
    try:
        # Sorted dm_control keys, position (8) then velocity (9), concatenated.
        base_spec = env.base_env.observation_spec
        assert [(key, base_spec[key].shape[0]) for key in sorted(base_spec)] == [
            ("position", 8),
            ("velocity", 9),
        ]
        assert env.observation_spec["observation"].shape == torch.Size([17])
        assert env.action_spec.shape == torch.Size([6])
        assert (env.action_spec.space.low == -1).all()
        assert (env.action_spec.space.high == 1).all()
        # The environment receives the clipped action; replay keeps the raw one.
        raw = TensorDict({"action": torch.full((6,), 3.0)}, [])
        assert (env.transform.inv(raw.clone())["action"] == 1.0).all()
        rollout = env.rollout(3)
        assert rollout["observation"].shape == (3, 17)
        assert rollout["observation"].dtype == torch.float32
    finally:
        env.close()


@_requires_dm_control
def test_dreamer_v3_walker_vision_env(monkeypatch):
    if not _dmc_renders():
        pytest.skip("dm_control cannot render here")
    benchmark = _load_example(monkeypatch, "benchmark")
    agent = _load_example(monkeypatch, "dreamer_v3_agent")
    cfg = benchmark["effective_config"]("config_dmc_walker_vision")
    env = agent["make_env"](cfg, 0)
    try:
        agent["check_rendered_frame"](env, "pixels")
        reset = env.reset()
        # Only the image and the control keys: no proprioceptive entry.
        assert set(reset.keys()) == {
            "pixels",
            "done",
            "terminated",
            "truncated",
            "step_count",
            "is_init",
        }
        pixels = reset["pixels"]
        assert env.observation_spec["pixels"].shape == torch.Size([64, 64, 3])
        assert pixels.shape == (64, 64, 3)
        assert pixels.dtype == torch.uint8
        assert pixels.float().std() > 0
        rollout = env.rollout(3)
        assert rollout["next", "pixels"].shape == (3, 64, 64, 3)
        assert rollout["next", "pixels"].dtype == torch.uint8
    finally:
        env.close()


@_requires_dm_control
@pytest.mark.parametrize(
    "config_name,observation_shape,protocol,model_size",
    [
        ("config_dmc_cheetah", [17], "dmc_proprio", "size1m"),
        ("config_dmc_walker_vision", [64, 64, 3], "dmc_vision", "size12m"),
    ],
)
def test_dreamer_v3_dmc_end_to_end(
    tmp_path, config_name, observation_shape, protocol, model_size
):
    """One small run of a DMC preset: collection, replay and updates."""
    if "vision" in config_name and not _dmc_renders():
        pytest.skip("dm_control cannot render here")
    metrics_path = tmp_path / "metrics.jsonl"
    # Two workers, ten-step episodes: 44 driver records hold 40 actions
    # and 4 reset records, in five batches of 8 actions.
    overrides = [
        f"--config-name={config_name}",
        f"hydra.run.dir={tmp_path / 'run'}",
        f"logger.metrics_jsonl={metrics_path}",
        "logger.output_plot=null",
        "optimization.device=cpu",
        "env.max_episode_steps=10",
        "collector.num_envs=2",
        "collector.frames_per_batch=8",
        "collector.total_frames=44",
        "replay_buffer.buffer_size=1000",
        "replay_buffer.batch_size=2",
        "replay_buffer.seq_len=4",
        "replay_buffer.warmup_factor=1",
        "optimization.train_ratio=null",
        "optimization.updates_per_batch=1",
        "logger.eval_every=20",
        "logger.eval_episodes=1",
        "logger.train_every=10",
        "networks.rnn_hidden_dim=16",
        "networks.hidden_dim=8",
        "networks.num_categoricals=4",
        "networks.num_classes=4",
        "networks.image_depth=2",
        "networks.encoder_layers=1",
        "networks.decoder_layers=1",
        "networks.actor_layers=1",
        "networks.value_layers=1",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[2]), env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, str(_EXAMPLE_DIR / "train.py"), *overrides],
        check=True,
        env=env,
        timeout=600,
    )
    records = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line
    ]
    summary = next(record for record in records if record["type"] == "summary")
    assert summary["observation_shape"] == observation_shape
    assert summary["action_dim"] == 6
    assert summary["total_environment_steps"] == 44
    assert summary["total_action_steps"] == 40
    assert summary["updates"] == 5
    assert (summary["protocol"], summary["model_size"]) == (protocol, model_size)
    assert summary["config"]["env"]["name"] == summary["environment"]
    train = [record for record in records if record["type"] == "train"]
    assert train
    # Images: the L2 error on [0, 1] cannot exceed the pixel count. Vectors:
    # symlog errors of DMC observations stay small per event dimension.
    # symlog errors of unit-scale DMC observations are below one per event.
    bound = 64 * 64 * 3 if "vision" in config_name else 17.0
    assert all(record["loss_reconstruction"] < bound for record in train)
    assert all(record["updates_in_window"] >= 1 for record in train)
    episodes = [record for record in records if record["type"] == "train_episode"]
    # Two workers finish their 10-step episodes at driver steps 22 and 44.
    assert sorted(record["environment_steps"] for record in episodes) == [
        21,
        22,
        43,
        44,
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
def test_dreamer_v3_dmc_reproduction_modes(tmp_path):
    repo_root = Path(__file__).parents[2]
    script = repo_root / "sota-implementations/dreamer_v3/reproduce_dmc_walker.sh"
    benchmark = repo_root / "sota-implementations/dreamer_v3/benchmark.py"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    fast = subprocess.run(
        ["bash", str(script), "--fast", "benchmark.seeds=[0]"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    ).stdout.splitlines()
    assert fast == [
        str(benchmark),
        "--output-dir",
        "dmc_walker_runs",
        "optimization.compile_rssm=scan",
        "optimization.rssm_scan_unroll=8",
        "benchmark.seeds=[0]",
    ]

    smoke = subprocess.run(
        ["bash", str(script), "--smoke"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    ).stdout.splitlines()
    assert smoke[:3] == [str(benchmark), "--output-dir", "dmc_walker_smoke"]
    assert "replay_buffer.buffer_size=400" in smoke
    assert "optimization.compile_rssm=null" in smoke
    assert "optimization.updates_per_batch=1" in smoke
    assert "optimization.train_ratio=null" in smoke

    incompatible = subprocess.run(
        ["bash", str(script), "--fast", "--smoke"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert incompatible.returncode == 2
    assert "mutually exclusive" in incompatible.stderr


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
