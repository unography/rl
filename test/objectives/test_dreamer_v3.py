# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the DreamerV3 loss modules.

The RSSM modules are covered in test/modules/test_dreamer_components.py and the
walker example in test_dreamer_v3_example.py and test_dreamer_v3_walker.py.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import argparse

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from _dreamer_v3_common import _DreamerV3Rig
from _objectives_common import LossModuleTestBase
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.modules import SymExpTwoHot
from torchrl.modules.models.model_based_v3 import (
    _straight_through_categorical,
    RSSMPosteriorV3,
    RSSMPriorV3,
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

_has_hydra = importlib.util.find_spec("hydra") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None


_EXAMPLE_DIR = Path(__file__).parents[2] / "sota-implementations/dreamer_v3"


def test_dreamer_v3_categorical_draw_matches_torch_distributions():
    """The fast draw is the distribution's draw, not merely its distribution."""
    logits = torch.randn(4, 3, 5)
    torch.manual_seed(0)
    state = _straight_through_categorical(logits, unimix=0.01)

    probs = torch.softmax(logits.float(), dim=-1)
    probs = 0.99 * probs + 0.01 / probs.shape[-1]
    torch.manual_seed(0)
    reference = torch.distributions.Categorical(probs=probs).sample()

    assert torch.equal(state.argmax(-1), reference)
    assert torch.equal(state.sum(-1), torch.ones_like(state.sum(-1)))


def test_dreamer_v3_categorical_autocast_matches_float32_reference():
    projection = nn.Linear(5, 6, bias=False)
    inputs = torch.linspace(-1.0, 1.0, 10).reshape(2, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = projection(inputs).reshape(2, 2, 3)
        torch.manual_seed(0)
        state = _straight_through_categorical(logits, unimix=0.01)

    assert logits.dtype == torch.bfloat16
    assert state.dtype == logits.dtype

    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_probs = torch.softmax(reference_logits.float(), dim=-1)
    reference_probs = 0.99 * reference_probs + 0.01 / reference_probs.shape[-1]
    torch.manual_seed(0)
    indices = torch.distributions.Categorical(probs=reference_probs).sample()
    reference_one_hot = torch.zeros_like(reference_probs)
    reference_one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    reference_state = (
        reference_probs + (reference_one_hot - reference_probs).detach()
    ).to(logits.dtype)
    torch.testing.assert_close(state, reference_state)

    weights = torch.linspace(-0.75, 1.25, state.numel()).reshape_as(state)
    (actual_grad,) = torch.autograd.grad((state.float() * weights).sum(), logits)
    (reference_grad,) = torch.autograd.grad(
        (reference_state.float() * weights).sum(), reference_logits
    )
    assert actual_grad.dtype == logits.dtype
    torch.testing.assert_close(actual_grad, reference_grad)


def test_dreamer_v3_categorical_kl_autocast_matches_float32_reference():
    posterior_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    prior_logits = torch.tensor(
        [[[0.0, 1.0, -1.0], [1.0, -0.5, 0.5]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dynamics, representation = categorical_kl_terms(
            posterior_logits,
            prior_logits,
            free_nats=0.0,
            unimix=0.01,
        )
        balanced = categorical_kl_balanced(
            posterior_logits,
            prior_logits,
            free_bits=0.0,
        )

    assert dynamics.dtype == torch.float32
    assert representation.dtype == torch.float32
    assert balanced.dtype == torch.float32

    reference_posterior_logits = posterior_logits.detach().float().requires_grad_(True)
    reference_prior_logits = prior_logits.detach().float().requires_grad_(True)
    reference_posterior = torch.softmax(reference_posterior_logits, dim=-1)
    reference_prior = torch.softmax(reference_prior_logits, dim=-1)
    reference_posterior = 0.99 * reference_posterior + 0.01 / 3
    reference_prior = 0.99 * reference_prior + 0.01 / 3
    reference_posterior_log = reference_posterior.log()
    reference_prior_log = reference_prior.log()
    reference_dynamics = reference_posterior.detach() * (
        reference_posterior_log.detach() - reference_prior_log
    )
    reference_dynamics = reference_dynamics.sum((-1, -2)).mean()
    reference_representation = reference_posterior * (
        reference_posterior_log - reference_prior_log.detach()
    )
    reference_representation = reference_representation.sum((-1, -2)).mean()
    torch.testing.assert_close(dynamics, reference_dynamics)
    torch.testing.assert_close(representation, reference_representation)

    actual_grads = torch.autograd.grad(
        dynamics + representation, (posterior_logits, prior_logits)
    )
    reference_grads = torch.autograd.grad(
        reference_dynamics + reference_representation,
        (reference_posterior_logits, reference_prior_logits),
    )
    for actual, reference in zip(actual_grads, reference_grads):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, reference.to(torch.bfloat16))


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3Numerics(_DreamerV3Rig):
    """symlog, two-hot and the balanced categorical KL."""

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

    def test_dreamer_v3_unimix_default_mixes(self, device):
        prior_logits = (
            torch.randn(2, self.num_cats, self.num_classes, device=device) * 3
        )
        posterior_logits = (
            torch.randn(2, self.num_cats, self.num_classes, device=device) * 3
        )
        default = categorical_kl_balanced(posterior_logits, prior_logits, free_bits=0.0)
        torch.testing.assert_close(
            default,
            categorical_kl_balanced(
                posterior_logits, prior_logits, free_bits=0.0, unimix=0.01
            ),
        )
        assert not torch.allclose(
            default,
            categorical_kl_balanced(
                posterior_logits, prior_logits, free_bits=0.0, unimix=0.0
            ),
        )

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


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ModelLoss(_DreamerV3Rig, LossModuleTestBase):  # type: ignore[misc]
    def test_reset_parameters_recursive(self, device):
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        self.reset_parameters_recursive_test(loss_fn)

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

    @pytest.mark.parametrize("reco_loss", ["l1", "l2"])
    @pytest.mark.parametrize("global_average", [False, True])
    def test_dreamer_v3_reference_reconstruction_loss(
        self, device, reco_loss, global_average
    ):
        batch_size = (2, 3)
        pixels = torch.tensor(
            [
                [[-3.0, -1.0, 0.5, 2.0]] * batch_size[1],
                [[4.0, -2.0, 0.0, 1.0]] * batch_size[1],
            ],
            device=device,
        )
        prediction = torch.tensor(
            [
                [[-0.5, -0.25, 0.75, 1.25]] * batch_size[1],
                [[1.5, -1.0, 0.25, 0.5]] * batch_size[1],
            ],
            device=device,
        )

        class FixedWorldModel(nn.Module):
            def forward(self_, tensordict):
                logits = torch.zeros(
                    *tensordict.batch_size,
                    self.num_cats,
                    self.num_classes,
                    device=tensordict.device,
                )
                reward_logits = torch.zeros(
                    *tensordict.batch_size,
                    self.num_reward_bins,
                    device=tensordict.device,
                )
                tensordict.set(("next", "prior_logits"), logits)
                tensordict.set(("next", "posterior_logits"), logits)
                # The decoder head predicts in symlog space and symexps back,
                # so the loss's symlog(prediction) recovers the raw head output.
                tensordict.set(("next", "reco_pixels"), symexp(prediction))
                tensordict.set(("next", "reward_logits"), reward_logits)
                return tensordict

        tensordict = TensorDict(
            {
                "next": {
                    "pixels": pixels,
                    "reward": torch.zeros(*batch_size, 1, device=device),
                }
            },
            batch_size,
            device=device,
        )
        loss_module = DreamerV3ModelLoss(
            FixedWorldModel(),
            num_reward_bins=self.num_reward_bins,
            reco_loss=reco_loss,
            global_average=global_average,
        )
        loss_td, _ = loss_module(tensordict)

        distance = symlog(pixels) - prediction
        expected = distance.square() if reco_loss == "l2" else distance.abs()
        if not global_average:
            expected = expected.sum(-1)
        expected = expected.mean().unsqueeze(-1)
        torch.testing.assert_close(loss_td["loss_model_reco"], expected)

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

    @pytest.mark.parametrize("detach_output", [True, False])
    def test_dreamer_v3_model_loss_detach_output(self, device, detach_output):
        """``detach_output=False`` keeps the returned features attached.

        The example backpropagates the replay value loss through them into the
        world model, which a detached output silently prevents.
        """
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
            detach_output=detach_output,
        )
        _, features = loss_module(tensordict)
        posterior = features.get(("next", "posterior_logits"))
        assert posterior.requires_grad is not detach_output
        if detach_output:
            return
        world_model.zero_grad(set_to_none=True)
        posterior.sum().backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in world_model.parameters()
        )

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


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ActorLoss(_DreamerV3Rig, LossModuleTestBase):  # type: ignore[misc]
    def test_reset_parameters_recursive(self, device):
        loss_fn = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
        )
        self.reset_parameters_recursive_test(loss_fn)

    def test_dreamer_v3_imagination_fast_path_matches_env_rollout(self, device):
        """Imagining without the env must be the same rollout, entry by entry."""
        tensordict = self._create_actor_data().to(device).reshape(-1)
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=4,
        )
        assert loss_module._fast_imagination

        def run(fast: bool):
            loss_module._fast_imagination = fast
            torch.manual_seed(0)
            return loss_module(tensordict.copy())

        env_losses, env_fake = run(False)
        fast_losses, fast_fake = run(True)

        env_keys = set(env_fake.keys(include_nested=True, leaves_only=True))
        assert env_keys == set(fast_fake.keys(include_nested=True, leaves_only=True))
        for key in env_keys:
            assert torch.equal(env_fake.get(key), fast_fake.get(key)), key
        for key in env_losses.keys():
            assert torch.equal(env_losses.get(key), fast_losses.get(key)), key

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
                return torch.full_like(state[..., :1], 0.5)

        continuation_model = TensorDictModule(
            _ConstantContinuation(),
            in_keys=["state", "belief"],
            out_keys=["continuation"],
        ).to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            value_model,
            self._create_mb_env().to(device),
            continuation_model=continuation_model,
            imagination_horizon=3,
            discount_loss=True,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=0.9, lmbda=0.5)

        reward = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
        value = torch.tensor([[[10.0], [20.0], [30.0]]], device=device)
        continuation = torch.full_like(reward, 0.5)
        torch.testing.assert_close(
            loss_module.lambda_target(reward, value, continuation),
            torch.tensor([[[5.5478125], [10.2125], [16.5]]], device=device),
        )

        _, fake_data = loss_module(self._create_actor_data().to(device).reshape(-1))
        # The reference weights the action at imagined feature t by
        # prod_{i=0}^{t} continuation_i, so the first factor is con_0 rather
        # than an undiscounted 1.0.
        expected_weight = torch.tensor([0.5, 0.225, 0.10125], device=device)
        torch.testing.assert_close(
            fake_data["discount_weight"][0, :, 0], expected_weight
        )
        torch.testing.assert_close(
            fake_data["next", "continuation"],
            torch.full_like(fake_data["next", "continuation"], 0.5),
        )

        value_loss = DreamerV3ValueLoss(
            value_model,
            discount_loss=True,
            actor_loss=loss_module,
        )
        with patch.object(
            value_loss,
            "_resolved_gamma",
            side_effect=AssertionError("provided weights must bypass gamma sync"),
        ):
            value_loss(fake_data.detach())

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

    def test_dreamer_v3_reference_policy_loss(self, device):
        actor_model = self._create_normal_actor_model_with_log_prob().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
            discount_loss=False,
            entropy_bonus=3e-4,
            use_reinforce=True,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        assert loss_module._return_normalization_quantiles.device == torch.device(
            device
        )
        assert "_return_normalization_quantiles" not in loss_module.state_dict()
        loss_module.return_low.fill_(-2.0)
        loss_module.return_high.fill_(8.0)
        loss_module.eval()

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        baseline_td = fake_data.select(*value_model.in_keys, strict=False)
        value_model(baseline_td)
        advantage = (
            fake_data["lambda_target"] - baseline_td["state_value"]
        ).detach() / 10.0
        policy_input = fake_data.select(*actor_model.in_keys, strict=False).detach()
        dist = actor_model.get_dist(policy_input)
        log_prob = _match_trailing_dim(
            dist.log_prob(fake_data["action"].detach()), advantage
        )
        entropy = _match_trailing_dim(dist.entropy(), advantage)
        expected = -(log_prob * advantage).mean() - 3e-4 * entropy.mean()
        torch.testing.assert_close(loss_td["loss_actor"], expected)

        loss_td["loss_actor"].backward()
        actor_grad = sum(
            parameter.grad.square().sum().item()
            for parameter in actor_model.parameters()
            if parameter.grad is not None
        )
        assert actor_grad > 0

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
        # The loss re-scores a stopped action from a stopped imagined state
        # rather than reusing the rollout's reparameterized log-probability.
        policy_input = fake_data.select(*actor_model.in_keys, strict=False).detach()
        log_prob = _match_trailing_dim(
            actor_model.get_dist(policy_input).log_prob(fake_data["action"].detach()),
            fake_data["lambda_target"],
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


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3ValueLoss(_DreamerV3Rig, LossModuleTestBase):  # type: ignore[misc]
    def test_reset_parameters_recursive(self, device):
        loss_fn = DreamerV3ValueLoss(self._create_value_model().to(device))
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_v3_value_loss_symlog_mse(self, device, discount_loss):
        tensordict = self._create_value_data().to(device)
        value_model = self._create_value_model(out_features=1).to(device)
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            discount_loss=discount_loss,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_value" in loss_td.keys()
        loss_td["loss_value"].backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert (
            grad_total > 0
        ), "All gradients are zero after value (symlog_mse) backward"

    @staticmethod
    def _replay_td(features, reward, done, terminated, bootstrap):
        replay = features.copy()
        replay.set("bootstrap", bootstrap)
        replay.set(("next", "reward"), reward)
        replay.set(("next", "done"), done)
        replay.set(("next", "terminated"), terminated)
        return replay

    @pytest.mark.parametrize(
        "value_loss_type,slow_critic_regularization",
        [("symlog_mse", 0.0), ("two_hot", 1.0)],
    )
    def test_dreamer_v3_replay_value_loss_episode_boundaries(
        self, device, value_loss_type, slow_critic_regularization
    ):
        """Non-zero ``done``/``terminated`` must drive the masks and the weight.

        ``done`` and ``terminated`` are set at *different* positions, so
        swapping them, or masking the wrong end, changes the result. The second
        parametrization is what the example runs.
        """
        batch, time_steps = 2, 5
        horizon, lmbda = 10.0, 0.5
        torch.manual_seed(0)
        state = torch.randn(
            batch, time_steps, self.state_dim, device=device, requires_grad=True
        )
        belief = torch.randn(
            batch, time_steps, self.rnn_hidden_dim, device=device, requires_grad=True
        )
        features = TensorDict({"state": state, "belief": belief}, [batch, time_steps])
        reward = torch.randn(batch, time_steps, device=device)
        bootstrap = torch.randn(batch, time_steps, device=device)
        done = torch.zeros(batch, time_steps, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        # Row 0: a truncation (done, not terminated) mid-sequence.
        done[0, 2] = True
        # Row 1: a true terminal, which is also a done.
        done[1, 3] = True
        terminated[1, 3] = True

        two_hot = value_loss_type == "two_hot"
        value_model = self._create_value_model(
            out_features=self.num_reward_bins if two_hot else 1
        ).to(device)
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss=value_loss_type,
            num_value_bins=self.num_reward_bins,
            slow_critic_regularization=slow_critic_regularization,
        ).to(device)
        actual = value_loss.replay_value_loss(
            self._replay_td(features, reward, done, terminated, bootstrap),
            horizon=horizon,
            lmbda=lmbda,
        )["loss_replay_value"]

        # Independent transcription of the reference lambda_return.
        discount = 1 - 1 / horizon
        live = (~terminated[..., 1:]).float() * discount
        continuation = (~done[..., 1:]).float() * lmbda
        intermediate = reward[..., 1:] + (1 - continuation) * live * bootstrap[..., 1:]
        next_return = bootstrap[..., -1]
        returns = []
        for time_index in reversed(range(time_steps - 1)):
            next_return = (
                intermediate[..., time_index]
                + live[..., time_index] * continuation[..., time_index] * next_return
            )
            returns.append(next_return)
        target = torch.stack(returns[::-1], -1)

        prediction_td = features.select(*value_model.in_keys, strict=False)
        with value_loss.value_model_params.to_module(
            value_model, preserve_module_state=False
        ):
            value_model(prediction_td)

        def step_loss(target):
            if two_hot:
                return two_hot_cross_entropy(
                    prediction_td["state_value_logits"][..., :-1, :],
                    target,
                    value_loss.value_bins,
                )
            return (
                symlog(prediction_td["state_value"][..., :-1, 0]) - symlog(target)
            ).square()

        per_step = step_loss(target)
        if slow_critic_regularization:
            slow_td = features.select(*value_model.in_keys, strict=False)
            with torch.no_grad(), value_loss.target_value_model_params.to_module(
                value_model, preserve_module_state=False
            ):
                value_model(slow_td)
            per_step = per_step + slow_critic_regularization * step_loss(
                slow_td["state_value"][..., :-1, 0]
            )
        weight = (~done[..., :-1]).to(per_step.dtype)
        torch.testing.assert_close(actual, (weight * per_step).mean())

        swapped = value_loss.replay_value_loss(
            self._replay_td(features, reward, terminated, done, bootstrap),
            horizon=horizon,
            lmbda=lmbda,
        )["loss_replay_value"]
        assert not torch.isclose(actual, swapped)
        zeros = torch.zeros_like(done)
        assert not torch.isclose(
            actual,
            value_loss.replay_value_loss(
                self._replay_td(features, reward, zeros, terminated, bootstrap),
                horizon=horizon,
                lmbda=lmbda,
            )["loss_replay_value"],
        )
        assert not torch.isclose(
            actual,
            value_loss.replay_value_loss(
                self._replay_td(features, reward, done, zeros, bootstrap),
                horizon=horizon,
                lmbda=lmbda,
            )["loss_replay_value"],
        )

        actual.backward()
        assert state.grad is not None and state.grad.abs().sum() > 0
        assert belief.grad is not None and belief.grad.abs().sum() > 0
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in value_loss.parameters()
        )

    def test_dreamer_v3_replay_value_loss_set_keys(self, device):
        """set_keys must redirect the replay entries, including nested keys."""
        batch, time_steps = 2, 4
        features = TensorDict(
            {
                "state": torch.randn(batch, time_steps, self.state_dim, device=device),
                "belief": torch.randn(
                    batch, time_steps, self.rnn_hidden_dim, device=device
                ),
            },
            [batch, time_steps],
        )
        reward = torch.randn(batch, time_steps, device=device)
        done = torch.zeros(batch, time_steps, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        bootstrap = torch.randn(batch, time_steps, device=device)

        value_model = self._create_value_model(out_features=1).to(device)
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            slow_critic_regularization=0.0,
        ).to(device)
        expected = value_loss.replay_value_loss(
            self._replay_td(features, reward, done, terminated, bootstrap)
        )["loss_replay_value"]

        renamed = features.copy()
        renamed.set("first_return", bootstrap)
        renamed.set(("next", "replay", "reward"), reward)
        renamed.set(("next", "replay", "done"), done)
        renamed.set(("next", "replay", "terminated"), terminated)
        value_loss.set_keys(
            reward=("replay", "reward"),
            done=("replay", "done"),
            terminated=("replay", "terminated"),
            bootstrap="first_return",
        )
        actual = value_loss.replay_value_loss(renamed)["loss_replay_value"]
        torch.testing.assert_close(actual, expected)

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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
