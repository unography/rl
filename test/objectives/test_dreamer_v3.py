# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DreamerV3 loss modules and RSSM components.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import copy
import importlib.util
import json
import runpy
from pathlib import Path
from unittest.mock import patch

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

from torchrl.data import LazyTensorStorage, ReplayBuffer, RoundRobinWriter, Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based import DreamerActor
from torchrl.modules.models.model_based_v3 import (
    _straight_through_categorical,
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
    uniform = torch.rand_like(reference_probs)
    tiny = torch.finfo(reference_probs.dtype).tiny
    gumbel = -torch.log(-torch.log(uniform.clamp_min(tiny)))
    indices = (reference_probs.log() + gumbel).argmax(-1, keepdim=True)
    reference_one_hot = torch.zeros_like(reference_probs)
    reference_one_hot.scatter_(-1, indices, 1.0)
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
        loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=1.0, lmbda=0.5)

        reward = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
        value = torch.tensor([[[10.0], [20.0], [30.0]]], device=device)
        continuation = torch.full_like(reward, 0.5)
        torch.testing.assert_close(
            loss_module.lambda_target(reward, value, continuation),
            torch.tensor([[[6.375], [11.5], [18.0]]], device=device),
        )

        _, fake_data = loss_module(self._create_actor_data().to(device).reshape(-1))
        # The reference weights the action at imagined feature t by
        # prod_{i=0}^{t} continuation_i, so the first factor is con_0 rather
        # than an undiscounted 1.0.
        expected_weight = torch.tensor([0.5, 0.25, 0.125], device=device)
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

    # ------------------------------------------------------------------ #
    # Value loss tests
    # ------------------------------------------------------------------ #

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

    def test_dreamer_v3_replay_value_loss(self, device):
        batch, time_steps = 2, 4
        state = torch.randn(
            batch, time_steps, self.state_dim, device=device, requires_grad=True
        )
        belief = torch.randn(
            batch,
            time_steps,
            self.rnn_hidden_dim,
            device=device,
            requires_grad=True,
        )
        features = TensorDict({"state": state, "belief": belief}, [batch, time_steps])
        reward = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [0.0, -1.0, 0.5, 2.0]], device=device
        )
        done = torch.zeros(batch, time_steps, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        bootstrap = torch.tensor(
            [[10.0, 20.0, 30.0, 40.0], [2.0, 3.0, 4.0, 5.0]], device=device
        )
        horizon = 10.0
        lmbda = 0.5

        value_model = self._create_value_model(out_features=1).to(device)
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            slow_critic_regularization=0.0,
        ).to(device)
        actual = value_loss.replay_value_loss(
            features,
            reward,
            done,
            terminated,
            bootstrap,
            horizon=horizon,
            lmbda=lmbda,
        )

        discount = 1 - 1 / horizon
        live = torch.full_like(reward[..., 1:], discount)
        continuation = torch.full_like(reward[..., 1:], lmbda)
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
        prediction = prediction_td["state_value"][..., :-1, 0]
        expected = (symlog(prediction) - symlog(target)).square().mean()
        torch.testing.assert_close(actual, expected)

        actual.backward()
        assert state.grad is not None and state.grad.abs().sum() > 0
        assert belief.grad is not None and belief.grad.abs().sum() > 0
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in value_loss.parameters()
        )

    def test_dreamer_v3_replay_value_loss_episode_boundaries(self, device):
        """Non-zero ``done``/``terminated`` must drive the masks and the weight.

        The other replay-value tests pass all-zero flags, so ``live``,
        ``continuation`` and the ``~done`` loss weight are never exercised where
        they differ. Here ``done`` and ``terminated`` are set at *different*
        positions, so swapping them, or masking the wrong end, changes the
        result.
        """
        batch, time_steps = 2, 5
        horizon, lmbda = 10.0, 0.5
        torch.manual_seed(0)
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
        bootstrap = torch.randn(batch, time_steps, device=device)
        done = torch.zeros(batch, time_steps, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        # Row 0: a truncation (done, not terminated) mid-sequence.
        done[0, 2] = True
        # Row 1: a true terminal, which is also a done.
        done[1, 3] = True
        terminated[1, 3] = True

        value_model = self._create_value_model(out_features=1).to(device)
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            slow_critic_regularization=0.0,
        ).to(device)
        actual = value_loss.replay_value_loss(
            features, reward, done, terminated, bootstrap, horizon=horizon, lmbda=lmbda
        )

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
        prediction = prediction_td["state_value"][..., :-1, 0]
        per_step = (symlog(prediction) - symlog(target)).square()
        weight = (~done[..., :-1]).to(per_step.dtype)
        torch.testing.assert_close(actual, (weight * per_step).mean())

        # The flags must actually matter: swapping done and terminated, or
        # dropping either, has to change the loss.
        swapped = value_loss.replay_value_loss(
            features, reward, terminated, done, bootstrap, horizon=horizon, lmbda=lmbda
        )
        assert not torch.isclose(actual, swapped)
        zeros = torch.zeros_like(done)
        assert not torch.isclose(
            actual,
            value_loss.replay_value_loss(
                features,
                reward,
                zeros,
                terminated,
                bootstrap,
                horizon=horizon,
                lmbda=lmbda,
            ),
        )
        assert not torch.isclose(
            actual,
            value_loss.replay_value_loss(
                features, reward, done, zeros, bootstrap, horizon=horizon, lmbda=lmbda
            ),
        )

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
    def test_dreamer_v3_sota_shares_imagination_parameters(self, device):
        from omegaconf import OmegaConf

        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_test",
        )
        cfg = OmegaConf.load(repo_root / "sota-implementations/dreamer_v3/config.yaml")
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
        actor_model = example["build_actor"](cfg=cfg, action_dim=self.action_dim).to(
            device
        )
        # Real transitions clear the collector root marker; only the explicit
        # terminal-to-reset replay edge is marked and resets the RSSM.
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

        parameter = nn.Parameter(torch.tensor([2.0, -1.0], device=device))
        optimizer = example["_DreamerV3Optimizer"](
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
        # The JAX linear warmup schedule applies zero LR to update zero.
        torch.testing.assert_close(parameter, initial)
        parameter.grad = torch.tensor([4.0, 3.0], device=device)
        optimizer.step()
        torch.testing.assert_close(parameter, initial - 0.05)
        assert optimizer.state[parameter]["rms"].dtype == torch.float32

        # Multiple differently shaped parameters share the foreach bucket but
        # retain independent AGC, RMS, and momentum statistics.
        parameters = [
            nn.Parameter(torch.tensor([2.0, -1.0], device=device)),
            nn.Parameter(torch.tensor([[0.5, -3.0, 1.5]], device=device)),
        ]
        gradients = [
            torch.tensor([4.0, -3.0], device=device),
            torch.tensor([[-2.0, 1.0, 5.0]], device=device),
        ]
        initials = [parameter.detach().clone() for parameter in parameters]
        optimizer = example["_DreamerV3Optimizer"](
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
    def test_dreamer_v3_sota_replay_prefetch_delay_and_overlap(self, device):
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_replay_pipeline_test",
        )

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
        pipeline = example["_DreamerV3ReplayPipeline"]()
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
        delayed = example["_DreamerV3ReplayPipeline"]()
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
    def test_dreamer_v3_sota_continuous_online_replay(self, device):
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_replay_test",
        )

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
        builder = example["_DreamerV3ReplayRecordBuilder"](num_streams)
        records = builder(collector_data)

        # The first reset is only the context record, as in JAX. Every later
        # reset gets one zero-action edge that targets and trains the reset obs.
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
        torch.testing.assert_close(
            records["collector", "replay_stream"],
            torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]),
        )

        sampler = example["_DreamerV3ReplaySampler"](
            slice_len=3,
            traj_key=("collector", "replay_stream"),
            cache_values=True,
            online=True,
        )
        replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=40, ndim=2, device=device),
            dim_extend=1,
            writer=RoundRobinWriter(track_generations=True),
            sampler=sampler,
            batch_size=6,
        )
        shifted_writer = example["_DreamerV3ShiftedReplayWriter"](num_streams)
        written = shifted_writer.extend(replay, sampler, records.to(device))
        assert written.shape == (10, 2)
        assert replay.storage.shape == torch.Size([5, num_streams])
        # Four completed shifted edges plus the mutable tail produce all three
        # starts of a three-record window. Omitting the initial context would
        # leave only two starts.
        assert replay.storage.shape[0] - sampler.slice_len + 1 == 3
        torch.testing.assert_close(
            replay.storage[:]["action"].transpose(0, 1),
            torch.cat([records["action"], torch.zeros(num_streams, 1, 1)], dim=1).to(
                device
            ),
        )
        assert not replay.storage[-1]["collector", "context_valid"].any()
        assert sampler.online_queue_size == 2

        sample, info = replay.sample(return_info=True)
        sample = sample.reshape(2, 3)
        sampled_index = torch.stack(info["index"], -1).reshape(2, 3, 2)
        assert sampled_index.shape == (2, 3, 2)
        assert sample.shape == (2, 3)
        # JAX has a one-batch background prefetch before its warmup gate. The
        # first batch is uniform and must leave the queued online blocks intact.
        assert sampler.online_queue_size == 2

        _, prefetched_info = replay.sample(return_info=True)
        prefetched_index = torch.stack(prefetched_info["index"], -1).reshape(2, 3, 2)
        torch.testing.assert_close(
            prefetched_index[0, :, 0],
            torch.tensor([1, 2, 3], device=prefetched_index.device),
        )
        torch.testing.assert_close(
            prefetched_index[1, :, 0],
            torch.tensor([1, 2, 3], device=prefetched_index.device),
        )
        assert (prefetched_index[0, :, 1] == 0).all()
        assert (prefetched_index[1, :, 1] == 1).all()
        assert sampler.online_queue_size == 0

        # A learner window ending at the mutable tail can infer that record's
        # posterior before its outgoing fields exist. Finalization must retain
        # that context and must not change the tail generation.
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

        # Once learning has started, JAX interleaves each worker add with one
        # update. Vectorized collection therefore admits one block per batch.
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

        _, second_info = replay.sample(return_info=True)
        second_index = torch.stack(second_info["index"], -1).reshape(2, 3, 2)
        torch.testing.assert_close(
            second_index[0, :, 0],
            torch.tensor([4, 5, 6], device=second_index.device),
        )
        assert (second_index[0, :, 1] == 0).all()
        assert sampler.online_queue_size == 1

        _, third_info = replay.sample(return_info=True)
        third_index = torch.stack(third_info["index"], -1).reshape(2, 3, 2)
        torch.testing.assert_close(
            third_index[0, :, 0],
            torch.tensor([4, 5, 6], device=third_index.device),
        )
        assert (third_index[0, :, 1] == 1).all()
        assert sampler.online_queue_size == 0

        uniform_sample, _ = replay.sample(return_info=True)
        assert uniform_sample.shape == (6,)

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_initial_context_cardinality(self, device):
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_replay_cardinality_test",
        )

        records = TensorDict(
            {
                "action": torch.tensor([[[1.0], [2.0]]], device=device),
                "is_init": torch.zeros(1, 2, 1, dtype=torch.bool, device=device),
                "state": torch.tensor([[[3.0], [4.0]]], device=device),
                "belief": torch.tensor([[[5.0], [6.0]]], device=device),
                "collector": {
                    "traj_ids": torch.zeros(1, 2, dtype=torch.long, device=device),
                    "replay_stream": torch.zeros(1, 2, dtype=torch.long, device=device),
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
        sampler = example["_DreamerV3ReplaySampler"](
            slice_len=3,
            traj_key=("collector", "replay_stream"),
            cache_values=True,
            online=False,
        )
        replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=10, device=device),
            writer=RoundRobinWriter(track_generations=True),
            sampler=sampler,
            batch_size=3,
            generator=torch.Generator().manual_seed(0),
        )
        shifted_writer = example["_DreamerV3ShiftedReplayWriter"](1)

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

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_real_world_actor(self, device):
        from omegaconf import OmegaConf

        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_policy_test",
        )
        cfg = OmegaConf.load(repo_root / "sota-implementations/dreamer_v3/config.yaml")
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

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_jax_behavior_policy_sync(self, device):
        del device
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_behavior_sync_test",
        )
        learner = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            learner.weight.zero_()
        behavior = copy.deepcopy(learner)
        sync = example["_DreamerV3BehaviorPolicySync"](learner, behavior)

        assert learner.weight is not behavior.weight
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

        stochastic = TensorDictModule(
            lambda value: torch.rand_like(value),
            in_keys=["input"],
            out_keys=["sample"],
        )
        seeded_policy = example["_DreamerV3SeededPolicy"](stochastic, seed=3)
        torch.manual_seed(123)
        global_state = torch.random.get_rng_state().clone()
        first = TensorDict({"input": torch.zeros(4)}, [])
        second = TensorDict({"input": torch.zeros(4)}, [])
        seeded_policy(first)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)
        seeded_policy(second)
        assert not torch.equal(first["sample"], second["sample"])
        repeated_policy = example["_DreamerV3SeededPolicy"](stochastic, seed=3)
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
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_counter_test",
        )
        action_budget = example["_collector_action_budget"]
        driver_step = example["_driver_step_for_action"]
        learner_updates = example["_learner_updates_for_records"]

        assert action_budget(1_000_000, 16, 1000) == 998_992
        assert action_budget(1_100_000, 16, 1000) == 1_098_896
        assert [driver_step(1000, index, 16, 1000) for index in range(16)] == list(
            range(16_001, 16_017)
        )
        assert [driver_step(2000, index, 16, 1000) for index in range(16)] == list(
            range(32_017, 32_033)
        )
        assert (
            learner_updates(
                16,
                1024,
                16,
                64,
                16,
                first_eligible_record=True,
            )
            == 1
        )
        assert learner_updates(16, 1024, 16, 64, 16) == 16

        for record_budget in (1_000_000, 1_100_000):
            actions = action_budget(record_budget, 16, 1000)
            actions_per_env = actions // 16
            resets_per_env = 1 + (actions_per_env - 1) // 1000
            assert actions + 16 * resets_per_env == record_budget

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
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_diagnostics_test",
        )
        reference_diagnostics = example["_reference_diagnostics"]

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

        # The pass is read-only: neither the training flag nor the
        # return-normalization EMA may move.
        assert actor_loss.training
        torch.testing.assert_close(actor_loss.return_low, return_state[0])
        torch.testing.assert_close(actor_loss.return_high, return_state[1])
        for key in ("val", "ret", "adv", "adv_mag", "ent_action", "weight", "con"):
            assert key in diagnostics

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_compiled_policy_retains_carrier_outputs(self, device):
        from omegaconf import OmegaConf

        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_compiled_policy_carrier_test",
        )
        cfg = OmegaConf.load(repo_root / "sota-implementations/dreamer_v3/config.yaml")
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
            # Collector carriers retain all policy outputs until they are
            # stacked. A CUDA graph-backed compiled head reuses its output
            # buffers on the second policy call and makes this stack fail.
            stacked = torch.stack([first, second], 0)
        assert stacked.shape == (2, 2)

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_dmc_parameter_parity(self, device):
        from omegaconf import OmegaConf

        del device
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_parameter_test",
        )
        cfg = OmegaConf.merge(
            OmegaConf.load(repo_root / "sota-implementations/dreamer_v3/config.yaml"),
            OmegaConf.load(
                repo_root / "sota-implementations/dreamer_v3/config_dmc_walker.yaml"
            ),
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
        assert sum(counts.values()) == 640_867
        # The decoder is a sequential: heads, then symexp back to observation
        # space so the loss's symlog(prediction) recovers the raw head output.
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
        repo_root = Path(__file__).parents[2]
        benchmark = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/benchmark.py",
            run_name="dreamer_v3_benchmark_test",
        )
        paths = []
        for seed, returns in enumerate(([1.0, 4.0], [3.0, 6.0], [2.0, 5.0])):
            path = tmp_path / f"seed_{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "total_environment_steps": 200,
                        "training_episode_steps": [50, 150],
                        "training_episode_returns": returns,
                    }
                )
            )
            paths.append(path)

        summary = benchmark["aggregate_runs"](paths, window_size=100)
        assert summary["environment_steps"] == [100, 200]
        assert summary["median_return"] == [2.0, 5.0]
        assert summary["lower_quartile_return"] == [1.5, 4.5]
        assert summary["upper_quartile_return"] == [2.5, 5.5]
        assert summary["window_size"] == 100

        config = OmegaConf.load(
            repo_root / "sota-implementations/dreamer_v3/config_dmc_walker.yaml"
        )
        base_config = OmegaConf.load(
            repo_root / "sota-implementations/dreamer_v3/config.yaml"
        )
        assert config.env.name == "walker"
        assert config.env.task == "walk"
        assert not config.env.use_seed
        assert config.collector.total_frames == 1_100_000
        assert config.collector.count_reset_records
        assert config.optimization.train_ratio == 1024
        assert config.optimization.jax_behavior_policy_sync
        assert config.optimization.separate_policy_rng
        assert config.replay_buffer.warmup_factor == 2
        assert config.replay_buffer.online
        assert config.logger.train_every == 4096

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

    def _create_normal_actor_model_with_log_prob(self):
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
                distribution_class=IndependentNormal,
                return_log_prob=True,
                log_prob_key="action_log_prob",
            ),
        )
        with torch.no_grad():
            actor_model(
                TensorDict(
                    {
                        "state": torch.randn(1, 2, self.state_dim),
                        "belief": torch.randn(1, 2, self.rnn_hidden_dim),
                    },
                    batch_size=[1],
                )
            )
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
