# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DreamerV3 loss modules and RSSM components.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

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
)
from torch import nn

from torchrl.data import Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.model_based import DreamerActor
from torchrl.modules.models.model_based_v3 import (
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
    categorical_kl_balanced,
    symexp,
    symlog,
    two_hot_decode,
    two_hot_encode,
)
from torchrl.objectives.utils import ValueEstimators
from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv


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
        value_model = TensorDictModule(
            MLP(out_features=out_features, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["state_value"],
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

    @pytest.mark.parametrize("free_bits", [0.0, 0.5])
    def test_dreamer_v3_kl_balanced_gradients(self, device, free_bits):
        """Both prior_logits and posterior_logits must receive gradients (KL balancing).

        Run with free_bits=0 (no clamp) and free_bits=0.5 (typical) to confirm
        that the combined balanced loss can reach both sets of logits.
        """
        # Larger logits make the summed KL exceed any modest free-bits floor,
        # ensuring the clamp does not zero out the full gradient.
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
        """When the summed KL is below ``free_bits``, the loss is the floor and
        its gradient is zero.
        """
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

    def test_dreamer_v3_model_tensor_keys(self, device):
        world_model = self._create_world_model()
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        default_keys = {
            "reward": "reward",
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

    def test_dreamer_v3_value_invalid_loss_type(self, device):
        value_model = self._create_value_model()
        with pytest.raises(ValueError, match="symlog_mse.*two_hot"):
            DreamerV3ValueLoss(value_model, value_loss="bad_loss_type")

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
            num_reward_bins=self.num_reward_bins,
        )
        # state/belief in the default data are zeros, so the continue_head
        # weight gradient is always zero (W*0 = 0). Use non-zero inputs so
        # the BCE gradient reaches both weight and bias.
        base_td["state"] = torch.randn_like(base_td["state"])
        base_td["belief"] = torch.randn_like(base_td["belief"])
        # seed a mix of done / not-done so the BCE target is non-degenerate
        base_td["next", "done"][0, 0] = True
        loss_td, _ = loss_module(base_td)
        assert "loss_model_continue" in loss_td.keys()
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
                td.set(("next", "reward"), reward_pred)
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

    def test_default_bins_reward_space(self, device):
        """``bin_space="reward"`` reproduces the reference ``symexp_twohot`` grid."""
        n = 255
        symlog_bins = _default_bins(n, bin_space="symlog")
        reward_bins = _default_bins(n, bin_space="reward")
        # Same points, expressed in the two spaces.
        torch.testing.assert_close(symexp(symlog_bins), reward_bins, atol=1e-3, rtol=1e-4)
        # Exactly symmetric, so a uniform distribution decodes to exactly 0.
        torch.testing.assert_close(reward_bins, -reward_bins.flip(0))
        uniform = torch.zeros(1, n)
        assert two_hot_decode(uniform, reward_bins).abs().max() == 0.0
        # Two-hot round-trip in reward space is exact.
        x = torch.tensor([-3.7, 0.0, 0.42, 12.5])
        probs = two_hot_encode(x, reward_bins)
        torch.testing.assert_close((probs * reward_bins).sum(-1), x)
        with pytest.raises(ValueError, match="bin_space"):
            _default_bins(n, bin_space="nonsense")

    @pytest.mark.parametrize("bin_space", ["symlog", "reward"])
    def test_dreamer_v3_model_loss_bin_space(self, device, bin_space):
        """The reward two-hot target is built in the configured space."""
        world_model = self._create_world_model().to(device)
        loss_module = DreamerV3ModelLoss(
            world_model, num_reward_bins=self.num_reward_bins, bin_space=bin_space
        ).to(device)
        expected = _default_bins(self.num_reward_bins, bin_space=bin_space).to(device)
        torch.testing.assert_close(loss_module.reward_bins, expected)
        loss_td, _ = loss_module(self._create_world_model_data().to(device))
        assert torch.isfinite(loss_td["loss_model_reward"]).all()

    def test_dreamer_v3_model_loss_continue_uses_terminated(self, device):
        """The continue target is ``1 - terminated``: truncation must not count.

        A time-limit truncation carries ``done=True`` with ``terminated=False``;
        DreamerV3 trains the continue head against ``is_terminal``
        (``agent.py:174``), so such a step must still target "continue".
        """
        B, T = 2, 3

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
                    ("next", "continue_pred"), self_.continue_head(cat_in).squeeze(-1)
                )
                return td

        world_model = _StubWithContinue(self._create_world_model()).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model, lambda_continue=1.0, num_reward_bins=self.num_reward_bins
        )

        fixed = self._create_world_model_data().to(device)
        fixed["state"] = torch.randn_like(fixed["state"])
        fixed["belief"] = torch.randn_like(fixed["belief"])

        def continue_loss(done, terminated):
            td = fixed.copy()
            td["next", "done"] = torch.full_like(td["next", "done"], done)
            td["next", "terminated"] = torch.full_like(
                td["next", "terminated"], terminated
            )
            return loss_module(td)[0]["loss_model_continue"]

        none = continue_loss(False, False)
        truncated = continue_loss(True, False)
        terminated = continue_loss(True, True)
        torch.testing.assert_close(truncated, none)
        assert not torch.isclose(terminated, none)
        assert B == 2 and T == 3

    def test_dreamer_v3_model_loss_continue_target_scale(self, device):
        """The JAX contdisc path scales live continue targets by the horizon."""
        class _ConstantContinue(nn.Module):
            def __init__(self_, base):
                super().__init__()
                self_.base = base

            def forward(self_, td):
                td = self_.base(td)
                td.set(
                    ("next", "continue_pred"),
                    torch.ones_like(td["next", "done"], dtype=torch.float32),
                )
                return td

        data = self._create_world_model_data().to(device)
        data["next", "terminated"] = torch.zeros_like(
            data["next", "terminated"]
        )
        scale = 1.0 - 1.0 / 333.0
        loss_module = DreamerV3ModelLoss(
            _ConstantContinue(self._create_world_model()).to(device),
            lambda_continue=1.0,
            continue_target_scale=scale,
            num_reward_bins=self.num_reward_bins,
        ).to(device)
        loss = loss_module(data)[0]["loss_model_continue"]
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.ones_like(data["next", "done"], dtype=torch.float32).squeeze(-1),
            torch.full_like(
                data["next", "done"], scale, dtype=torch.float32
            ).squeeze(-1),
        ).unsqueeze(-1)
        torch.testing.assert_close(loss, expected)

        with pytest.raises(ValueError, match="continue_target_scale"):
            DreamerV3ModelLoss(
                self._create_world_model(), continue_target_scale=1.01
            )

    def test_rssm_rollout_v3_fast_path_matches_generic(self, device):
        """The tensor fast path must reproduce the generic TensorDict loop."""
        B, T, obs_embed_dim = 2, 5, 12
        prior_net = RSSMPriorV3(
            action_shape=torch.Size([self.action_dim]),
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=self.action_dim,
            jax_core=True,
            blocks=2,
            norm=True,
            img_layers=2,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
            norm=True,
        ).to(device)
        rollout = RSSMRolloutV3(
            TensorDictModule(
                prior_net,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_logits"),
                    ("next", "state"),
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
                posterior_net,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )
        assert rollout._fast_path

        def make_td():
            return TensorDict(
                {
                    "state": torch.zeros(B, T, self.state_dim, device=device),
                    "belief": torch.zeros(B, T, self.rnn_hidden_dim, device=device),
                    "action": torch.randn(B, T, self.action_dim, device=device),
                    "next": {
                        "encoded_latents": torch.randn(
                            B, T, obs_embed_dim, device=device
                        )
                    },
                },
                [B, T],
            )

        td = make_td()
        torch.manual_seed(0)
        fast = rollout(td.copy())
        rollout._fast_path = False
        torch.manual_seed(0)
        generic = rollout(td.copy())
        rollout._fast_path = True

        # Step 0 is fully determined by the (identical) inputs, so every
        # deterministic output must match bit-for-bit there. Later steps cannot
        # be compared this way: the generic path draws a prior sample per step
        # that the posterior discards, so the two paths walk the RNG stream
        # differently and the sampled states diverge.
        for key in (("next", "prior_logits"), ("next", "belief")):
            torch.testing.assert_close(
                fast.get(key)[:, 0], generic.get(key)[:, 0]
            )
        for key in (("next", "state"), ("next", "posterior_logits")):
            assert fast.get(key).shape == generic.get(key).shape

        # The invariant the fast path relies on: dropping the prior's discarded
        # sample leaves the belief and the prior logits untouched.
        state = torch.randn(B, self.state_dim, device=device)
        belief = torch.randn(B, self.rnn_hidden_dim, device=device)
        action = torch.randn(B, self.action_dim, device=device)
        logits_fast, belief_fast = prior_net.belief_and_logits(state, belief, action)
        logits_full, _, belief_full = prior_net(state, belief, action)
        torch.testing.assert_close(logits_fast, logits_full)
        torch.testing.assert_close(belief_fast, belief_full)

    @pytest.mark.slow
    def test_rssm_rollout_v3_compile_scan_matches_eager(self, device):
        """``compile_scan`` must not change the recurrence, forward or backward.

        Sampling is made deterministic for the comparison: ``torch.compile`` is
        free to reorder RNG draws, so a compiled rollout is only statistically
        equivalent to an eager one under real sampling, never bit-equal.
        """
        B, T, obs_embed_dim = 2, 3, 12
        prior_net = RSSMPriorV3(
            action_shape=torch.Size([self.action_dim]),
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=self.action_dim,
            jax_core=True,
            blocks=2,
            norm=True,
            img_layers=2,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
            norm=True,
        ).to(device)
        rollout = RSSMRolloutV3(
            TensorDictModule(
                prior_net,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_logits"),
                    ("next", "state"),
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
                posterior_net,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )

        def deterministic_categorical(logits, unimix=0.0):
            probs = torch.softmax(logits, dim=-1)
            if unimix:
                probs = (1 - unimix) * probs + unimix / probs.shape[-1]
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(-1, probs.argmax(-1, keepdim=True), 1.0)
            return probs + (one_hot - probs).detach()

        inputs = (
            torch.randn(B, self.state_dim, device=device),
            torch.randn(B, self.rnn_hidden_dim, device=device),
            torch.randn(B, T, self.action_dim, device=device),
            torch.randn(B, T, obs_embed_dim, device=device),
        )

        def run(fn):
            rollout.zero_grad(set_to_none=True)
            outs = fn(*inputs)
            # A scalar touching every output, so backward covers the full graph.
            sum(o.square().mean() for o in outs).backward()
            grads = {
                name: p.grad.detach().clone()
                for name, p in rollout.named_parameters()
                if p.grad is not None
            }
            return [o.detach().clone() for o in outs], grads

        with patch(
            "torchrl.modules.models.model_based_v3._straight_through_categorical",
            deterministic_categorical,
        ):
            eager_out, eager_grads = run(rollout._scan)
            rollout.compile_scan()
            compiled_out, compiled_grads = run(rollout._scan_fn)

        assert len(compiled_out) == len(eager_out) == 6
        for expected, actual in zip(eager_out, compiled_out):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        assert eager_grads and eager_grads.keys() == compiled_grads.keys()
        for name in eager_grads:
            torch.testing.assert_close(
                compiled_grads[name], eager_grads[name], rtol=1e-3, atol=1e-5
            )

    def test_rssm_prior_v3_img_layers(self, device):
        """``img_layers`` sets the prior head's hidden-layer count."""
        for img_layers in (1, 2, 3):
            prior_net = RSSMPriorV3(
                action_shape=torch.Size([self.action_dim]),
                hidden_dim=self.rnn_hidden_dim,
                rnn_hidden_dim=self.rnn_hidden_dim,
                num_categoricals=self.num_cats,
                num_classes=self.num_classes,
                action_dim=self.action_dim,
                norm=True,
                img_layers=img_layers,
            ).to(device)
            linears = [
                m
                for m in prior_net.rnn_to_prior_projector
                if isinstance(m, torch.nn.Linear)
            ]
            assert len(linears) == img_layers + 1
            logits, state, belief = prior_net(
                torch.zeros(2, self.state_dim, device=device),
                torch.zeros(2, self.rnn_hidden_dim, device=device),
                torch.randn(2, self.action_dim, device=device),
            )
            assert logits.shape == (2, self.num_cats, self.num_classes)

    def test_dreamer_v3_replay_value_loss(self, device):
        """``replay_value_loss`` matches a hand-rolled lambda return and trains the critic."""
        B, T = 2, 4
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(device)
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
            bin_space="reward",
        ).to(device)
        feat = TensorDict(
            {
                "state": torch.randn(B, T, self.state_dim, device=device),
                "belief": torch.randn(B, T, self.rnn_hidden_dim, device=device),
            },
            [B, T],
        )
        reward = torch.randn(B, T, device=device)
        done = torch.zeros(B, T, dtype=torch.bool, device=device)
        terminated = torch.zeros(B, T, dtype=torch.bool, device=device)
        boot = torch.randn(B, T, device=device)
        horizon, lam = 10.0, 0.95
        loss = loss_module.replay_value_loss(
            feat,
            next_reward=reward,
            next_done=done,
            next_terminated=terminated,
            bootstrap=boot,
            horizon=horizon,
            lam=lam,
        )
        assert loss.shape == ()
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in value_model.parameters()
        )

        # Reference lambda return (agent.py:482 lambda_return, boot != val).
        disc = 1.0 - 1.0 / horizon
        expected = torch.empty(B, T - 1, device=device)
        nxt = boot[:, -1]
        for t in reversed(range(T - 1)):
            nxt = reward[:, t] + disc * (
                (1.0 - lam) * boot[:, t + 1] + lam * nxt
            )
            expected[:, t] = nxt
        # Recompute what the method used, via a zero-slowreg / single-term path.
        with torch.no_grad():
            pred = value_model(feat.select("state", "belief"))["state_value"]
        manual = -(
            two_hot_encode(expected, loss_module.value_bins)
            * torch.log_softmax(pred[:, :-1], dim=-1)
        ).sum(-1)
        # No slow critic was passed, so there is no slowreg term.
        torch.testing.assert_close(loss, manual.mean(), rtol=1e-4, atol=1e-5)

    def test_dreamer_v3_terminated_cuts_replay_return(self, device):
        """A real termination must cut the replay lambda return; truncation must not."""
        B, T = 1, 3
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(device)
        loss_module = DreamerV3ValueLoss(
            value_model, value_loss="two_hot", num_value_bins=self.num_reward_bins
        ).to(device)
        feat = TensorDict(
            {
                "state": torch.randn(B, T, self.state_dim, device=device),
                "belief": torch.randn(B, T, self.rnn_hidden_dim, device=device),
            },
            [B, T],
        )
        kwargs = dict(
            next_reward=torch.zeros(B, T, device=device),
            bootstrap=torch.full((B, T), 5.0, device=device),
            horizon=10.0,
        )
        no_flag = loss_module.replay_value_loss(
            feat,
            next_done=torch.zeros(B, T, dtype=torch.bool, device=device),
            next_terminated=torch.zeros(B, T, dtype=torch.bool, device=device),
            **kwargs,
        )
        truncated = loss_module.replay_value_loss(
            feat,
            next_done=torch.ones(B, T, dtype=torch.bool, device=device),
            next_terminated=torch.zeros(B, T, dtype=torch.bool, device=device),
            **kwargs,
        )
        terminated = loss_module.replay_value_loss(
            feat,
            next_done=torch.ones(B, T, dtype=torch.bool, device=device),
            next_terminated=torch.ones(B, T, dtype=torch.bool, device=device),
            **kwargs,
        )
        # terminated -> live=0 -> the return collapses to the reward (0 here).
        assert not torch.isclose(terminated, no_flag)
        assert not torch.isclose(terminated, truncated)


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "--capture", "no", "--exitfirst"] + sys.argv[1:])
