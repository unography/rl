# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared fixtures for the DreamerV3 loss and example tests."""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl.data import Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based import DreamerActor
from torchrl.modules.models.model_based_v3 import RSSMPriorV3
from torchrl.modules.models.models import MLP
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv


class _DreamerV3Rig:
    """Shared sizes and module factories for the DreamerV3 loss tests."""

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


_has_hydra = importlib.util.find_spec("hydra") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None

_EXAMPLE_DIR = Path(__file__).parents[2] / "sota-implementations/dreamer_v3"


def _load_example() -> dict:
    """Return the example's namespace, including the helpers it imports."""
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        namespace = runpy.run_path(
            str(_EXAMPLE_DIR / "dreamer_v3.py"), run_name="dreamer_v3_test"
        )
        modules = [
            importlib.import_module(name)
            for name in ("dreamer_v3_utils", "dreamer_v3_replay", "dreamer_v3_agent")
        ]
    finally:
        sys.path.remove(str(_EXAMPLE_DIR))
    example: dict = {}
    for module in modules:
        example.update(vars(module))
    example.update(namespace)
    return example
