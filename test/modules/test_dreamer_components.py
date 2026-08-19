# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import pickle

import pytest
import torch
from packaging import version
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import unravel_key
from torchrl.data.tensor_specs import Bounded
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.model_based_v3 import (
    DreamerV3MLP,
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)

from torchrl.testing import get_default_devices


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [[], [3], [5]])
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.11.0"),
    reason="""Dreamer works with batches of null to 2 dimensions. Torch < 1.11
requires one-dimensional batches (for RNN and Conv nets for instance). If you'd like
to see torch < 1.11 supported for dreamer, please submit an issue.""",
)
class TestDreamerComponents:
    @pytest.mark.parametrize("out_features", [3, 5])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_actor(self, device, batch_size, temporal_size, out_features):
        actor = DreamerActor(
            out_features,
        ).to(device)
        emb = torch.randn(*batch_size, *temporal_size, 15, device=device)
        state = torch.randn(*batch_size, *temporal_size, 2, device=device)
        loc, scale = actor(emb, state)
        assert loc.shape == (*batch_size, *temporal_size, out_features)
        assert scale.shape == (*batch_size, *temporal_size, out_features)
        assert torch.all(scale > 0)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_encoder(self, device, temporal_size, batch_size, depth):
        encoder = ObsEncoder(channels=depth).to(device)
        obs = torch.randn(*batch_size, *temporal_size, 3, 64, 64, device=device)
        emb = encoder(obs)
        assert emb.shape == (*batch_size, *temporal_size, depth * 8 * 4)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_decoder(
        self, device, batch_size, temporal_size, depth, stoch_size, deter_size
    ):
        decoder = ObsDecoder(channels=depth).to(device)
        stoch_state = torch.randn(
            *batch_size, *temporal_size, stoch_size, device=device
        )
        det_state = torch.randn(*batch_size, *temporal_size, deter_size, device=device)
        obs = decoder(stoch_state, det_state)
        assert obs.shape == (*batch_size, *temporal_size, 3, 64, 64)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("out_channels", [1, 3])
    @pytest.mark.parametrize("stoch_size", [10])
    @pytest.mark.parametrize("deter_size", [20])
    def test_dreamer_decoder_out_channels(
        self, device, batch_size, depth, out_channels, stoch_size, deter_size
    ):
        decoder = ObsDecoder(channels=depth, out_channels=out_channels).to(device)
        stoch_state = torch.randn(*batch_size, stoch_size, device=device)
        det_state = torch.randn(*batch_size, deter_size, device=device)
        obs = decoder(stoch_state, det_state)
        assert obs.shape == (*batch_size, out_channels, 64, 64)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("action_size", [3, 6])
    def test_rssm_prior(self, device, batch_size, stoch_size, deter_size, action_size):
        action_spec = Bounded(shape=(action_size,), dtype=torch.float32, low=-1, high=1)
        rssm_prior = RSSMPrior(
            action_spec,
            hidden_dim=stoch_size,
            rnn_hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        state = torch.randn(*batch_size, deter_size, device=device)
        action = torch.randn(*batch_size, action_size, device=device)
        belief = torch.randn(*batch_size, stoch_size, device=device)
        prior_mean, prior_std, next_state, belief = rssm_prior(state, belief, action)
        assert prior_mean.shape == (*batch_size, deter_size)
        assert prior_std.shape == (*batch_size, deter_size)
        assert next_state.shape == (*batch_size, deter_size)
        assert belief.shape == (*batch_size, stoch_size)
        assert torch.all(prior_std > 0)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    def test_rssm_posterior(self, device, batch_size, stoch_size, deter_size):
        rssm_posterior = RSSMPosterior(
            hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        belief = torch.randn(*batch_size, stoch_size, device=device)
        obs_emb = torch.randn(*batch_size, 1024, device=device)
        # Init of lazy linears
        _ = rssm_posterior(belief.clone(), obs_emb.clone())

        torch.manual_seed(0)
        posterior_mean, posterior_std, next_state = rssm_posterior(
            belief.clone(), obs_emb.clone()
        )
        assert posterior_mean.shape == (*batch_size, deter_size)
        assert posterior_std.shape == (*batch_size, deter_size)
        assert next_state.shape == (*batch_size, deter_size)
        assert torch.all(posterior_std > 0)

        torch.manual_seed(0)
        posterior_mean_bis, posterior_std_bis, next_state_bis = rssm_posterior(
            belief.clone(), obs_emb.clone()
        )
        assert torch.allclose(posterior_mean, posterior_mean_bis)
        assert torch.allclose(posterior_std, posterior_std_bis)
        assert torch.allclose(next_state, next_state_bis)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("temporal_size", [2, 4])
    @pytest.mark.parametrize("action_size", [3, 6])
    def test_rssm_rollout(
        self, device, batch_size, temporal_size, stoch_size, deter_size, action_size
    ):
        action_spec = Bounded(shape=(action_size,), dtype=torch.float32, low=-1, high=1)
        rssm_prior = RSSMPrior(
            action_spec,
            hidden_dim=stoch_size,
            rnn_hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        rssm_posterior = RSSMPosterior(
            hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)

        rssm_rollout = RSSMRollout(
            SafeModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            SafeModule(
                rssm_posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[
                    ("next", "posterior_mean"),
                    ("next", "posterior_std"),
                    ("next", "state"),
                ],
            ),
        )

        state = torch.randn(*batch_size, temporal_size, deter_size, device=device)
        belief = torch.randn(*batch_size, temporal_size, stoch_size, device=device)
        action = torch.randn(*batch_size, temporal_size, action_size, device=device)
        obs_emb = torch.randn(*batch_size, temporal_size, 1024, device=device)

        tensordict = TensorDict(
            {
                "state": state.clone(),
                "action": action.clone(),
                "next": {
                    "encoded_latents": obs_emb.clone(),
                    "belief": belief.clone(),
                },
            },
            device=device,
            batch_size=torch.Size([*batch_size, temporal_size]),
        )
        ## Init of lazy linears
        _ = rssm_rollout(tensordict.clone())
        torch.manual_seed(0)
        rollout = rssm_rollout(tensordict)
        assert rollout["next", "prior_mean"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "prior_std"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "state"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "belief"].shape == (
            *batch_size,
            temporal_size,
            stoch_size,
        )
        assert rollout["next", "posterior_mean"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "posterior_std"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert torch.all(rollout["next", "prior_std"] > 0)
        assert torch.all(rollout["next", "posterior_std"] > 0)

        state[..., 1:, :] = 0
        belief[..., 1:, :] = 0
        # Only the first state is used for the prior. The rest are recomputed

        tensordict_bis = TensorDict(
            {
                "state": state.clone(),
                "action": action.clone(),
                "next": {"encoded_latents": obs_emb.clone(), "belief": belief.clone()},
            },
            device=device,
            batch_size=torch.Size([*batch_size, temporal_size]),
        )
        torch.manual_seed(0)
        rollout_bis = rssm_rollout(tensordict_bis)

        assert torch.allclose(
            rollout["next", "prior_mean"], rollout_bis["next", "prior_mean"]
        ), (rollout["next", "prior_mean"] - rollout_bis["next", "prior_mean"]).norm()
        assert torch.allclose(
            rollout["next", "prior_std"], rollout_bis["next", "prior_std"]
        )
        assert torch.allclose(rollout["next", "state"], rollout_bis["next", "state"])
        assert torch.allclose(rollout["next", "belief"], rollout_bis["next", "belief"])
        assert torch.allclose(
            rollout["next", "posterior_mean"], rollout_bis["next", "posterior_mean"]
        )
        assert torch.allclose(
            rollout["next", "posterior_std"], rollout_bis["next", "posterior_std"]
        )


class TestDreamerV3Components:
    def test_mlp_output_scale_and_multiple_inputs(self):
        module = DreamerV3MLP(
            6,
            4,
            depth=2,
            num_cells=8,
            outscale=0.0,
        )
        output = module(torch.randn(3, 2), torch.randn(3, 4))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_mlp_without_output_projection(self):
        module = DreamerV3MLP(
            6,
            None,
            depth=3,
            num_cells=8,
        )
        output = module(torch.randn(4, 6))
        assert output.shape == (4, 8)
        assert (
            sum(isinstance(child, torch.nn.Linear) for child in module.modules()) == 3
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_block_gru_golden_values(self, device):
        """Pin the block-GRU forward against accidental change.

        The expected tensors are golden values generated from this
        implementation, not transcribed from the reference DreamerV3. They
        catch unintended changes to the arithmetic; they cannot detect
        divergence from the reference, which
        ``test_dreamer_v3_dmc_parameter_parity`` pins at the architecture level.
        """
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=4,
            rnn_hidden_dim=4,
            num_categoricals=2,
            num_classes=2,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
            num_layers=1,
            prior_num_layers=1,
            device=device,
        )
        with torch.no_grad():
            for parameter in prior.parameters():
                parameter.copy_(
                    torch.linspace(
                        -0.2,
                        0.2,
                        parameter.numel(),
                        device=device,
                    ).reshape_as(parameter)
                )
        state = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device)
        belief = torch.tensor([[0.1, -0.2, 0.3, -0.4]], device=device)
        action = torch.tensor([[2.0, -4.0]], device=device)

        torch.manual_seed(0)
        logits, sampled_state, next_belief = prior(state, belief, action)

        torch.testing.assert_close(
            logits,
            torch.tensor(
                [[[-0.2538432, -0.0850009], [0.0838414, 0.2526837]]],
                device=device,
            ),
            atol=5e-5,
            rtol=5e-5,
        )
        torch.testing.assert_close(
            next_belief,
            torch.tensor(
                [[0.0575409, -0.1607322, 0.2253285, -0.2480658]], device=device
            ),
            atol=5e-5,
            rtol=5e-5,
        )
        assert sampled_state.shape == (1, 4)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_block_gru_action_normalization_and_gradients(self, device):
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=8,
            rnn_hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
            device=device,
        )
        state = torch.randn(3, 8, device=device, requires_grad=True)
        belief = torch.randn(3, 8, device=device, requires_grad=True)
        action = torch.tensor([[2.0, -4.0], [0.5, -0.25], [-3.0, 2.0]], device=device)
        normalized_action = action / action.abs().clamp_min(1)

        torch.manual_seed(0)
        logits, _, next_belief = prior(state, belief, action)
        torch.manual_seed(0)
        normalized_logits, _, normalized_belief = prior(
            state, belief, normalized_action
        )

        torch.testing.assert_close(logits, normalized_logits)
        torch.testing.assert_close(next_belief, normalized_belief)
        (logits.square().mean() + next_belief.square().mean()).backward()
        assert state.grad is not None
        assert belief.grad is not None
        assert all(parameter.grad is not None for parameter in prior.parameters())

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_block_gru_bfloat16_autocast_forward_and_gradients(self):
        torch.manual_seed(0)
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=8,
            rnn_hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
            device="cuda",
        )
        recurrent_core = prior.rnn
        parameters = tuple(recurrent_core.parameters())
        inputs = (
            torch.randn(3, 8, device="cuda"),
            torch.randn(3, 8, device="cuda"),
            torch.randn(3, 2, device="cuda"),
        )

        def run(module, *, explicit_compute_dtype=False):
            local_inputs = tuple(value.clone().requires_grad_() for value in inputs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                module_inputs = (
                    tuple(value.to(torch.bfloat16) for value in local_inputs)
                    if explicit_compute_dtype
                    else local_inputs
                )
                output = module(*module_inputs)
            gradients = torch.autograd.grad(
                output.float().square().sum(), (*local_inputs, *parameters)
            )
            return output, gradients

        reference, reference_gradients = run(
            recurrent_core, explicit_compute_dtype=True
        )
        eager, eager_gradients = run(recurrent_core)
        assert reference.dtype is torch.bfloat16
        assert eager.dtype is torch.bfloat16
        torch.testing.assert_close(eager, reference, rtol=0, atol=0)
        for eager_gradient, reference_gradient in zip(
            eager_gradients, reference_gradients
        ):
            assert eager_gradient.dtype is torch.float32
            torch.testing.assert_close(
                eager_gradient, reference_gradient, rtol=0, atol=0
            )

        compiled, compiled_gradients = run(
            torch.compile(recurrent_core, fullgraph=True)
        )
        assert compiled.dtype is torch.bfloat16
        torch.testing.assert_close(compiled, reference, rtol=3e-2, atol=3e-2)
        for compiled_gradient, reference_gradient in zip(
            compiled_gradients, reference_gradients
        ):
            assert compiled_gradient.dtype is torch.float32
            torch.testing.assert_close(
                compiled_gradient,
                reference_gradient,
                rtol=3e-2,
                atol=3e-2,
            )

    def test_block_gru_torch_compile(self):
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=8,
            rnn_hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
        )
        state = torch.randn(3, 8)
        belief = torch.randn(3, 8)
        action = torch.randn(3, 2)
        compiled = torch.compile(prior, fullgraph=True)

        torch.manual_seed(0)
        expected = prior(state, belief, action)
        torch.manual_seed(0)
        actual = compiled(state, belief, action)
        for expected_item, actual_item in zip(expected, actual):
            torch.testing.assert_close(expected_item, actual_item)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_posterior_rms_norm(self, device):
        posterior = RSSMPosteriorV3(
            hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            rnn_hidden_dim=8,
            obs_embed_dim=6,
            use_rms_norm=True,
            num_layers=1,
            device=device,
        )
        belief = torch.randn(3, 8, device=device)
        embedding = torch.randn(3, 6, device=device)
        logits, state = posterior(belief, embedding)
        assert logits.shape == (3, 2, 4)
        assert state.shape == (3, 8)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_rssm_posterior_v3_forward_shapes_and_grads(self, device):
        num_cats = num_classes = 4
        rnn_hidden_dim = 8
        state_dim = num_cats * num_classes
        B = 4
        obs_embed_dim = 16
        posterior = RSSMPosteriorV3(
            hidden_dim=rnn_hidden_dim,
            num_categoricals=num_cats,
            num_classes=num_classes,
            rnn_hidden_dim=rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        belief = torch.randn(B, rnn_hidden_dim, device=device, requires_grad=True)
        obs_embed = torch.randn(B, obs_embed_dim, device=device, requires_grad=True)

        logits, state = posterior(belief, obs_embed)
        assert logits.shape == (B, num_cats, num_classes)
        assert state.shape == (B, state_dim)
        # one-hot forward: each categorical sums to 1
        state_grid = state.view(B, num_cats, num_classes)
        assert torch.allclose(
            state_grid.sum(-1), torch.ones(B, num_cats, device=device), atol=1e-5
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

    @pytest.mark.parametrize("device", get_default_devices())
    def test_rssm_rollout_v3_forward(self, device):
        num_cats = num_classes = 4
        rnn_hidden_dim = 8
        state_dim = num_cats * num_classes
        B, T = 2, 4
        obs_embed_dim = 12
        action_dim = 3

        prior_net = RSSMPriorV3(
            action_shape=torch.Size([action_dim]),
            hidden_dim=rnn_hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_categoricals=num_cats,
            num_classes=num_classes,
            action_dim=action_dim,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=rnn_hidden_dim,
            num_categoricals=num_cats,
            num_classes=num_classes,
            rnn_hidden_dim=rnn_hidden_dim,
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
                "state": torch.zeros(B, T, state_dim, device=device),
                "belief": torch.zeros(B, T, rnn_hidden_dim, device=device),
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
        assert prior_logits.shape == (B, T, num_cats, num_classes)
        assert post_logits.shape == (B, T, num_cats, num_classes)

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

        # Step 1 is not a reset, so its action must reach the recurrence.
        td_c = td_a.clone()
        td_c["action"][:, 1] += 1.0
        torch.manual_seed(0)
        out_c = rollout(td_c)
        moved = (
            out_c["next", "prior_logits"][:, 1] - out_a["next", "prior_logits"][:, 1]
        )
        assert moved.abs().max() > 1e-6


class _ReorderedPrior(torch.nn.Module):
    """Feed a prior its positional inputs from an arbitrary in_keys order."""

    def __init__(self, prior_net: torch.nn.Module, argument_order: list[int]):
        super().__init__()
        self.prior_net = prior_net
        self.argument_order = argument_order

    def forward(self, *inputs: torch.Tensor):
        ordered = [None] * len(inputs)
        for value, position in zip(inputs, self.argument_order):
            ordered[position] = value
        return self.prior_net(*ordered)


class TestDreamerV3RolloutFastPath:
    """The tensor path must be a drop-in replacement for the TensorDict path.

    It is the default whenever the wiring is the standard DreamerV3 one, so any
    difference between the two would silently change what every walker run
    trains, including the random stream a seed selects.
    """

    @staticmethod
    def _build(fast_path, device):
        torch.manual_seed(0)
        prior = TensorDictModule(
            RSSMPriorV3(
                action_shape=torch.Size([6]),
                hidden_dim=32,
                rnn_hidden_dim=32,
                num_categoricals=8,
                num_classes=4,
                action_dim=6,
                recurrent_model="block_gru",
                device=device,
            ),
            in_keys=["state", "belief", "action"],
            out_keys=[("next", "prior_logits"), ("next", "state"), ("next", "belief")],
        )
        posterior = TensorDictModule(
            RSSMPosteriorV3(
                hidden_dim=32,
                num_categoricals=8,
                num_classes=4,
                rnn_hidden_dim=32,
                obs_embed_dim=10,
                device=device,
            ),
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[("next", "posterior_logits"), ("next", "state")],
        )
        rollout = RSSMRolloutV3(prior, posterior)
        if fast_path is not None:
            rollout._fast_path = fast_path
        return rollout

    @staticmethod
    def _make_input(device, batch=3, time_steps=7):
        torch.manual_seed(123)
        tensordict = TensorDict(
            {
                "state": torch.randn(batch, time_steps, 32, device=device),
                "belief": torch.randn(batch, time_steps, 32, device=device),
                "action": torch.randn(batch, time_steps, 6, device=device),
                "is_init": torch.zeros(
                    batch, time_steps, 1, dtype=torch.bool, device=device
                ),
                "next": TensorDict(
                    {
                        "encoded_latents": torch.randn(
                            batch, time_steps, 10, device=device
                        )
                    },
                    [batch, time_steps],
                    device=device,
                ),
            },
            [batch, time_steps],
            device=device,
        )
        tensordict["is_init"][:, 0] = True
        # A reset inside the sequence, not only at its start.
        tensordict["is_init"][1, 3] = True
        return tensordict

    def test_standard_wiring_uses_the_tensor_path(self):
        assert self._build(None, "cpu")._fast_path is True

    @pytest.mark.parametrize("device", get_default_devices())
    def test_paths_agree_exactly(self, device):
        loop = self._build(False, device)
        fast = self._build(True, device)
        fast.load_state_dict(loop.state_dict())
        tensordict = self._make_input(device)

        torch.manual_seed(7)
        out_loop = loop(tensordict.copy())
        torch.manual_seed(7)
        out_fast = fast(tensordict.copy())

        loop_keys = set(out_loop.keys(include_nested=True, leaves_only=True))
        assert loop_keys == set(out_fast.keys(include_nested=True, leaves_only=True))
        for key in loop_keys:
            # Exact, not close: the tensor path runs the same operations in the
            # same order and draws from the random stream the same number of
            # times, so a seeded run reproduces the TensorDict path bit for bit.
            torch.testing.assert_close(
                out_fast.get(key), out_loop.get(key), rtol=0, atol=0
            )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_gradients_agree_exactly(self, device):
        loop = self._build(False, device)
        fast = self._build(True, device)
        fast.load_state_dict(loop.state_dict())
        tensordict = self._make_input(device)

        def grads(rollout):
            rollout.zero_grad(set_to_none=True)
            torch.manual_seed(7)
            out = rollout(tensordict.copy())
            out.get(("next", "posterior_logits")).square().mean().backward()
            return {
                name: parameter.grad
                for name, parameter in rollout.named_parameters()
                if parameter.grad is not None
            }

        loop_grads, fast_grads = grads(loop), grads(fast)
        assert set(loop_grads) == set(fast_grads) and loop_grads
        for name, grad in loop_grads.items():
            torch.testing.assert_close(fast_grads[name], grad, rtol=0, atol=0)

    def test_reset_masks_the_action_on_both_paths(self):
        tensordict = self._make_input("cpu")
        for fast_path in (False, True):
            rollout = self._build(fast_path, "cpu")
            out = rollout(tensordict.copy())
            reset = tensordict.get("is_init").squeeze(-1)
            masked = out.get("action")[reset]
            assert masked.abs().max() == 0.0, f"fast_path={fast_path}"

    @staticmethod
    def _build_with_action_key(action_key, in_keys=None):
        torch.manual_seed(0)
        prior = TensorDictModule(
            RSSMPriorV3(
                action_shape=torch.Size([6]),
                hidden_dim=32,
                rnn_hidden_dim=32,
                num_categoricals=8,
                num_classes=4,
                action_dim=6,
                recurrent_model="block_gru",
            ),
            in_keys=in_keys or ["state", "belief", action_key],
            out_keys=[("next", "prior_logits"), ("next", "state"), ("next", "belief")],
        )
        posterior = TensorDictModule(
            RSSMPosteriorV3(
                hidden_dim=32,
                num_categoricals=8,
                num_classes=4,
                rnn_hidden_dim=32,
                obs_embed_dim=10,
            ),
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[("next", "posterior_logits"), ("next", "state")],
        )
        return prior, posterior

    def test_nested_action_key_uses_the_tensor_path(self):
        # The tensor path calls the modules positionally, so the action's name
        # is irrelevant to it and a nested key must not cost the fast path.
        prior, posterior = self._build_with_action_key(("agent", "action"))
        loop = RSSMRolloutV3(prior, posterior)
        loop._fast_path = False
        fast = RSSMRolloutV3(prior, posterior)
        assert fast._fast_path is True
        assert fast.action_key == ("agent", "action")

        tensordict = self._make_input("cpu")
        action = tensordict.get("action")
        tensordict = tensordict.exclude("action")
        tensordict.set(("agent", "action"), action)

        torch.manual_seed(7)
        out_loop = loop(tensordict.copy())
        torch.manual_seed(7)
        out_fast = fast(tensordict.copy())
        for key in out_loop.keys(include_nested=True, leaves_only=True):
            torch.testing.assert_close(
                out_fast.get(key), out_loop.get(key), rtol=0, atol=0
            )

    def test_reordered_wiring_falls_back_to_the_tensordict_path(self):
        # Positional calls make the carry order load-bearing, so a reordered
        # prior must take the TensorDict path rather than be called wrongly.
        prior, posterior = self._build_with_action_key(
            "action", in_keys=["belief", "state", "action"]
        )
        rollout = RSSMRolloutV3(prior, posterior)
        assert rollout._fast_path is False

    def test_compile_step_keeps_the_random_stream(self):
        """compile_step must not move the sampled trajectory.

        Only the deterministic per-step work is compiled, so the draws stay in
        eager and the sampled states must be identical. Fusion reorders float
        arithmetic, so the logits are only close.
        """
        rollout = self._build(True, "cpu")
        tensordict = self._make_input("cpu")

        torch.manual_seed(7)
        eager = rollout(tensordict.copy())
        rollout.compile_rollout("step")
        torch.manual_seed(7)
        compiled = rollout(tensordict.copy())

        # Masking is pure indexing, so the action is exact.
        torch.testing.assert_close(
            compiled.get("action"), eager.get("action"), rtol=0, atol=0
        )
        # The drawn category is what the random stream decides. A different
        # draw would move a one-hot by 1.0; fusion moves it by rounding, since
        # the straight-through value is ``probs + (one_hot - probs)``.
        for key in ("state", ("next", "state")):
            eager_state = eager.get(key).unflatten(-1, (8, 4))
            compiled_state = compiled.get(key).unflatten(-1, (8, 4))
            assert torch.equal(
                compiled_state.argmax(-1), eager_state.argmax(-1)
            ), f"{key} drew a different category"
        for key in (
            "state",
            "belief",
            ("next", "state"),
            ("next", "belief"),
            ("next", "prior_logits"),
            ("next", "posterior_logits"),
        ):
            torch.testing.assert_close(
                compiled.get(key), eager.get(key), rtol=1e-4, atol=1e-5
            )

    def test_compile_rollout_rejects_an_unknown_scope(self):
        rollout = self._build(True, "cpu")
        with pytest.raises(ValueError, match="scope must be"):
            rollout.compile_rollout("everything")
        # A rejected scope must not discard an existing compile.
        rollout.compile_rollout("step")
        with pytest.raises(ValueError, match="scope must be"):
            rollout.compile_rollout("everything")
        assert rollout._step_fn is not None

    def test_compile_scan_runs_and_backpropagates(self):
        # "scan" draws differently from eager, so only shapes, finiteness and
        # gradient flow are asserted.
        rollout = self._build(True, "cpu")
        # Inductor compiles the unrolled recurrence, so keep the horizon short.
        tensordict = self._make_input("cpu", time_steps=4)
        rollout.compile_rollout("scan")
        out = rollout(tensordict.copy())
        logits = out.get(("next", "posterior_logits"))
        assert logits.shape == (3, 4, 8, 4) and logits.isfinite().all()
        logits.square().mean().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in rollout.parameters()
        )

    def test_compiled_rollout_is_picklable(self):
        # Compiled callables cannot be pickled, so a copy falls back to eager.
        rollout = self._build(True, "cpu")
        rollout.compile_rollout("step")
        restored = pickle.loads(pickle.dumps(rollout))
        assert restored._step_fn is None and rollout._step_fn is not None
        torch.testing.assert_close(
            restored(self._make_input("cpu").copy()).get(("next", "prior_logits")),
            rollout(self._make_input("cpu").copy()).get(("next", "prior_logits")),
        )

    def test_compile_requires_the_tensor_path(self):
        rollout = self._build(False, "cpu")
        with pytest.raises(RuntimeError, match="requires the tensor path"):
            rollout.compile_rollout("step")

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "action_key,prior_in_keys,explicit",
        [
            ("action", ["state", "belief", "action"], False),
            (("agent", "action"), ["state", "belief", ("agent", "action")], False),
            (("agent", "action"), [("agent", "action"), "state", "belief"], True),
            (("agent", "action"), [("agent", "action"), "state", "belief"], False),
        ],
    )
    def test_dreamer_v3_rollout_masks_action_on_reset(
        self, device, action_key, prior_in_keys, explicit
    ):
        """A reset step must not condition on the previous action.

        The reference masks ``(deter, stoch, action)`` together in
        ``rssm._observe``, so only masking the carry would leak the action
        across an episode boundary. The action key defaults to whichever prior
        input is not the carry, and can also be named explicitly.
        """
        num_cats = num_classes = 4
        rnn_hidden_dim = 8
        state_dim = num_cats * num_classes
        B, T, obs_embed_dim = 2, 2, 12
        action_dim = 3
        prior_net = RSSMPriorV3(
            action_shape=torch.Size([action_dim]),
            hidden_dim=rnn_hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_categoricals=num_cats,
            num_classes=num_classes,
            action_dim=action_dim,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=rnn_hidden_dim,
            num_categoricals=num_cats,
            num_classes=num_classes,
            rnn_hidden_dim=rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)
        # ``prior_net`` reads its inputs positionally, so reordering the module
        # keys must be matched by reordering the values it receives.
        order = {"state": 0, "belief": 1}
        argument_order = [order.get(key, 2) for key in prior_in_keys]
        natural_order = argument_order == [0, 1, 2]
        prior_module = TensorDictModule(
            prior_net if natural_order else _ReorderedPrior(prior_net, argument_order),
            in_keys=prior_in_keys,
            out_keys=[
                ("next", "prior_logits"),
                ("next", "state"),
                ("next", "belief"),
            ],
        )
        rollout = RSSMRolloutV3(
            prior_module,
            TensorDictModule(
                posterior_net,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
            action_key=action_key if explicit else None,
        )
        assert rollout.action_key == unravel_key(action_key)
        assert rollout._fast_path is natural_order

        def run(action):
            td = TensorDict(
                {
                    "state": torch.zeros(B, T, state_dim, device=device),
                    "belief": torch.zeros(B, T, rnn_hidden_dim, device=device),
                    "is_init": torch.ones(B, T, 1, dtype=torch.bool, device=device),
                    "next": {
                        "encoded_latents": torch.zeros(
                            B, T, obs_embed_dim, device=device
                        )
                    },
                },
                [B, T],
            )
            td.set(action_key, action)
            torch.manual_seed(0)
            return rollout(td).get(("next", "belief")).clone()

        # Every step is a reset, so the belief must not depend on the action.
        differing = run(torch.randn(B, T, action_dim, device=device))
        zeroed = run(torch.zeros(B, T, action_dim, device=device))
        torch.testing.assert_close(differing, zeroed)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
