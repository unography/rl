# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 loss modules.

Implements the three loss modules from DreamerV3 (Mastering Diverse Domains in
World Models, Hafner et al. 2023, https://arxiv.org/abs/2301.04104):

- :class:`DreamerV3ModelLoss` — world model (KL balancing + symlog reconstruction)
- :class:`DreamerV3ActorLoss` — actor (REINFORCE + entropy bonus)
- :class:`DreamerV3ValueLoss` — value function (symlog MSE or two-hot CE)

Utility functions :func:`symlog`, :func:`symexp`, :func:`two_hot_encode`,
:func:`two_hot_decode`, and :func:`two_hot_cross_entropy` are also exported for
use in custom models.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict.utils import NestedKey, unravel_key

from torchrl._utils import _maybe_record_function_decorator
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules.functional import symexp as _symexp, symlog as symlog
from torchrl.modules.models.model_based_v3 import (  # noqa: F401
    _default_bins,
    _DEFAULT_NUM_BINS,
    _unimix_probs,
    two_hot_cross_entropy as two_hot_cross_entropy,
    two_hot_decode as _two_hot_decode,
    two_hot_encode as _two_hot_encode,
)
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    dispatch_value_estimator,
    hold_out_net,
    ValueEstimators,
)
from torchrl.objectives.value import ValueEstimatorBase

symexp = _symexp
two_hot_decode = _two_hot_decode
two_hot_encode = _two_hot_encode

_SHARED_VALUE_LOGITS_KEY = "_dreamer_v3_shared_value_logits"
_SHARED_SLOW_VALUE_KEY = "_dreamer_v3_shared_slow_value"

# ---------------------------------------------------------------------------
# KL balancing for categorical distributions (DreamerV3 §3)
# ---------------------------------------------------------------------------


def categorical_kl_terms(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_nats: float = 1.0,
    unimix: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DreamerV3 dynamics and representation KL losses.

    The dynamics term stops gradients through the posterior and the
    representation term stops gradients through the prior. KL divergence is
    summed over the stochastic categoricals before applying the free-nat
    threshold, matching the aggregated one-hot distribution used by the
    reference DreamerV3 implementation.

    Args:
        posterior_logits (torch.Tensor): Posterior logits with shape
            ``[..., num_categoricals, num_classes]``.
        prior_logits (torch.Tensor): Prior logits with the same shape.
        free_nats (float, optional): Minimum aggregated KL in nats. Defaults to
            ``1.0``.
        unimix (float, optional): Fraction of uniform probability mixed into
            each categorical. Defaults to ``0.01``.

    Returns:
        A pair containing the scalar dynamics and representation KL losses.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import categorical_kl_terms
        >>> posterior = torch.randn(2, 4, 8, requires_grad=True)
        >>> prior = torch.randn(2, 4, 8, requires_grad=True)
        >>> dynamics, representation = categorical_kl_terms(posterior, prior)
        >>> dynamics.shape, representation.shape
        (torch.Size([]), torch.Size([]))
    """
    posterior = _unimix_probs(posterior_logits, unimix)
    prior = _unimix_probs(prior_logits, unimix)
    posterior_log = posterior.log()
    prior_log = prior.log()

    dynamics = (posterior.detach() * (posterior_log.detach() - prior_log)).sum((-1, -2))
    representation = (posterior * (posterior_log - prior_log.detach())).sum((-1, -2))
    if free_nats:
        dynamics = dynamics.clamp_min(free_nats)
        representation = representation.clamp_min(free_nats)
    return dynamics.mean(), representation.mean()


def categorical_kl_balanced(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.8,
    free_bits: float = 1.0,
    unimix: float = 0.01,
) -> torch.Tensor:
    """KL divergence with balancing between posterior and prior.

    Computes:
        loss = alpha * KL(sg(posterior) || prior)
             + (1 - alpha) * KL(posterior || sg(prior))

    The first term trains only the *prior*; the second trains only the
    *posterior*. Free bits are applied **per categorical** (clamped before
    averaging across categoricals and batch), matching Hafner et al. 2023
    eq. 5: ``L_KL = max(free_bits, KL_per_categorical)``.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        posterior_logits: Shape ``[..., num_categoricals, num_classes]``.
        prior_logits: Shape ``[..., num_categoricals, num_classes]``.
        alpha (float): Balancing weight (0.8 in the paper). Default: 0.8.
        free_bits (float): Minimum per-categorical KL in nats. Default: 1.0.
        unimix (float): Fraction of uniform probability mixed into each
            categorical before the KL, matching the reference categorical
            distribution. Default: 0.01.

    Returns:
        Scalar KL loss.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import categorical_kl_balanced
        >>> posterior = torch.randn(4, 8, 16, requires_grad=True)
        >>> prior = torch.randn(4, 8, 16, requires_grad=True)
        >>> kl = categorical_kl_balanced(posterior, prior, alpha=0.8, free_bits=0.1)
        >>> kl.backward()
    """
    # Match the reference categorical distribution: KL probabilities and
    # gradients are evaluated in FP32 even when model logits use autocast.
    posterior = _unimix_probs(posterior_logits, unimix)
    prior = _unimix_probs(prior_logits, unimix)

    eps = 1e-8
    posterior = posterior.clamp(min=eps)
    prior = prior.clamp(min=eps)

    post_sg = posterior.detach()
    kl_term1 = (post_sg * (post_sg.log() - prior.log())).sum(-1)

    prior_sg = prior.detach()
    kl_term2 = (posterior * (posterior.log() - prior_sg.log())).sum(-1)

    # Free bits per categorical (clamp before reducing). Hafner et al. 2023, eq. 5.
    kl_term1 = kl_term1.clamp_min(free_bits).mean()
    kl_term2 = kl_term2.clamp_min(free_bits).mean()

    return alpha * kl_term1 + (1.0 - alpha) * kl_term2


def _match_trailing_dim(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Align ``source`` to the trailing feature dim of ``reference`` for broadcast.

    Multivariate action distributions often emit per-dim log-probs, while the
    discount / advantage tensors carry a singleton trailing dim. This helper
    keeps them compatible by summing over the extra feature dim (multi-D
    log-prob) or unsqueezing when the trailing dim is missing.
    """
    if source.ndim == reference.ndim:
        return source
    if source.ndim == reference.ndim - 1:
        return source.unsqueeze(-1)
    if source.ndim == reference.ndim + 1:
        return source.sum(-1, keepdim=True)
    raise ValueError(
        f"Cannot align source shape {tuple(source.shape)} to "
        f"reference shape {tuple(reference.shape)}"
    )


# ---------------------------------------------------------------------------
# DreamerV3ModelLoss
# ---------------------------------------------------------------------------


class DreamerV3ModelLoss(LossModule):
    """DreamerV3 World Model Loss.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for an overview
    of the world model, RSSM, and training flow.

    Computes three terms:

    1. **KL loss** — balanced KL between prior and posterior categorical
       distributions (see :func:`categorical_kl_balanced`).
    2. **Reconstruction loss** — MSE between the decoder's symlog-space
       prediction and the symlog-transformed observation target.
    3. **Reward loss** — two-hot cross-entropy or symlog MSE for the predicted
       reward.

    Optionally a **continue loss** (binary cross-entropy) can be enabled
    when the world model outputs a continue predictor.

    .. note::
        The decoder is expected to emit **symlog-space** predictions, as the
        reference implementation's ``symlog_mse`` head does: it applies
        ``symlog`` to the target only, and compares it against the raw linear
        output. Before v0.15 this loss applied ``symlog`` to the decoder output
        as well, which compressed the prediction twice and did not implement
        DreamerV3. A decoder that emits observation-space values (for example
        one ending in a sigmoid) must now apply ``symlog`` itself.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        world_model (TensorDictModule): World model that takes a tensordict with
            observations/actions and writes predicted observations, rewards, and
            RSSM prior/posterior logits.
        lambda_kl (float, optional): KL loss weight. Default: 1.0.
        lambda_reco (float, optional): Reconstruction loss weight. Default: 1.0.
        lambda_reward (float, optional): Reward prediction loss weight. Default: 1.0.
        lambda_continue (float, optional): Continue prediction loss weight.
            Default: 0.0 (disabled).
        continue_target_scale (float, optional): Multiplier applied to
            non-terminal continuation targets, for encoding the finite-horizon
            discount in the continuation model. Defaults to 1.0.
        kl_mode ("balanced" or "separate", optional): KL formulation.
            ``"balanced"`` preserves the historical weighted aggregate;
            ``"separate"`` emits the reference dynamics and representation
            losses. Defaults to ``"balanced"``.
        lambda_dynamic (float, optional): Dynamics KL weight in separate mode.
            Defaults to 1.0.
        lambda_representation (float, optional): Representation KL weight in
            separate mode. Defaults to 0.1.
        unimix (float, optional): Uniform mixture used by the categorical KL
            distributions. Defaults to 0.0 for compatibility.
        kl_alpha (float, optional): KL balancing factor (alpha in the paper).
            Default: 0.8.
        free_bits (float, optional): Minimum KL per categorical in nats.
            Default: 1.0.
        reco_loss (str, optional): Reconstruction loss type (``"l2"`` or
            ``"l1"``). Default: ``"l2"``.
        reward_two_hot (bool, optional): If ``True``, the reward head is
            expected to output **logits over** ``num_reward_bins`` and the loss
            is two-hot cross-entropy. If ``False``, the reward head outputs a
            **scalar** prediction and the loss is symlog MSE. Default: ``True``.
        num_reward_bins (int, optional): Number of bins for the two-hot reward
            distribution. Default: 255.
        global_average (bool, optional): If ``True``, averages losses over all
            dimensions. Otherwise sums over non-batch/time dims first. Default:
            ``False``.
        detach_output (bool, optional): If ``True``, detach the returned world
            model output. Set to ``False`` when a replay value loss must update
            the representation, matching DreamerV3 ``repval_grad=True``.
            Default: ``True``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.modules import SymExpTwoHot
        >>> from torchrl.objectives import DreamerV3ModelLoss
        >>> class StubWorldModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.head = nn.LazyLinear(4 * 4)
        ...         self.reward_head = nn.LazyLinear(16)
        ...         self.reward_decoder = SymExpTwoHot(16)
        ...         self.decoder = nn.LazyLinear(3 * 8 * 8)
        ...     def forward(self, td):
        ...         B, T = td.shape
        ...         x = torch.cat([td["state"], td["belief"]], dim=-1)
        ...         logits = self.head(x).view(B, T, 4, 4)
        ...         reco = self.decoder(x).view(B, T, 3, 8, 8)
        ...         reward_logits = self.reward_head(x)
        ...         td.set(("next", "prior_logits"), logits)
        ...         td.set(("next", "posterior_logits"), logits)
        ...         td.set(("next", "reco_pixels"), reco)
        ...         td.set(("next", "reward_logits"), reward_logits)
        ...         td.set(("next", "reward"), self.reward_decoder(reward_logits))
        ...         return td
        >>> wm = StubWorldModel()
        >>> td = TensorDict({
        ...     "state": torch.zeros(2, 3, 16),
        ...     "belief": torch.zeros(2, 3, 8),
        ...     "action": torch.randn(2, 3, 2),
        ...     "next": {
        ...         "pixels": torch.rand(2, 3, 3, 8, 8),
        ...         "reward": torch.randn(2, 3, 1),
        ...         "done": torch.zeros(2, 3, dtype=torch.bool),
        ...     },
        ... }, [2, 3])
        >>> with torch.no_grad():
        ...     wm(td.clone())
        TensorDict(...)
        >>> loss = DreamerV3ModelLoss(wm, num_reward_bins=16, free_bits=0.1)
        >>> loss_td, _ = loss(td)
        >>> sorted(loss_td.keys())
        ['loss_model_kl', 'loss_model_reco', 'loss_model_reward']
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            reward (NestedKey): Decoded predicted reward. Defaults to
                ``"reward"``.
            reward_logits (NestedKey): Categorical reward logits. Defaults to
                ``"reward_logits"``.
            true_reward (NestedKey): Ground-truth reward (stored temporarily).
                Defaults to ``"true_reward"``.
            prior_logits (NestedKey): Prior categorical logits from the prior
                RSSM. Defaults to ``"prior_logits"``.
            posterior_logits (NestedKey): Posterior categorical logits.
                Defaults to ``"posterior_logits"``.
            pixels (NestedKey): Ground-truth pixel observation.
                Defaults to ``"pixels"``.
            reco_pixels (NestedKey): Predicted pixel observation, in symlog
                space. Defaults to ``"reco_pixels"``.
            continue_pred (NestedKey): Predicted continue logit (optional).
                Defaults to ``"continue_pred"``.
            done (NestedKey): Ground-truth done flag (optional).
                Defaults to ``"done"``.
            terminated (NestedKey): Ground-truth terminal flag (optional).
                Defaults to ``"terminated"``.
        """

        reward: NestedKey = "reward"
        reward_logits: NestedKey = "reward_logits"
        true_reward: NestedKey = "true_reward"
        prior_logits: NestedKey = "prior_logits"
        posterior_logits: NestedKey = "posterior_logits"
        pixels: NestedKey = "pixels"
        reco_pixels: NestedKey = "reco_pixels"
        continue_pred: NestedKey = "continue_pred"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    def __init__(
        self,
        world_model: TensorDictModule,
        *,
        lambda_kl: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        lambda_continue: float = 0.0,
        kl_mode: Literal["balanced", "separate"] = "balanced",
        lambda_dynamic: float = 1.0,
        lambda_representation: float = 0.1,
        unimix: float = 0.0,
        continue_target_scale: float = 1.0,
        kl_alpha: float = 0.8,
        free_bits: float = 1.0,
        reco_loss: str = "l2",
        reward_two_hot: bool = True,
        num_reward_bins: int = _DEFAULT_NUM_BINS,
        global_average: bool = False,
        detach_output: bool = True,
    ):
        super().__init__()
        self.world_model = world_model
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.lambda_continue = lambda_continue
        if kl_mode not in ("balanced", "separate"):
            raise ValueError(
                "kl_mode must be 'balanced' or 'separate', got " f"{kl_mode!r}."
            )
        self.kl_mode = kl_mode
        self.lambda_dynamic = lambda_dynamic
        self.lambda_representation = lambda_representation
        self.unimix = unimix
        if not 0 < continue_target_scale <= 1:
            raise ValueError("continue_target_scale must be in (0, 1].")
        self.continue_target_scale = continue_target_scale
        self.kl_alpha = kl_alpha
        self.free_bits = free_bits
        self.reco_loss = reco_loss
        self.reward_two_hot = reward_two_hot
        self.num_reward_bins = num_reward_bins
        self.global_average = global_average
        self.detach_output = detach_output
        self.register_buffer(
            "reward_bins",
            _default_bins(num_reward_bins),
        )

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @_maybe_record_function_decorator("dreamer_v3/world_model_loss")
    def forward(self, tensordict: TensorDict) -> tuple[TensorDict, TensorDict]:
        tensordict = tensordict.copy()
        tensordict.rename_key_(
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.true_reward),
        )

        tensordict = self.world_model(tensordict)

        # ---- KL loss ----
        prior_logits = tensordict.get(("next", self.tensor_keys.prior_logits))
        posterior_logits = tensordict.get(("next", self.tensor_keys.posterior_logits))
        if self.kl_mode == "separate":
            dynamic_loss, representation_loss = categorical_kl_terms(
                posterior_logits,
                prior_logits,
                free_nats=self.free_bits,
                unimix=self.unimix,
            )
            dynamic_loss = dynamic_loss.unsqueeze(-1)
            representation_loss = representation_loss.unsqueeze(-1)
        else:
            kl_loss = categorical_kl_balanced(
                posterior_logits,
                prior_logits,
                alpha=self.kl_alpha,
                free_bits=self.free_bits,
                unimix=self.unimix,
            ).unsqueeze(-1)

        # ---- Reconstruction loss ----
        pixels = tensordict.get(("next", self.tensor_keys.pixels)).contiguous()
        reco_pixels = tensordict.get(
            ("next", self.tensor_keys.reco_pixels)
        ).contiguous()
        # The decoder predicts in symlog space directly. Applying symlog to its
        # output again would compress the prediction twice.
        if self.reco_loss == "l2":
            reco_loss = (symlog(pixels) - reco_pixels).pow(2)
        else:
            reco_loss = (symlog(pixels) - reco_pixels).abs()
        if not self.global_average:
            # The first dimensions are TensorDict batch dimensions (batch and
            # time in DreamerV3); every trailing observation/event dimension is
            # summed, matching embodied.jax.outs.Agg.
            event_dims = tuple(range(tensordict.batch_dims, reco_loss.ndim))
            if event_dims:
                reco_loss = reco_loss.sum(event_dims)
        reco_loss = reco_loss.mean().unsqueeze(-1)

        # ---- Reward loss ----
        true_reward = tensordict.get(("next", self.tensor_keys.true_reward))
        if self.reward_two_hot:
            reward_logits_key = unravel_key(("next", self.tensor_keys.reward_logits))
            pred_reward = tensordict.get(reward_logits_key, None)
            if pred_reward is None:
                legacy_key = unravel_key(("next", self.tensor_keys.reward))
                pred_reward = tensordict.get(legacy_key)
                warnings.warn(
                    "Storing DreamerV3 categorical reward logits under the decoded "
                    f"reward key {legacy_key!r} is deprecated and will be removed in "
                    "v0.16. Write logits to the configured reward_logits key instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if pred_reward.shape[-1] != self.num_reward_bins:
                raise ValueError(
                    f"reward_two_hot=True expects the reward head to output "
                    f"logits over {self.num_reward_bins} bins, got trailing "
                    f"dim {pred_reward.shape[-1]}."
                )
            reward_loss = two_hot_cross_entropy(
                pred_reward, true_reward, self.reward_bins
            )
        else:
            pred_reward = tensordict.get(("next", self.tensor_keys.reward))
            reward_loss = (symlog(true_reward) - symlog(pred_reward)).pow(2).squeeze(-1)
        reward_loss = reward_loss.mean().unsqueeze(-1)

        td_out = TensorDict(
            loss_model_reco=self.lambda_reco * reco_loss,
            loss_model_reward=self.lambda_reward * reward_loss,
        )
        if self.kl_mode == "separate":
            td_out.set(
                "loss_model_dynamic",
                self.lambda_kl * self.lambda_dynamic * dynamic_loss,
            )
            td_out.set(
                "loss_model_representation",
                self.lambda_kl * self.lambda_representation * representation_loss,
            )
        else:
            td_out.set("loss_model_kl", self.lambda_kl * kl_loss)

        # ---- Optional continue loss ----
        if self.lambda_continue > 0:
            continue_pred = tensordict.get(
                ("next", self.tensor_keys.continue_pred), None
            )
            terminated = tensordict.get(("next", self.tensor_keys.terminated), None)
            if terminated is None:
                terminated = tensordict.get(("next", self.tensor_keys.done), None)
            if continue_pred is not None and terminated is not None:
                continue_target = (~terminated).float() * self.continue_target_scale
                continue_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    continue_pred.squeeze(-1), continue_target.squeeze(-1)
                ).unsqueeze(-1)
                td_out.set("loss_model_continue", self.lambda_continue * continue_loss)

        self._clear_weakrefs(tensordict, td_out)
        return td_out, tensordict.data if self.detach_output else tensordict


# ---------------------------------------------------------------------------
# DreamerV3ActorLoss
# ---------------------------------------------------------------------------


class _DreamerV3ImaginationRollout(TensorDictModuleBase):
    """Tensor-only DreamerV3 latent imagination rollout.

    This private helper is specialized for the standard DreamerV3 tensor
    modules. It avoids the environment and TensorDict work inside the recurrent
    loop while retaining a TensorDict input/output boundary.
    """

    def __init__(
        self,
        prior_model: torch.nn.Module,
        actor_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        reward_decoder: torch.nn.Module,
        horizon: int,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        self.prior_model = prior_model
        self.actor_model = actor_model
        self.reward_model = reward_model
        self.reward_decoder = reward_decoder
        self.horizon = horizon
        self.in_keys = ["state", "belief"]
        self.out_keys = [
            "state",
            "belief",
            "action",
            ("next", "state"),
            ("next", "belief"),
            ("next", "reward"),
        ]
        self._scan_fn = None

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Roll out ``horizon`` imagined transitions from replay features."""
        scan = self._scan_fn if self._scan_fn is not None else self._scan
        state, belief, action, next_state, next_belief, reward = scan(
            tensordict.get("state"), tensordict.get("belief")
        )
        batch_size = tensordict.batch_size + torch.Size([self.horizon])
        return TensorDict(
            {
                "state": state,
                "belief": belief,
                "action": action,
                "next": {
                    "state": next_state,
                    "belief": next_belief,
                    "reward": reward,
                },
            },
            batch_size,
            device=tensordict.device,
        )

    def _scan(
        self, state: torch.Tensor, belief: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the fixed-length recurrence using tensors only."""
        states = []
        beliefs = []
        actions = []
        next_states = []
        next_beliefs = []

        for _ in range(self.horizon):
            states.append(state)
            beliefs.append(belief)
            location, scale = self.actor_model(state, belief)
            action = location + scale * torch.randn_like(location)
            actions.append(action)
            _, state, belief = self.prior_model(state, belief, action)
            next_states.append(state)
            next_beliefs.append(belief)

        states = torch.stack(states, -2)
        beliefs = torch.stack(beliefs, -2)
        actions = torch.stack(actions, -2)
        next_states = torch.stack(next_states, -2)
        next_beliefs = torch.stack(next_beliefs, -2)
        # JAX prediction heads consume [deter, stoch]. The decoder is the only
        # head that intentionally keeps [stoch, deter].
        reward_logits = self.reward_model(next_beliefs, next_states)
        # The JAX categorical output probabilities and decoded scalar are FP32
        # even when the head itself runs in reduced precision.
        reward = self.reward_decoder(reward_logits.float())
        return states, beliefs, actions, next_states, next_beliefs, reward

    def compile_scan(self, **compile_kwargs) -> None:
        """Compile the fixed-shape tensor recurrence with :func:`torch.compile`."""
        compile_kwargs.setdefault("dynamic", False)
        self._scan_fn = torch.compile(self._scan, **compile_kwargs)


class DreamerV3ActorLoss(LossModule):
    """DreamerV3 Actor Loss.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for an overview
    of latent imagination, actor training, and DreamerV3 nomenclature.

    Rolls out imagined trajectories in latent space using the world model
    environment, then computes:

    .. code-block:: text

        loss_actor = -E[log pi(a_t | z_t) * sg(A_t)] - eta * H[pi(. | z_t)]

    where ``A_t = V_lambda(z_t) - v(z_t)`` is the advantage (lambda return
    minus baseline) and ``eta`` is the entropy bonus weight.

    When the actor is a reparameterizable (continuous) policy the
    reparameterization gradient is used directly instead of REINFORCE.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        actor_model (TensorDictModule): The actor / policy network.
        value_model (TensorDictModule): The value network.
        model_based_env (DreamerEnv): The imagination environment.
        continuation_model (TensorDictModuleBase, optional): Shared trained
            model that writes continuation probabilities for imagined states.
            Defaults to ``None``.
        imagination_rollout (TensorDictModuleBase, optional): Optimized rollout
            module that maps initial state and belief tensors to an imagined
            transition TensorDict. This is supported for DreamerV3 REINFORCE
            semantics, where the rollout itself is stop-gradient and the actor
            is trained by re-scoring detached actions. Defaults to ``None`` and
            uses ``model_based_env.rollout``.
        imagination_horizon (int, optional): Rollout length inside imagination.
            Default: 15.
        discount_loss (bool, optional): If ``True``, discount the actor loss
            with a cumulative gamma factor. Default: ``True``.
        entropy_bonus (float, optional): Weight for the entropy regularisation
            term ``eta``. Default: ``3e-4``.
        use_reinforce (bool, optional): If ``True``, uses REINFORCE (log-prob
            * stop-gradient advantage). If ``False``, uses the straight
            reparameterization gradient (suitable for continuous Gaussian
            actors). Default: ``False``.
        policy_loss_mode ("legacy" or "dreamer_v3", optional): Policy-gradient
            semantics. ``"dreamer_v3"`` re-scores detached actions from
            detached imagined features, uses analytic policy entropy, and
            averages across batch, time, and event dimensions, matching the
            JAX DreamerV3 reference. ``"legacy"`` preserves the historical
            sampled log-probability entropy surrogate and summed time/event
            reduction. Default: ``"legacy"``.
        return_normalization (bool, optional): Normalize detached REINFORCE
            advantages by an EMA return-percentile span. Default: ``True``.
        return_normalization_rate (float, optional): EMA update rate for the
            return statistics. Default: ``0.01``.
        return_normalization_quantiles (tuple of float, optional): Lower and
            upper return quantiles. Default: ``(0.05, 0.95)``.
        return_normalization_min_scale (float, optional): Minimum divisor for
            REINFORCE advantages. Default: ``1.0``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     InteractionType,
        ...     ProbabilisticTensorDictModule,
        ...     ProbabilisticTensorDictSequential,
        ...     TensorDictModule,
        ... )
        >>> from torchrl.data import Unbounded
        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.envs.model_based.dreamer import DreamerEnv
        >>> from torchrl.envs.transforms import TensorDictPrimer
        >>> from torchrl.modules import MLP, SafeSequential, WorldModelWrapper
        >>> from torchrl.modules.distributions.continuous import TanhNormal
        >>> from torchrl.modules.models.model_based import DreamerActor
        >>> from torchrl.modules.models.model_based_v3 import RSSMPriorV3
        >>> from torchrl.objectives import DreamerV3ActorLoss
        >>> from torchrl.objectives.utils import ValueEstimators
        >>> from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv
        >>> base_env = TransformedEnv(
        ...     ContinuousActionConvMockEnv(pixel_shape=[3, 16, 16]),
        ...     TensorDictPrimer(
        ...         random=False, default_value=0,
        ...         state=Unbounded(16), belief=Unbounded(8),
        ...     ),
        ... )
        >>> action_dim = base_env.action_spec.shape[0]
        >>> rssm_prior = RSSMPriorV3(
        ...     action_shape=base_env.action_spec.shape,
        ...     hidden_dim=8, rnn_hidden_dim=8,
        ...     num_categoricals=4, num_classes=4, action_dim=action_dim,
        ... )
        >>> transition = SafeSequential(
        ...     TensorDictModule(
        ...         rssm_prior,
        ...         in_keys=["state", "belief", "action"],
        ...         out_keys=["_", "state", "belief"],
        ...     ),
        ... )
        >>> reward = TensorDictModule(
        ...     MLP(out_features=1, depth=1, num_cells=8),
        ...     in_keys=["state", "belief"], out_keys=["reward"],
        ... )
        >>> mb_env = DreamerEnv(
        ...     world_model=WorldModelWrapper(transition, reward),
        ...     prior_shape=torch.Size([16]),
        ...     belief_shape=torch.Size([8]),
        ... )
        >>> mb_env.set_specs_from_env(base_env)
        >>> with torch.no_grad():
        ...     _ = mb_env.rollout(3)
        >>> actor_module = DreamerActor(out_features=action_dim, depth=1, num_cells=8)
        >>> actor = ProbabilisticTensorDictSequential(
        ...     TensorDictModule(
        ...         actor_module, in_keys=["state", "belief"], out_keys=["loc", "scale"],
        ...     ),
        ...     ProbabilisticTensorDictModule(
        ...         in_keys=["loc", "scale"], out_keys=["action"],
        ...         default_interaction_type=InteractionType.RANDOM,
        ...         distribution_class=TanhNormal,
        ...     ),
        ... )
        >>> warmup = TensorDict(
        ...     {"state": torch.randn(1, 2, 16), "belief": torch.randn(1, 2, 8)}, [1]
        ... )
        >>> _ = actor(warmup)
        >>> value = TensorDictModule(
        ...     MLP(out_features=1, depth=1, num_cells=8),
        ...     in_keys=["state", "belief"], out_keys=["state_value"],
        ... )
        >>> _ = value(warmup)
        >>> loss = DreamerV3ActorLoss(actor, value, mb_env, imagination_horizon=3)
        >>> loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> td = TensorDict(
        ...     {"state": torch.randn(2, 16), "belief": torch.randn(2, 8)}, [2]
        ... )
        >>> loss_td, _ = loss(td)
        >>> "loss_actor" in loss_td.keys()
        True
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            state (NestedKey): Stochastic latent state. Defaults to ``"state"``.
            belief (NestedKey): Deterministic GRU hidden state. Defaults to ``"belief"``.
            reward (NestedKey): Imagined reward. Defaults to ``"reward"``.
            value (NestedKey): State value. Defaults to ``"state_value"``.
            action_log_prob (NestedKey): Log-prob of the taken action.
                Defaults to ``"action_log_prob"``.
            done (NestedKey): Done flag. Defaults to ``"done"``.
            terminated (NestedKey): Terminated flag. Defaults to ``"terminated"``.
            continuation (NestedKey): Predicted continuation probability.
                Defaults to ``"continuation"``.
            discount_weight (NestedKey): Cumulative imagination weight.
                Defaults to ``"discount_weight"``.
        """

        state: NestedKey = "state"
        belief: NestedKey = "belief"
        reward: NestedKey = "reward"
        value: NestedKey = "state_value"
        action_log_prob: NestedKey = "action_log_prob"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        continuation: NestedKey = "continuation"
        discount_weight: NestedKey = "discount_weight"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TDLambda

    value_model: TensorDictModule
    actor_model: TensorDictModule

    def __init__(
        self,
        actor_model: TensorDictModule,
        value_model: TensorDictModule,
        model_based_env: DreamerEnv,
        *,
        continuation_model: TensorDictModuleBase | None = None,
        imagination_rollout: TensorDictModuleBase | None = None,
        imagination_horizon: int = 15,
        discount_loss: bool = True,
        entropy_bonus: float = 3e-4,
        use_reinforce: bool = False,
        policy_loss_mode: Literal["legacy", "dreamer_v3"] = "legacy",
        return_normalization: bool = True,
        return_normalization_rate: float = 0.01,
        return_normalization_quantiles: tuple[float, float] = (0.05, 0.95),
        return_normalization_min_scale: float = 1.0,
        gamma: float | None = None,
        lmbda: float | None = None,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.__dict__["value_model"] = value_model
        self.model_based_env = model_based_env
        self.__dict__["continuation_model"] = continuation_model
        self.__dict__["imagination_rollout"] = imagination_rollout
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.entropy_bonus = entropy_bonus
        self.use_reinforce = use_reinforce
        if policy_loss_mode not in ("legacy", "dreamer_v3"):
            raise ValueError(
                "policy_loss_mode must be 'legacy' or 'dreamer_v3', got "
                f"{policy_loss_mode!r}."
            )
        self.policy_loss_mode = policy_loss_mode
        if imagination_rollout is not None and (
            not use_reinforce or policy_loss_mode != "dreamer_v3"
        ):
            raise ValueError(
                "imagination_rollout requires use_reinforce=True and "
                "policy_loss_mode='dreamer_v3'."
            )
        self.return_normalization = return_normalization
        if not 0 <= return_normalization_rate <= 1:
            raise ValueError("return_normalization_rate must be in [0, 1].")
        lower_quantile, upper_quantile = return_normalization_quantiles
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError(
                "return_normalization_quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )
        if return_normalization_min_scale <= 0:
            raise ValueError("return_normalization_min_scale must be positive.")
        self.return_normalization_rate = return_normalization_rate
        self.return_normalization_quantiles = return_normalization_quantiles
        self.return_normalization_min_scale = return_normalization_min_scale
        self.register_buffer(
            "_return_normalization_quantiles",
            torch.tensor(return_normalization_quantiles),
            persistent=False,
        )
        self.register_buffer("return_low", torch.tensor(0.0))
        self.register_buffer("return_high", torch.tensor(0.0))
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        if lmbda is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(value=self._tensor_keys.value)

    def set_shared_value_forward(self, value_loss: DreamerV3ValueLoss | None) -> None:
        """Share one critic evaluation with a :class:`DreamerV3ValueLoss`.

        By default the actor loss evaluates the critic over the imagined
        features and the value loss evaluates it again. Wiring the two together
        evaluates the online and slow critics once over the H+1 features and
        reuses the result, which is what the DreamerV3 example does.

        This changes the actor loss's output: ``fake_data`` is returned with
        gradients attached and carries two extra private keys holding the
        shared value logits and slow values. Pass ``None`` to unshare.

        Args:
            value_loss (DreamerV3ValueLoss or None): The value loss to share
                with. It must use ``value_loss="two_hot"``.

        Raises:
            RuntimeError: If ``value_loss`` does not use ``"two_hot"``.

        Examples:
            >>> actor_loss.set_shared_value_forward(value_loss)  # doctest: +SKIP
        """
        if value_loss is None:
            self.__dict__.pop("_shared_value_forward", None)
            return
        if value_loss.value_loss != "two_hot":
            raise RuntimeError(
                "Shared imagination values require the value loss to use "
                f"value_loss='two_hot', got {value_loss.value_loss!r}."
            )
        self.__dict__["_shared_value_forward"] = value_loss._shared_value_forward

    @_maybe_record_function_decorator("dreamer_v3/actor_loss")
    def forward(self, tensordict: TensorDict) -> tuple[TensorDict, TensorDict]:
        tensordict = tensordict.select(
            self.tensor_keys.state, self.tensor_keys.belief
        ).data

        with hold_out_net(self.model_based_env), set_exploration_type(
            ExplorationType.RANDOM
        ):
            imagination_rollout = self.__dict__.get("imagination_rollout")
            if imagination_rollout is None:
                tensordict = self.model_based_env.reset(tensordict.copy())
                fake_data = self.model_based_env.rollout(
                    max_steps=self.imagination_horizon,
                    policy=self.actor_model,
                    auto_reset=False,
                    tensordict=tensordict,
                )
            else:
                # JAX uses stop-gradient imagined features for the REINFORCE
                # heads. Actor gradients are produced below by re-scoring the
                # detached sampled actions, so retaining this rollout graph is
                # both unnecessary and expensive.
                with torch.no_grad():
                    fake_data = imagination_rollout(tensordict)
            next_tensordict = step_mdp(fake_data, keep_other=True)

        shared_value_forward = self.__dict__.get("_shared_value_forward")
        if shared_value_forward is None:
            with hold_out_net(self.value_model):
                next_tensordict = self.value_model(next_tensordict)
            next_value = next_tensordict.get(self.tensor_keys.value)
            root_value = None
        else:
            value_logits, all_values, slow_values = shared_value_forward(fake_data)
            next_value = all_values[..., 1:, :]
            root_value = all_values[..., :-1, :]
            fake_data.set(_SHARED_VALUE_LOGITS_KEY, value_logits[..., :-1, :])
            fake_data.set(_SHARED_SLOW_VALUE_KEY, slow_values[..., :-1, :])

        reward = fake_data.get(("next", self.tensor_keys.reward))
        continuation = None
        root_continuation = None
        continuation_model = self.__dict__.get("continuation_model")
        if continuation_model is not None:
            root_continuation_td = fake_data.select(
                *continuation_model.in_keys, strict=False
            )
            continuation_td = next_tensordict.select(
                *continuation_model.in_keys, strict=False
            )
            with hold_out_net(continuation_model):
                continuation_model(root_continuation_td)
                continuation_model(continuation_td)
            root_continuation = root_continuation_td.get(self.tensor_keys.continuation)
            continuation = continuation_td.get(self.tensor_keys.continuation)
            fake_data.set(self.tensor_keys.continuation, root_continuation)
            fake_data.set(("next", self.tensor_keys.continuation), continuation)
        lambda_target = self.lambda_target(reward, next_value, continuation)
        fake_data.set(
            "lambda_target",
            lambda_target.detach()
            if shared_value_forward is not None
            else lambda_target,
        )

        # Every branch assigns ``discount`` outright, so adding a case cannot
        # leave it unbound.
        if not self.discount_loss:
            discount = torch.ones_like(lambda_target)
        elif continuation is not None and self.policy_loss_mode == "dreamer_v3":
            gamma = self.value_estimator.gamma.to(tensordict.device)
            # The reference weights the action at feature t by
            # prod_{i=0}^t continuation_i.  Lambda returns, in contrast,
            # use the next-feature continuations above (indices 1..H).
            # Keep the first factor undiscounted to match
            # cumprod(gamma * continuation) / gamma.
            discount = torch.cat(
                [
                    root_continuation[..., :1, :],
                    gamma * root_continuation[..., 1:, :],
                ],
                dim=-2,
            ).cumprod(dim=-2)
        else:
            gamma = self.value_estimator.gamma.to(tensordict.device)
            step_discount = (
                gamma.expand(lambda_target.shape)
                if continuation is None
                else gamma * continuation
            )
            discount = torch.cat(
                [
                    torch.ones_like(step_discount[..., :1, :]),
                    step_discount[..., :-1, :],
                ],
                dim=-2,
            ).cumprod(dim=-2)
        fake_data.set(self.tensor_keys.discount_weight, discount.detach())
        objective_discount = (
            discount.detach() if self.policy_loss_mode == "dreamer_v3" else discount
        )

        policy_dist = None
        if self.use_reinforce:
            # REINFORCE: log pi(a|z) * sg(A_t)
            if self.policy_loss_mode == "dreamer_v3":
                # The JAX reference stops gradients through both the imagined
                # features and sampled actions before evaluating log pi(a|z).
                policy_input = fake_data.select(
                    *self.actor_model.in_keys, strict=False
                ).detach()
                policy_dist = self.actor_model.get_dist(policy_input)
                log_prob = policy_dist.log_prob(fake_data.get("action").detach())
            else:
                log_prob = fake_data.get(self.tensor_keys.action_log_prob)
            log_prob = _match_trailing_dim(log_prob, lambda_target)
            if root_value is None:
                with hold_out_net(self.value_model):
                    baseline_td = fake_data.select(
                        *self.value_model.in_keys, strict=False
                    )
                    self.value_model(baseline_td)
                baseline = baseline_td.get(self.tensor_keys.value)
            else:
                baseline = root_value
            advantage = (lambda_target - baseline).detach()
            return_scale = self._return_scale(lambda_target)
            advantage = advantage / return_scale
            weighted_objective = objective_discount * log_prob * advantage
            if self.policy_loss_mode == "dreamer_v3":
                actor_loss = -weighted_objective.mean()
            else:
                actor_loss = -weighted_objective.sum((-2, -1)).mean()
        else:
            # Reparameterization gradient
            return_scale = torch.ones(
                (), dtype=lambda_target.dtype, device=lambda_target.device
            )
            actor_loss = -(objective_discount * lambda_target).sum((-2, -1)).mean()

        # Entropy bonus. DreamerV3 uses the distribution's analytic entropy;
        # retain the historical sampled -log_prob estimator in legacy mode.
        if self.entropy_bonus > 0:
            if self.policy_loss_mode == "dreamer_v3":
                if policy_dist is None:
                    policy_input = fake_data.select(
                        *self.actor_model.in_keys, strict=False
                    ).detach()
                    policy_dist = self.actor_model.get_dist(policy_input)
                entropy = _match_trailing_dim(policy_dist.entropy(), objective_discount)
                entropy = (objective_discount * entropy).mean()
                actor_loss = actor_loss - self.entropy_bonus * entropy
            else:
                log_prob_for_entropy = fake_data.get(
                    self.tensor_keys.action_log_prob, None
                )
                if log_prob_for_entropy is not None:
                    log_prob_for_entropy = _match_trailing_dim(
                        log_prob_for_entropy, discount
                    )
                    entropy = -(discount * log_prob_for_entropy).sum((-2, -1)).mean()
                    actor_loss = actor_loss - self.entropy_bonus * entropy

        loss_tensordict = TensorDict(
            {
                "loss_actor": actor_loss,
                "return_low": self.return_low.detach().clone(),
                "return_high": self.return_high.detach().clone(),
                "return_scale": return_scale.detach().clone(),
                "continuation_mean": (
                    (
                        torch.cat(
                            [root_continuation[..., :1, :], continuation], dim=-2
                        ).mean()
                        if self.policy_loss_mode == "dreamer_v3"
                        else continuation.mean()
                    ).detach()
                    if continuation is not None
                    else torch.ones((), device=actor_loss.device)
                ),
            },
            [],
        )
        self._clear_weakrefs(tensordict, loss_tensordict)
        return (
            loss_tensordict,
            fake_data if shared_value_forward is not None else fake_data.data,
        )

    def _return_scale(self, returns: torch.Tensor) -> torch.Tensor:
        if not self.return_normalization:
            return torch.ones((), dtype=returns.dtype, device=returns.device)
        if self.training:
            current_low, current_high = torch.quantile(
                returns.detach().to(self.return_low),
                self._return_normalization_quantiles,
            )
            with torch.no_grad():
                self.return_low.lerp_(current_low, self.return_normalization_rate)
                self.return_high.lerp_(current_high, self.return_normalization_rate)
        return (self.return_high - self.return_low).clamp_min(
            self.return_normalization_min_scale
        )

    def lambda_target(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        continuation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if continuation is not None:
            gamma = self.value_estimator.gamma.to(reward)
            lmbda = self.value_estimator.lmbda.to(reward)
            next_return = value[..., -1, :]
            returns = []
            for reward_t, value_t, continuation_t in zip(
                reversed(reward.unbind(-2)),
                reversed(value.unbind(-2)),
                reversed(continuation.unbind(-2)),
            ):
                next_return = reward_t + gamma * continuation_t * (
                    (1 - lmbda) * value_t + lmbda * next_return
                )
                returns.append(next_return)
            return torch.stack(returns[::-1], dim=-2)
        done = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        terminated = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        input_tensordict = TensorDict(
            {
                ("next", self.tensor_keys.reward): reward,
                ("next", self.tensor_keys.value): value,
                ("next", self.tensor_keys.done): done,
                ("next", self.tensor_keys.terminated): terminated,
            },
            [],
        )
        return self.value_estimator.value_estimate(input_tensordict)

    SUPPORTED_VALUE_ESTIMATORS = (
        ValueEstimators.TD0,
        ValueEstimators.TD1,
        ValueEstimators.TDLambda,
    )

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        if isinstance(value_type, ValueEstimatorBase) or (
            isinstance(value_type, type) and issubclass(value_type, ValueEstimatorBase)
        ):
            return LossModule.make_value_estimator(self, value_type, **hyperparams)
        if hasattr(self, "lmbda"):
            hyperparams.setdefault("lmbda", self.lmbda)
        if value_type == ValueEstimators.TDLambda:
            hyperparams.setdefault("vectorized", True)
        dispatch_value_estimator(
            self,
            value_type,
            supported=self.SUPPORTED_VALUE_ESTIMATORS,
            tensor_keys={
                "value": self.tensor_keys.value,
                "value_target": "value_target",
            },
            value_network=None,
            **hyperparams,
        )


# ---------------------------------------------------------------------------
# DreamerV3ValueLoss
# ---------------------------------------------------------------------------


def _dreamer_v3_replay_value_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor,
    bootstrap: torch.Tensor,
    horizon: float,
    lmbda: float,
) -> torch.Tensor:
    """Compute fixed-length replay lambda returns using tensors only."""
    reward = reward.squeeze(-1)
    done = done.squeeze(-1)
    terminated = terminated.squeeze(-1)
    discount = 1 - 1 / horizon
    live = (~terminated[..., 1:]).to(reward.dtype) * discount
    continuation = (~done[..., 1:]).to(reward.dtype) * lmbda
    intermediate = reward[..., 1:] + (1 - continuation) * live * bootstrap[..., 1:]
    next_return = bootstrap[..., -1]
    returns = []
    for time_index in reversed(range(intermediate.shape[-1])):
        next_return = (
            intermediate[..., time_index]
            + live[..., time_index] * continuation[..., time_index] * next_return
        )
        returns.append(next_return)
    return torch.stack(returns[::-1], -1).detach()


def _dreamer_v3_replay_two_hot_loss(
    value_prediction: torch.Tensor,
    target: torch.Tensor,
    slow_target: torch.Tensor,
    weight: torch.Tensor,
    value_bins: torch.Tensor,
    slow_critic_regularization: float,
) -> torch.Tensor:
    """Compute the tensor-only replay critic loss with slow regularization."""
    loss = two_hot_cross_entropy(value_prediction, target, value_bins)
    slow_loss = two_hot_cross_entropy(value_prediction, slow_target, value_bins)
    loss = loss + slow_critic_regularization * slow_loss
    return (weight.to(loss.dtype) * loss).mean()


class DreamerV3ValueLoss(LossModule):
    """DreamerV3 Value Loss.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for an overview
    of the online critic, slow critic, and their update flow.

    Trains the value network to predict the lambda-target computed by
    :class:`DreamerV3ActorLoss`. Supports two loss modes:

    - ``"symlog_mse"`` (default): ``(symlog(v_pred) - symlog(target))^2``
    - ``"two_hot"``: Two-hot cross-entropy over a fixed bin grid (matches the
      full DreamerV3 distribution-valued critic).

    The discount factor used here must match the one the actor used to compute
    ``lambda_target``. The recommended way to keep them in lock-step is to
    pass the actor loss to the constructor via ``actor_loss=``: the value loss
    will then read ``gamma`` from the actor's value estimator at every forward
    call. The legacy ``gamma=`` kwarg + :meth:`sync_gamma_with_actor_loss`
    pattern is still supported.

    Setting ``slow_critic_regularization`` to a positive value creates a
    checkpointed target critic. Associate a :class:`~torchrl.objectives.SoftUpdate`
    and step it after each critic optimizer step.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        value_model (TensorDictModule): The value network.
        value_loss (str, optional): Loss type — ``"symlog_mse"`` or ``"two_hot"``.
            Default: ``"symlog_mse"``.
        discount_loss (bool, optional): If ``True``, discounts the loss with
            a cumulative gamma factor. Default: ``True``.
        gamma (float, optional): Discount factor used when ``discount_loss=True``.
            Ignored if ``actor_loss`` is provided. Default: ``0.99``.
        num_value_bins (int, optional): Number of bins for ``"two_hot"`` loss.
            Default: 255.
        actor_loss (DreamerV3ActorLoss, optional): If provided, ``gamma`` is
            read from this actor loss's value estimator on every forward call,
            avoiding any chance of a mismatch. Default: ``None``.
        slow_critic_regularization (float, optional): Weight of the auxiliary
            loss that trains the online critic toward decoded target-critic
            predictions. Default: ``0.0``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import MLP
        >>> from torchrl.objectives import DreamerV3ValueLoss
        >>> value_model = TensorDictModule(
        ...     MLP(out_features=1, depth=1, num_cells=8),
        ...     in_keys=["state"],
        ...     out_keys=["state_value"],
        ... )
        >>> td = TensorDict({
        ...     "state": torch.randn(8, 4),
        ...     "lambda_target": torch.randn(8, 1),
        ... }, [8])
        >>> loss = DreamerV3ValueLoss(value_model)
        >>> loss_td, _ = loss(td)
        >>> "loss_value" in loss_td.keys()
        True
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            value (NestedKey): Decoded predicted value key. Defaults to
                ``"state_value"``.
            value_logits (NestedKey): Categorical value logits key. Defaults to
                ``"state_value_logits"``.
            discount_weight (NestedKey): Optional cumulative imagination
                weight. Defaults to ``"discount_weight"``.
        """

        value: NestedKey = "state_value"
        value_logits: NestedKey = "state_value_logits"
        discount_weight: NestedKey = "discount_weight"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    value_model: TensorDictModule
    value_model_params: TensorDictParams
    target_value_model_params: TensorDictParams

    def __init__(
        self,
        value_model: TensorDictModule,
        value_loss: str = "symlog_mse",
        discount_loss: bool = True,
        gamma: float = 0.99,
        num_value_bins: int = _DEFAULT_NUM_BINS,
        actor_loss: DreamerV3ActorLoss | None = None,
        slow_critic_regularization: float = 0.0,
    ):
        super().__init__()
        if slow_critic_regularization < 0:
            raise ValueError("slow_critic_regularization must be non-negative.")
        self.slow_critic_regularization = slow_critic_regularization
        self.convert_to_functional(
            value_model,
            "value_model",
            create_target_params=bool(slow_critic_regularization),
        )
        self.value_loss = value_loss
        self.gamma = gamma
        self.discount_loss = discount_loss
        if value_loss not in ("symlog_mse", "two_hot"):
            raise ValueError(
                f"value_loss must be 'symlog_mse' or 'two_hot', got '{value_loss}'"
            )
        # Stash without registering as a submodule (avoid double parameter ownership)
        self.__dict__["_actor_loss"] = actor_loss
        self.register_buffer("value_bins", _default_bins(num_value_bins))
        self._replay_value_target_fn = None
        self._replay_two_hot_loss_fn = None

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def _resolved_gamma(self) -> float:
        actor_loss = self.__dict__.get("_actor_loss")
        if actor_loss is None:
            return float(self.gamma)
        estimator_gamma = actor_loss.value_estimator.gamma
        if torch.is_tensor(estimator_gamma):
            estimator_gamma = estimator_gamma.item()
        return float(estimator_gamma)

    def sync_gamma_with_actor_loss(self, actor_loss: DreamerV3ActorLoss) -> None:
        """Pull ``gamma`` from an actor loss's value estimator.

        Prefer passing ``actor_loss=`` to the constructor; this method exists
        for backward compatibility with the legacy two-step setup.
        """
        estimator_gamma = actor_loss.value_estimator.gamma
        if torch.is_tensor(estimator_gamma):
            estimator_gamma = estimator_gamma.item()
        self.gamma = float(estimator_gamma)

    def compile_replay_value_loss(self, **compile_kwargs: Any) -> None:
        """Compile the fixed-shape tensor portions of the replay value loss.

        This compiles the reverse lambda-return scan and, for a two-hot value
        loss with slow-critic regularization, its categorical loss reduction.
        Sequence length and input shapes are specialized on first use.

        CUDA graphs are disabled by default because callers may retain loss
        outputs across invocations. Additional keyword arguments are passed to
        :func:`torch.compile`.
        """
        compile_kwargs.setdefault("dynamic", False)
        compile_kwargs.setdefault("options", {"triton.cudagraphs": False})
        self._replay_value_target_fn = torch.compile(
            _dreamer_v3_replay_value_target, **compile_kwargs
        )
        self._replay_two_hot_loss_fn = torch.compile(
            _dreamer_v3_replay_two_hot_loss, **compile_kwargs
        )

    def replay_value_loss(
        self,
        features: TensorDictBase,
        reward: torch.Tensor,
        done: torch.Tensor,
        terminated: torch.Tensor,
        bootstrap: torch.Tensor,
        *,
        horizon: float = 333.0,
        lmbda: float = 0.95,
    ) -> torch.Tensor:
        """Compute DreamerV3's replay-sequence critic loss.

        The replay return at each state uses the following replay reward and
        bootstraps from the first imagined return of the following state. The
        input features remain attached so this loss can train the RSSM
        representation when the model loss returns live features.

        Args:
            features: Posterior replay features with batch size ``[B, T]``.
            reward: Replay rewards aligned with ``features``, shape ``[B, T]``
                or ``[B, T, 1]``.
            done: Episode boundary flags with the same leading shape.
            terminated: True terminal flags with the same leading shape.
            bootstrap: First imagined lambda return from every replay state,
                shape ``[B, T]``.
            horizon: Continuous-discount horizon. Default: 333.
            lmbda: Lambda-return coefficient. Default: 0.95.

        Returns:
            The scalar, unscaled replay value loss.
        """
        target_fn = self._replay_value_target_fn
        if target_fn is None:
            target_fn = _dreamer_v3_replay_value_target
        target = target_fn(reward, done, terminated, bootstrap, horizon, lmbda)
        done = done.squeeze(-1)
        # Single definition of the per-step mask: the last replay step has no
        # following state to bootstrap from, and finished steps are dropped.
        # Both the compiled and the eager paths below consume exactly this.
        weight = ~done[..., :-1]

        value_tensordict = features.select(*self.value_model.in_keys, strict=False)
        with self.value_model_params.to_module(
            self.value_model, preserve_module_state=False
        ):
            self.value_model(value_tensordict)
        if self.value_loss == "two_hot":
            value_prediction = value_tensordict.get(self.tensor_keys.value_logits)
            loss = two_hot_cross_entropy(
                value_prediction[..., :-1, :], target, self.value_bins
            )
        else:
            value_prediction = value_tensordict.get(self.tensor_keys.value)
            loss = (symlog(value_prediction[..., :-1, 0]) - symlog(target)).square()

        if self.slow_critic_regularization:
            target_tensordict = features.select(*self.value_model.in_keys, strict=False)
            with torch.no_grad(), self.target_value_model_params.to_module(
                self.value_model, preserve_module_state=False
            ):
                self.value_model(target_tensordict)
            slow_target = target_tensordict.get(self.tensor_keys.value)[..., :-1, 0]
            if self.value_loss == "two_hot":
                replay_two_hot_loss_fn = self._replay_two_hot_loss_fn
                if replay_two_hot_loss_fn is not None:
                    return replay_two_hot_loss_fn(
                        value_prediction[..., :-1, :],
                        target,
                        slow_target,
                        weight,
                        self.value_bins,
                        self.slow_critic_regularization,
                    )
                slow_loss = two_hot_cross_entropy(
                    value_prediction[..., :-1, :], slow_target, self.value_bins
                )
            else:
                slow_loss = (
                    symlog(value_prediction[..., :-1, 0]) - symlog(slow_target)
                ).square()
            loss = loss + self.slow_critic_regularization * slow_loss

        return (weight.to(loss.dtype) * loss).mean()

    def _shared_value_forward(
        self, fake_data: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate online and slow critics once over H+1 imagined features."""
        if self.value_loss != "two_hot":
            raise RuntimeError(
                "Shared imagination values require value_loss='two_hot'."
            )
        feature_values = {}
        for key in self.value_model.in_keys:
            root_key = unravel_key(key)
            next_key = (
                ("next", root_key) if isinstance(root_key, str) else ("next", *root_key)
            )
            root = fake_data.get(root_key)
            feature_values[root_key] = torch.cat(
                (root[..., :1, :], fake_data.get(next_key)), dim=-2
            ).detach()
        first_feature = next(iter(feature_values.values()))
        features = TensorDict(
            feature_values,
            batch_size=first_feature.shape[:-1],
            device=first_feature.device,
        )
        online = features.copy()
        with self.value_model_params.to_module(
            self.value_model, preserve_module_state=False
        ):
            self.value_model(online)
        slow = features.copy()
        with torch.no_grad(), self.target_value_model_params.to_module(
            self.value_model, preserve_module_state=False
        ):
            self.value_model(slow)
        return (
            online.get(self.tensor_keys.value_logits),
            online.get(self.tensor_keys.value),
            slow.get(self.tensor_keys.value),
        )

    @_maybe_record_function_decorator("dreamer_v3/value_loss")
    def forward(self, fake_data) -> tuple[TensorDict, TensorDict]:
        lambda_target = fake_data.get("lambda_target")

        shared_value_pred = fake_data.get(_SHARED_VALUE_LOGITS_KEY, None)
        shared_slow_value = fake_data.get(_SHARED_SLOW_VALUE_KEY, None)
        if shared_value_pred is None:
            tensordict_select = fake_data.select(
                *self.value_model.in_keys, strict=False
            )
            with self.value_model_params.to_module(
                self.value_model, preserve_module_state=False
            ):
                self.value_model(tensordict_select)
        else:
            tensordict_select = None
        # lambda_target shape: [N, 1] (flat) or [B, T, 1] (batch x time)
        # Squeeze the trailing 1 for loss computation
        target_sq = lambda_target.squeeze(-1)  # [N] or [B, T]

        provided_discount = fake_data.get(self.tensor_keys.discount_weight, None)
        if provided_discount is not None:
            discount = provided_discount.squeeze(-1)
        elif self.discount_loss and target_sq.ndim >= 2:
            gamma = self._resolved_gamma()
            discount = gamma * torch.ones_like(target_sq)
            discount[..., 0] = 1
            discount = discount.cumprod(dim=-1)
        else:
            discount = torch.ones_like(target_sq)

        if self.value_loss == "two_hot":
            value_pred = shared_value_pred
            if value_pred is None:
                value_pred = tensordict_select.get(self.tensor_keys.value_logits, None)
            if value_pred is None:
                value_pred = tensordict_select.get(self.tensor_keys.value)
                warnings.warn(
                    "Storing DreamerV3 categorical value logits under the decoded "
                    f"value key {unravel_key(self.tensor_keys.value)!r} is deprecated "
                    "and will be removed in v0.16. Write logits to the configured "
                    "value_logits key instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if value_pred.shape[-1] != self.value_bins.shape[0]:
                raise ValueError(
                    f"value_loss='two_hot' expects the value head to output "
                    f"logits over {self.value_bins.shape[0]} bins, got trailing "
                    f"dim {value_pred.shape[-1]}."
                )
            loss = two_hot_cross_entropy(value_pred, target_sq, self.value_bins)
        else:
            # symlog MSE
            value_pred = tensordict_select.get(self.tensor_keys.value)
            loss = (symlog(value_pred.squeeze(-1)) - symlog(target_sq)).pow(2)

        if self.slow_critic_regularization:
            target_value = shared_slow_value
            if target_value is None:
                target_tensordict = fake_data.select(
                    *self.value_model.in_keys, strict=False
                )
                with torch.no_grad(), self.target_value_model_params.to_module(
                    self.value_model, preserve_module_state=False
                ):
                    self.value_model(target_tensordict)
                target_value = target_tensordict.get(self.tensor_keys.value)
            if self.value_loss == "two_hot" and (
                target_value.shape[-1] == self.value_bins.shape[0]
            ):
                target_value = two_hot_decode(target_value, self.value_bins).unsqueeze(
                    -1
                )
            if self.value_loss == "two_hot":
                slow_loss = two_hot_cross_entropy(
                    value_pred, target_value.squeeze(-1), self.value_bins
                )
            else:
                slow_loss = (
                    symlog(value_pred.squeeze(-1)) - symlog(target_value.squeeze(-1))
                ).pow(2)
            loss = loss + self.slow_critic_regularization * slow_loss
            slow_metric = (discount * slow_loss).mean().detach()
        else:
            # No slow-critic term: report a zero rather than allocating and
            # reducing a zero tensor on every call.
            slow_metric = torch.zeros((), device=loss.device, dtype=loss.dtype)

        value_loss = (discount * loss).mean()

        loss_tensordict = TensorDict(
            {
                "loss_value": value_loss,
                "value_slow_loss": slow_metric,
            }
        )
        self._clear_weakrefs(fake_data, loss_tensordict)
        return loss_tensordict, fake_data.data
