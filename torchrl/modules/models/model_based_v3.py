# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 RSSM components: discrete categorical latent state.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import math

import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.utils import unravel_key
from torch import nn
from torch.nn import GRUCell


class BlockLinear(nn.Module):
    """Block-diagonal linear layer: ``blocks`` independent linears over grouped features.

    Splits the input feature axis into ``blocks`` contiguous groups, applies an
    independent ``(in/blocks -> out/blocks)`` linear to each, and concatenates.
    This is DreamerV3's ``BlockLinear`` (``rssm.py`` ``_core``), used inside the
    block GRU to cut the parameter count of the large deterministic state.

    Args:
        in_features (int): Total input features (must be divisible by ``blocks``).
        out_features (int): Total output features (must be divisible by ``blocks``).
        blocks (int): Number of independent blocks.
        device (torch.device, optional): Device. Defaults to None.
    """

    def __init__(
        self, in_features: int, out_features: int, blocks: int, device=None
    ):
        super().__init__()
        if in_features % blocks or out_features % blocks:
            raise ValueError(
                f"in_features ({in_features}) and out_features ({out_features}) "
                f"must both be divisible by blocks ({blocks})."
            )
        self.blocks = blocks
        self.in_per = in_features // blocks
        self.out_per = out_features // blocks
        self.weight = nn.Parameter(
            torch.empty(blocks, self.in_per, self.out_per, device=device)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        bound = 1.0 / math.sqrt(self.in_per)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[:-1]
        x = x.reshape(*batch, self.blocks, 1, self.in_per)
        # Broadcast matmul rather than ``einsum``: identical arithmetic, but
        # einsum re-parses its equation on every call, which is significant on
        # a recurrence that runs this once per timestep.
        out = torch.matmul(x, self.weight).reshape(
            *batch, self.blocks * self.out_per
        )
        return out + self.bias


class RSSMPriorV3(nn.Module):
    """DreamerV3 prior network with discrete categorical latent state.

    Implements the sequence model and dynamics predictor from DreamerV3.
    The GRU updates the deterministic hidden state:

    .. code-block:: text

        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

    Then the prior predicts a distribution over the stochastic latent:

    .. code-block:: text

        z_hat_t ~ Cat(MLP(h_t))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        action_spec (TensorSpec, optional): Action spec. Used only to read
            ``action_spec.shape``; mutually exclusive with ``action_shape``.
        action_shape (torch.Size or tuple of int, optional): Action tensor
            shape. Mutually exclusive with ``action_spec``.
        hidden_dim (int, optional): Hidden dimension of the linear projector.
            Defaults to 512.
        rnn_hidden_dim (int, optional): GRU hidden state dimension (belief size).
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables in the
            discrete latent. Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        action_dim (int, optional): Action dimension. If provided (along with
            ``num_categoricals * num_classes``), uses explicit ``nn.Linear``
            instead of ``nn.LazyLinear``. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.
        unimix (float, optional): Uniform-mixture weight for the categorical
            latent (DreamerV3 ``unimix`` — mixes ``unimix`` uniform into the
            softmax before sampling). Defaults to 0.0.
        jax_core (bool, optional): If ``True``, use DreamerV3's block GRU
            (``sigmoid(update - 1)`` remember bias, action soft-clip, normed
            input projections, block-diagonal linears) instead of ``nn.GRUCell``.
            Defaults to ``False``.
        blocks (int, optional): Number of blocks for the block GRU when
            ``jax_core=True``; ``rnn_hidden_dim`` must be divisible by it.
            Defaults to 8.
        norm (bool, optional): If ``True``, apply RMSNorm inside the prior head
            (DreamerV3 ``norm: rms``). Defaults to ``False``.
        img_layers (int, optional): Number of hidden layers in the prior
            (dynamics-predictor) head, DreamerV3's ``rssm.imglayers``. The
            reference sets ``2``; the default of ``1`` is kept for backward
            compatibility. Defaults to 1.
        norm_eps (float, optional): ``eps`` of the RMSNorm layers. DreamerV3
            uses ``1e-4`` (``embodied/jax/nets.py`` ``Norm``); the default of
            ``None`` keeps :class:`torch.nn.RMSNorm`'s own default. Defaults to
            ``None``.

    Examples:
        >>> import torch
        >>> from torchrl.modules.models.model_based_v3 import RSSMPriorV3
        >>> prior = RSSMPriorV3(
        ...     action_shape=torch.Size([2]),
        ...     hidden_dim=16,
        ...     rnn_hidden_dim=8,
        ...     num_categoricals=4,
        ...     num_classes=4,
        ...     action_dim=2,
        ... )
        >>> state = torch.zeros(3, 16)
        >>> belief = torch.zeros(3, 8)
        >>> action = torch.randn(3, 2)
        >>> logits, next_state, next_belief = prior(state, belief, action)
        >>> logits.shape, next_state.shape, next_belief.shape
        (torch.Size([3, 4, 4]), torch.Size([3, 16]), torch.Size([3, 8]))
    """

    def __init__(
        self,
        action_spec=None,
        hidden_dim: int = 512,
        rnn_hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        action_dim: int | None = None,
        device=None,
        *,
        action_shape: torch.Size | tuple[int, ...] | None = None,
        unimix: float = 0.0,
        jax_core: bool = False,
        blocks: int = 8,
        norm: bool = False,
        img_layers: int = 1,
        norm_eps: float | None = None,
    ):
        super().__init__()
        if action_spec is not None and action_shape is not None:
            raise ValueError(
                "Pass only one of `action_spec` or `action_shape`, not both."
            )
        self.unimix = unimix
        self.jax_core = jax_core
        self.blocks = blocks
        if action_spec is not None:
            self.action_shape = torch.Size(action_spec.shape)
        elif action_shape is not None:
            self.action_shape = torch.Size(action_shape)
        else:
            self.action_shape = None

        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.rnn_hidden_dim = rnn_hidden_dim
        state_dim = num_categoricals * num_classes
        norm_kwargs = {} if norm_eps is None else {"eps": norm_eps}

        if jax_core:
            # DreamerV3 block GRU (rssm.py:_core): normed input projections for
            # deter/stoch/action, block-diagonal linears, and the gated update
            # with the sigmoid(update - 1) remember bias.
            if rnn_hidden_dim % blocks:
                raise ValueError(
                    f"jax_core requires rnn_hidden_dim ({rnn_hidden_dim}) "
                    f"divisible by blocks ({blocks})."
                )
            self.dynin0 = nn.Linear(rnn_hidden_dim, hidden_dim, device=device)
            self.dynin1 = nn.Linear(state_dim, hidden_dim, device=device)
            if action_dim is not None:
                self.dynin2 = nn.Linear(action_dim, hidden_dim, device=device)
            else:
                self.dynin2 = nn.LazyLinear(hidden_dim, device=device)
            self.dynin_norm = nn.ModuleList(
                [
                    nn.RMSNorm(hidden_dim, device=device, **norm_kwargs)
                    for _ in range(3)
                ]
            )
            self.dynhid = BlockLinear(
                rnn_hidden_dim + 3 * blocks * hidden_dim,
                rnn_hidden_dim,
                blocks,
                device=device,
            )
            self.dynhid_norm = nn.RMSNorm(rnn_hidden_dim, device=device, **norm_kwargs)
            self.dyngru = BlockLinear(
                rnn_hidden_dim, 3 * rnn_hidden_dim, blocks, device=device
            )
            self.core_act = nn.SiLU()
        else:
            self.rnn = GRUCell(hidden_dim, rnn_hidden_dim, device=device)

            if action_dim is not None:
                projector_in = state_dim + action_dim
                first_linear = nn.Linear(projector_in, hidden_dim, device=device)
            else:
                first_linear = nn.LazyLinear(hidden_dim, device=device)
            self.action_state_projector = nn.Sequential(first_linear, nn.SiLU())

        # DreamerV3 ``_prior``: ``imglayers`` blocks of Linear -> Norm -> act,
        # then the logit layer (rssm.py:_prior).
        prior_layers = []
        in_features = rnn_hidden_dim
        for _ in range(img_layers):
            prior_layers.append(nn.Linear(in_features, hidden_dim, device=device))
            if norm:
                prior_layers.append(nn.RMSNorm(hidden_dim, device=device, **norm_kwargs))
            prior_layers.append(nn.SiLU())
            in_features = hidden_dim
        prior_layers.append(
            nn.Linear(in_features, num_categoricals * num_classes, device=device)
        )
        self.rnn_to_prior_projector = nn.Sequential(*prior_layers)

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prior distribution and update GRU belief.

        Args:
            state: Previous stochastic state, shape ``[..., num_categoricals * num_classes]``.
            belief: Previous GRU hidden state, shape ``[..., rnn_hidden_dim]``.
            action: Current action, shape ``[..., action_dim]``.

        Returns:
            prior_logits (torch.Tensor): Raw logits, shape
                ``[..., num_categoricals, num_classes]``.
            state (torch.Tensor): Sampled state (straight-through), shape
                ``[..., num_categoricals * num_classes]``.
            belief (torch.Tensor): Updated GRU hidden state, shape
                ``[..., rnn_hidden_dim]``.
        """
        if self.jax_core:
            belief = self._jax_core(state, belief, action)
        else:
            projector_input = torch.cat([state, action], dim=-1)
            action_state = self.action_state_projector(projector_input)

            # Run GRU in fp32 to avoid cuBLAS dispatch issues under autocast
            dtype = action_state.dtype
            device_type = action_state.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                belief = self.rnn(
                    action_state.float(),
                    belief.float() if belief is not None else None,
                )
            belief = belief.to(dtype)

        prior_logits_flat = self.rnn_to_prior_projector(belief)
        prior_logits = prior_logits_flat.view(
            *prior_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )

        state = _straight_through_categorical(prior_logits, self.unimix)
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)

        return prior_logits, state, belief

    def belief_and_logits(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward`` without sampling the prior latent.

        Returns ``(prior_logits, belief)``. The observation pass never uses the
        prior's *sample* -- the posterior replaces it -- and the reference does
        not draw one either (``rssm.py`` ``_observe`` calls ``_prior`` only for
        the logits). Skipping it saves a categorical sample per timestep.
        """
        if self.jax_core:
            belief = self._jax_core(state, belief, action)
        else:
            projector_input = torch.cat([state, action], dim=-1)
            action_state = self.action_state_projector(projector_input)
            dtype = action_state.dtype
            device_type = action_state.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                belief = self.rnn(
                    action_state.float(),
                    belief.float() if belief is not None else None,
                )
            belief = belief.to(dtype)
        prior_logits_flat = self.rnn_to_prior_projector(belief)
        prior_logits = prior_logits_flat.view(
            *prior_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )
        return prior_logits, belief

    def _jax_core(
        self, stoch: torch.Tensor, deter: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """DreamerV3 block-GRU update of the deterministic state (rssm.py:_core).

        Args:
            stoch: Previous stochastic state ``[..., state_dim]``.
            deter: Previous deterministic state / belief ``[..., rnn_hidden_dim]``.
            action: Current action ``[..., action_dim]``.

        Returns:
            The updated deterministic state ``[..., rnn_hidden_dim]``.
        """
        g = self.blocks
        d = deter.shape[-1]
        # Action soft-clip a / max(1, |a|) (stop-grad on the divisor).
        action = action / torch.clamp(action.abs(), min=1.0).detach()
        x0 = self.core_act(self.dynin_norm[0](self.dynin0(deter)))
        x1 = self.core_act(self.dynin_norm[1](self.dynin1(stoch)))
        x2 = self.core_act(self.dynin_norm[2](self.dynin2(action)))
        x = torch.cat([x0, x1, x2], dim=-1)  # [..., 3 * hidden]
        # Broadcast the projections across the g blocks, prepend the per-block
        # slice of deter, then flatten back to [..., d + 3 * g * hidden].
        x = x.unsqueeze(-2).expand(*x.shape[:-1], g, x.shape[-1])
        deter_grp = deter.reshape(*deter.shape[:-1], g, d // g)
        x = torch.cat([deter_grp, x], dim=-1)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        x = self.core_act(self.dynhid_norm(self.dynhid(x)))  # [..., d]
        x = self.dyngru(x)  # [..., 3 * d]
        # Split per block into reset / cand / update gates.
        xg = x.reshape(*x.shape[:-1], g, 3 * (d // g))
        reset, cand, update = torch.chunk(xg, 3, dim=-1)
        reset = torch.sigmoid(reset.reshape(*deter.shape))
        cand = torch.tanh(reset * cand.reshape(*deter.shape))
        # sigmoid(update - 1): bias the update gate toward remembering.
        update = torch.sigmoid(update.reshape(*deter.shape) - 1.0)
        return update * cand + (1 - update) * deter


class RSSMPosteriorV3(nn.Module):
    """DreamerV3 posterior (representation model) with discrete categorical latent.

    Given the deterministic hidden state ``h_t`` and an observation embedding
    ``e_t``, produces the posterior distribution over the stochastic latent:

    .. code-block:: text

        z_t ~ Cat(MLP([h_t, e_t]))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        hidden_dim (int, optional): Hidden dimension of the projector MLP.
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables.
            Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        rnn_hidden_dim (int, optional): Belief dimension. If provided along with
            ``obs_embed_dim``, uses explicit ``nn.Linear``. Defaults to None.
        obs_embed_dim (int, optional): Observation embedding dimension. If provided
            along with ``rnn_hidden_dim``, uses explicit ``nn.Linear``. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.
        unimix (float, optional): Uniform-mixture weight for the categorical
            latent (DreamerV3 ``unimix``). Defaults to 0.0.
        norm_eps (float, optional): ``eps`` of the RMSNorm layer. DreamerV3 uses
            ``1e-4``; ``None`` keeps :class:`torch.nn.RMSNorm`'s default.
            Defaults to ``None``.
        norm (bool, optional): If ``True``, apply RMSNorm inside the posterior
            projector (DreamerV3 ``norm: rms``). Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from torchrl.modules.models.model_based_v3 import RSSMPosteriorV3
        >>> posterior = RSSMPosteriorV3(
        ...     hidden_dim=16,
        ...     num_categoricals=4,
        ...     num_classes=4,
        ...     rnn_hidden_dim=8,
        ...     obs_embed_dim=12,
        ... )
        >>> belief = torch.randn(3, 8)
        >>> obs_embed = torch.randn(3, 12)
        >>> logits, state = posterior(belief, obs_embed)
        >>> logits.shape, state.shape
        (torch.Size([3, 4, 4]), torch.Size([3, 16]))
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        rnn_hidden_dim: int | None = None,
        obs_embed_dim: int | None = None,
        device=None,
        unimix: float = 0.0,
        norm: bool = False,
        norm_eps: float | None = None,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.unimix = unimix
        norm_kwargs = {} if norm_eps is None else {"eps": norm_eps}

        if rnn_hidden_dim is not None and obs_embed_dim is not None:
            projector_in = rnn_hidden_dim + obs_embed_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)

        post_layers = [first_linear]
        if norm:
            post_layers.append(nn.RMSNorm(hidden_dim, device=device, **norm_kwargs))
        post_layers += [
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes, device=device),
        ]
        self.obs_rnn_to_post_projector = nn.Sequential(*post_layers)

    def forward(
        self,
        belief: torch.Tensor,
        obs_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution given belief and observation embedding.

        Args:
            belief: Deterministic GRU hidden state from prior, shape
                ``[..., rnn_hidden_dim]``.
            obs_embedding: Encoded observation, shape ``[..., obs_embed_dim]``.

        Returns:
            posterior_logits (torch.Tensor): Raw logits, shape
                ``[..., num_categoricals, num_classes]``.
            state (torch.Tensor): Sampled state (straight-through), shape
                ``[..., num_categoricals * num_classes]``.
        """
        post_logits_flat = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        posterior_logits = post_logits_flat.view(
            *post_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )
        state = _straight_through_categorical(posterior_logits, self.unimix)
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)
        return posterior_logits, state


class RSSMRolloutV3(TensorDictModuleBase):
    """Roll out the DreamerV3 RSSM over a sequence.

    Given encoded observations and actions for ``T`` time steps, this module
    runs the prior (GRU + categorical) then the posterior (categorical) at each
    step and returns a stacked TensorDict of all intermediate states.

    The previous posterior state ``z_t`` is used as the prior input for step
    ``t+1``, matching the recurrent structure of DreamerV3.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        rssm_prior (TensorDictModule): Prior module wrapping :class:`RSSMPriorV3`.
        rssm_posterior (TensorDictModule): Posterior module wrapping
            :class:`RSSMPosteriorV3`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.model_based_v3 import (
        ...     RSSMPosteriorV3, RSSMPriorV3, RSSMRolloutV3,
        ... )
        >>> prior = TensorDictModule(
        ...     RSSMPriorV3(action_shape=torch.Size([2]), hidden_dim=8,
        ...                 rnn_hidden_dim=8, num_categoricals=4, num_classes=4,
        ...                 action_dim=2),
        ...     in_keys=["state", "belief", "action"],
        ...     out_keys=[("next", "prior_logits"), ("next", "state"), ("next", "belief")],
        ... )
        >>> posterior = TensorDictModule(
        ...     RSSMPosteriorV3(hidden_dim=8, num_categoricals=4, num_classes=4,
        ...                     rnn_hidden_dim=8, obs_embed_dim=6),
        ...     in_keys=[("next", "belief"), ("next", "encoded_latents")],
        ...     out_keys=[("next", "posterior_logits"), ("next", "state")],
        ... )
        >>> rollout = RSSMRolloutV3(prior, posterior)
        >>> td = TensorDict({
        ...     "state": torch.zeros(2, 4, 16),
        ...     "belief": torch.zeros(2, 4, 8),
        ...     "action": torch.randn(2, 4, 2),
        ...     "next": {"encoded_latents": torch.randn(2, 4, 6)},
        ... }, [2, 4])
        >>> out = rollout(td)
        >>> out.shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        rssm_prior: TensorDictModule,
        rssm_posterior: TensorDictModule,
    ):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self._fast_path = self._check_fast_path()
        self._scan_fn = None

    def _check_fast_path(self) -> bool:
        """Whether the standard V3 wiring is in place (see :meth:`_forward_fast`)."""
        keys = lambda module, attr: [unravel_key(k) for k in getattr(module, attr)]
        return (
            isinstance(getattr(self.rssm_prior, "module", None), RSSMPriorV3)
            and isinstance(getattr(self.rssm_posterior, "module", None), RSSMPosteriorV3)
            and keys(self.rssm_prior, "in_keys") == ["state", "belief", "action"]
            and keys(self.rssm_prior, "out_keys")
            == [("next", "prior_logits"), ("next", "state"), ("next", "belief")]
            and keys(self.rssm_posterior, "in_keys")
            == [("next", "belief"), ("next", "encoded_latents")]
            and keys(self.rssm_posterior, "out_keys")
            == [("next", "posterior_logits"), ("next", "state")]
        )

    def forward(self, tensordict):
        """Roll out the RSSM for one episode chunk.

        Args:
            tensordict (TensorDictBase): Input with shape ``[*batch, T]`` containing
                actions, encoded observations, and initial state/belief.

        Returns:
            TensorDictBase: Stacked outputs with shape ``[*batch, T]``.
        """
        if self._fast_path:
            return self._forward_fast(tensordict)
        tensordict_out = []
        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        _tensordict = update_values[0]

        # Cache the keys we want to keep; they're constant across timesteps.
        output_keys = list(
            update_values[0].keys(include_nested=True, leaves_only=True)
        ) + list(self.out_keys)

        for t in range(time_steps):
            self.rssm_prior(_tensordict)
            self.rssm_posterior(_tensordict)

            tensordict_out.append(_tensordict.select(*output_keys, strict=False))
            if t < time_steps - 1:
                next_state = _tensordict.get(("next", "state"))
                next_belief = _tensordict.get(("next", "belief"))
                _tensordict = update_values[t + 1]
                _tensordict.set("state", next_state)
                _tensordict.set("belief", next_belief)

        return torch.stack(tensordict_out, tensordict.ndim - 1)

    def _forward_fast(self, tensordict):
        """Tensor-only rollout, mathematically identical to :meth:`forward`.

        The generic loop pays TensorDict overhead (a module dispatch, a
        ``select`` and two ``set`` calls per timestep, then a stack of ``T``
        tensordicts) on every one of the ``T`` recurrent steps, which for a
        DreamerV3-sized RSSM dwarfs the actual GPU work by more than an order of
        magnitude. Here the recurrence runs on raw tensors and only the four
        produced keys are stacked, once.
        """
        action = tensordict.get("action")
        embed = tensordict.get(("next", "encoded_latents"))
        state = tensordict.get("state")[..., 0, :]
        belief = tensordict.get("belief")[..., 0, :]

        scan = self._scan_fn if self._scan_fn is not None else self._scan
        states_in, beliefs_in, prior_logits, posterior_logits, states, beliefs = scan(
            state, belief, action, embed
        )

        out = tensordict.exclude(*self.out_keys)
        out.set("state", states_in)
        out.set("belief", beliefs_in)
        out.set(("next", "prior_logits"), prior_logits)
        out.set(("next", "posterior_logits"), posterior_logits)
        out.set(("next", "state"), states)
        out.set(("next", "belief"), beliefs)
        return out

    def _scan(self, state, belief, action, embed):
        """Pure-tensor RSSM recurrence over the time axis of ``action``/``embed``.

        Kept free of TensorDict operations so it can be handed to
        :func:`torch.compile` (see :meth:`compile_scan`).
        """
        prior_net = self.rssm_prior.module
        posterior_net = self.rssm_posterior.module
        states_in, beliefs_in = [], []
        prior_logits, posterior_logits, next_states, next_beliefs = [], [], [], []
        for t in range(action.shape[-2]):
            states_in.append(state)
            beliefs_in.append(belief)
            # The prior's own sample is discarded by the posterior, so it is
            # never drawn (matching the reference's observation pass).
            logits, belief = prior_net.belief_and_logits(
                state, belief, action[..., t, :]
            )
            post_logits, state = posterior_net(belief, embed[..., t, :])
            prior_logits.append(logits)
            posterior_logits.append(post_logits)
            next_states.append(state)
            next_beliefs.append(belief)
        return (
            torch.stack(states_in, -2),
            torch.stack(beliefs_in, -2),
            torch.stack(prior_logits, -3),
            torch.stack(posterior_logits, -3),
            torch.stack(next_states, -2),
            torch.stack(next_beliefs, -2),
        )

    def compile_scan(self, **compile_kwargs) -> None:
        """``torch.compile`` the recurrence, unrolled over the sequence length.

        The RSSM step is a few dozen tiny kernels, so an eager rollout is
        dominated by per-op launch overhead rather than by arithmetic. Compiling
        the unrolled scan lets inductor fuse across timesteps (measured ~3x on
        the forward pass for a 64-step DreamerV3 rollout). The trade is a large
        one-off compile (minutes for a long sequence), so it pays off for
        training runs and not for short smoke tests. Requires a fixed sequence
        length; ``dynamic=False`` is the default for that reason.
        """
        compile_kwargs.setdefault("dynamic", False)
        self._scan_fn = torch.compile(self._scan, **compile_kwargs)


def _straight_through_categorical(
    logits: torch.Tensor, unimix: float = 0.0
) -> torch.Tensor:
    """Sample from categorical with straight-through gradient estimator.

    Forward: hard one-hot sample.
    Backward: gradients flow through the soft probabilities.

    Args:
        logits: ``[..., num_categoricals, num_classes]``
        unimix: Uniform-mixture weight (DreamerV3 ``unimix``). If non-zero, the
            categorical probs become ``(1 - unimix) * softmax + unimix / classes``
            for both sampling and the straight-through gradient. Default: 0.0.

    Returns:
        one_hot tensor with same shape, gradients through softmax.
    """
    probs = torch.softmax(logits, dim=-1)
    if unimix:
        probs = (1 - unimix) * probs + unimix / probs.shape[-1]
    # Gumbel-max is exactly categorical sampling, and avoids building a
    # ``torch.distributions.Categorical`` (argument validation plus several
    # extra kernels) once per RSSM timestep on the hot path.
    with torch.no_grad():
        uniform = torch.rand_like(probs)
        gumbel = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(probs.dtype).tiny)))
        indices = (probs.log() + gumbel).argmax(dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    # Straight-through: forward = one_hot, backward gradient = grad(probs).
    return probs + (one_hot - probs).detach()
