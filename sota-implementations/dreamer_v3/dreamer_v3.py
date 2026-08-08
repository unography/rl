# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 end-to-end training script (Gym or DM Control, proprioceptive).

``env.backend`` selects the task family: ``gym`` (e.g. ``Pendulum-v1``) or
``dmc`` (e.g. ``walker`` / ``walk``). Both are state-based (not pixel-based) to
keep the script compact — observations are treated as a flat feature vector with
``global_average=True`` in the model loss; DM Control's several proprio keys are
concatenated into one ``observation``. The wiring is still real:

- collector to replay buffer of sequences
- world model = MLP encoder + RSSMPriorV3 + RSSMPosteriorV3 + MLP decoder + reward head
- RSSM unrolled over each sequence via ``RSSMRolloutV3``
- actor trained via REINFORCE in imagination (``DreamerV3ActorLoss``)
- value trained on the same imagined rollout (``DreamerV3ValueLoss``)
- periodic eval rollouts in the real env, episode reward logged

Plots ``dreamer_v3_pendulum.png`` with two curves: (a) average eval reward,
(b) world-model KL / reconstruction / reward losses.

Usage::

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        collector.total_frames=5000 logger.eval_every=500
"""
from __future__ import annotations

import copy
import importlib.util
import math

import hydra
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import (
    ActionScaling,
    CatTensors,
    Compose,
    DoubleToFloat,
    InitTracker,
    SerialEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeSequential, WorldModelWrapper
from torchrl.modules.distributions import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based_v3 import (
    BlockLinear,
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.modules.models.models import MLP
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
    symexp,
    symlog,
    two_hot_decode,
)
from torchrl.objectives.dreamer_v3 import _default_bins
from torchrl.objectives.utils import ValueEstimators

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


def make_env(cfg: DictConfig, seed: int = 0):
    """Build the real environment for either a Gym task or a DM Control task.

    ``cfg.env.backend`` selects the library:

    - ``gym``: ``GymEnv(cfg.env.name)`` -- the observation is already a single
      flat ``observation`` vector.
    - ``dmc``: ``DMControlEnv(cfg.env.domain, cfg.env.task)`` with proprioceptive
      observations. DM Control returns several float64 observation keys (e.g.
      ``orientations`` / ``height`` / ``velocity`` for walker); we concatenate
      them (sorted, for a stable order) into one float32 ``observation`` vector
      so everything downstream -- which keys on ``observation`` -- is unchanged.

    The action space is normalized to [-1, 1] via ``ActionScaling`` (DreamerV3
    wraps envs this way): the policy emits [-1, 1] and it is rescaled to the
    env's native range, so the RSSM action soft-clip is a no-op and actions
    stored in the buffer are the normalized [-1, 1] ones. For DMC this is an
    identity map, since its native action range is already [-1, 1].
    """
    backend = cfg.env.get("backend", "gym")
    device = cfg.env.get("device", "cpu")
    if backend == "gym":
        env = TransformedEnv(GymEnv(cfg.env.name, device=device), StepCounter())
    elif backend == "dmc":
        base = DMControlEnv(
            cfg.env.domain, cfg.env.task, from_pixels=False, device=device
        )
        obs_keys = sorted(base.observation_spec.keys(True, True))
        env = TransformedEnv(
            base,
            Compose(
                CatTensors(in_keys=obs_keys, out_key="observation", del_keys=True),
                DoubleToFloat(),  # dm_control emits float64; the nets want float32
                StepCounter(),
            ),
        )
    else:
        raise ValueError(
            f"unknown env.backend={backend!r}, expected 'gym' or 'dmc'"
        )
    env.append_transform(ActionScaling())
    env.set_seed(seed)
    return env


class DreamerV3Optimizer(torch.optim.Optimizer):
    """DreamerV3's optimizer: AGC -> RMS scaling -> momentum -> warmup LR.

    Faithful port of Hafner's optax chain (agent.py:_make_opt,
    embodied/jax/opt.py): adaptive gradient clipping (``clip_by_agc``), then
    ``scale_by_rms`` (LaProp-style — the gradient is RMS-normalized *before*
    momentum, unlike Adam), then ``scale_by_momentum``, with a linear LR warmup.

    Args:
        params: Parameters to optimize.
        lr (float): Peak learning rate (reference: 4e-5). Default: 4e-5.
        agc (float): Adaptive-gradient-clip ratio; 0 disables. Default: 0.3.
        agc_pmin (float): Floor on the parameter norm in AGC. Default: 1e-3.
        beta1 (float): Momentum decay. Default: 0.9.
        beta2 (float): RMS decay. Default: 0.999.
        eps (float): RMS epsilon. Default: 1e-20.
        warmup (int): Linear LR warmup steps. Default: 1000.
    """

    def __init__(
        self,
        params,
        lr: float = 4e-5,
        agc: float = 0.3,
        agc_pmin: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        warmup: int = 1000,
    ):
        defaults = dict(
            lr=lr,
            agc=agc,
            agc_pmin=agc_pmin,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            warmup=warmup,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # noqa: D102
        for group in self.param_groups:
            b1, b2, eps = group["beta1"], group["beta2"], group["eps"]
            agc, pmin, warmup = group["agc"], group["agc_pmin"], group["warmup"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["nu"] = torch.zeros_like(p)
                    state["mu"] = torch.zeros_like(p)
                # Adaptive gradient clipping (clip_by_agc).
                if agc:
                    unorm = g.norm()
                    upper = agc * p.norm().clamp_min(pmin)
                    g = g * (1.0 / (unorm / (upper + 1e-12)).clamp_min(1.0))
                state["step"] += 1
                step = state["step"]
                nu, mu = state["nu"], state["mu"]
                # scale_by_rms (RMS-normalize the gradient, with bias correction).
                nu.mul_(b2).addcmul_(g, g, value=1 - b2)
                nu_hat = nu / (1 - b2**step)
                g = g / (nu_hat.sqrt() + eps)
                # scale_by_momentum (momentum of the normalized gradient).
                mu.mul_(b1).add_(g, alpha=1 - b1)
                mu_hat = mu / (1 - b1**step)
                lr_t = group["lr"] * (min(1.0, step / warmup) if warmup else 1.0)
                p.add_(mu_hat, alpha=-lr_t)


class TwoHotRewardDecoder(nn.Module):
    """Turn the two-hot reward head into a scalar reward for imagination.

    The world-model reward head emits logits over ``num_reward_bins``. The
    imagination :class:`DreamerEnv` needs a scalar reward, so we take the
    distribution's expectation over the bin grid -- and invert ``symlog`` when
    the grid lives in symlog space. Wrapping the *same* ``reward_mlp`` module
    keeps the imagined reward in lock-step with the trained world model.
    """

    def __init__(
        self,
        reward_mlp: nn.Module,
        reward_bins: torch.Tensor,
        bin_space: str = "symlog",
    ):
        super().__init__()
        self.reward_mlp = reward_mlp
        self.bin_space = bin_space
        self.register_buffer("reward_bins", reward_bins)

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        logits = self.reward_mlp(torch.cat([state, belief], dim=-1))
        reward = two_hot_decode(logits, self.reward_bins)
        if self.bin_space == "symlog":
            reward = symexp(reward)
        return reward.unsqueeze(-1)


class SymlogEncoder(nn.Module):
    """Encoder MLP over symlog-compressed vector observations.

    DreamerV3 squashes every non-image observation with ``symlog`` before the
    encoder MLP (``rssm.py`` ``SimpleEncoder.__call__``: ``squish = nn.symlog``)
    and reconstructs the symlog-space target (``symlog_mse`` head). Feeding raw
    observations instead leaves the encoder to cope with unbounded inputs -- for
    ``walker`` the ``velocity`` entries reach a couple of orders of magnitude
    beyond ``orientations``/``height``.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(symlog(obs))


# DreamerV3 RMSNorm eps (embodied/jax/nets.py ``Norm.eps``). torch's RMSNorm
# defaults to the dtype epsilon (~1e-7), which is four orders of magnitude
# smaller and lets tiny-magnitude activations blow up on renormalization.
_NORM_EPS = 1e-4


def _norm_mlp(in_features: int, out_features: int, cfg: DictConfig, depth=None):
    """MLP with SiLU activation + RMSNorm after each hidden layer (JAX-faithful).

    DreamerV3 uses ``act: silu, norm: rms`` throughout (configs.yaml); TorchRL's
    bare ``MLP`` defaults to Tanh and no norm.

    ``depth`` overrides ``cfg.networks.depth`` for the heads whose JAX ``layers``
    differs from the shared trunk width (``rewhead``/``conhead``: 1).
    """
    return MLP(
        in_features=in_features,
        out_features=out_features,
        depth=cfg.networks.depth if depth is None else depth,
        num_cells=cfg.networks.hidden_dim,
        activation_class=nn.SiLU,
        norm_class=nn.RMSNorm,
        norm_kwargs={
            "normalized_shape": cfg.networks.hidden_dim,
            "eps": _NORM_EPS,
        },
    )


def _mlp_trunk(in_features: int, cfg: DictConfig):
    """JAX ``nn.MLP``: ``layers`` blocks of Linear -> RMSNorm -> SiLU, no output layer.

    The DreamerV3 encoder *is* this trunk -- its "embedding" is the last hidden
    activation (``rssm.py`` ``Encoder.__call__``), not a further linear
    projection. Emitting one is an extra unnormalized, unactivated layer the
    reference does not have.
    """
    units = cfg.networks.hidden_dim
    layers = []
    for _ in range(cfg.networks.depth):
        layers += [
            nn.Linear(in_features, units),
            nn.RMSNorm(units, eps=_NORM_EPS),
            nn.SiLU(),
        ]
        in_features = units
    return nn.Sequential(*layers)


def _apply_jax_init(module: nn.Module, outscale: float | None = None) -> nn.Module:
    """Initialize every ``nn.Linear`` the way DreamerV3 does, and scale the output layer.

    JAX ``winit: trunc_normal_in`` (``embodied/jax/nets.py`` ``Initializer``) draws
    from a truncated normal on ``[-2, 2]`` scaled by ``1.1368 * sqrt(1 / fan_in)``
    (the factor undoes the truncation's variance shrinkage, so the result has
    std ``~1/sqrt(fan_in)``), and biases start at zero. torch's default is
    ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` for both -- std ``0.577/sqrt(fan_in)``,
    i.e. 1.7x narrower, with non-zero biases.

    ``outscale`` additionally multiplies the *last* linear's weights, matching
    the reference's per-head ``outscale``: ``0.0`` for the reward and value heads
    (so both predict exactly zero at init) and ``0.01`` for the policy.
    """
    linears = [m for m in module.modules() if isinstance(m, (nn.Linear, BlockLinear))]
    for linear in linears:
        if isinstance(linear, BlockLinear):
            # JAX ``compute_fans`` on a (blocks, in_per, out_per) kernel gives
            # fan_in = in_per * blocks, i.e. the *total* input width.
            fan_in = linear.in_per * linear.blocks
        else:
            fan_in = linear.weight.shape[1]
        with torch.no_grad():
            nn.init.trunc_normal_(linear.weight, std=1.0, a=-2.0, b=2.0)
            linear.weight.mul_(1.1368 * math.sqrt(1.0 / fan_in))
            if linear.bias is not None:
                linear.bias.zero_()
    if outscale is not None and linears:
        with torch.no_grad():
            linears[-1].weight.mul_(outscale)
    return module


def build_shared_modules(*, cfg: DictConfig, action_dim: int):
    """Create the RSSM prior + reward head **once**.

    The same module instances are handed to both the world model (which trains
    them) and the imagination :class:`DreamerEnv` (which rolls out with them),
    so imagination always reflects the *learned* dynamics and reward instead of
    an independent, frozen network.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    prior_net = RSSMPriorV3(
        action_shape=torch.Size([action_dim]),
        hidden_dim=cfg.networks.hidden_dim,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        action_dim=action_dim,
        unimix=cfg.networks.unimix,
        jax_core=cfg.networks.jax_core,
        blocks=cfg.networks.blocks,
        norm=cfg.networks.jax_core,
        # JAX rssm.imglayers: the dynamics predictor has 2 hidden layers.
        img_layers=cfg.networks.get("img_layers", 2),
        norm_eps=_NORM_EPS,
    )
    _apply_jax_init(prior_net)
    # JAX rewhead/conhead: `layers: 1` (the size1m overlay only changes `units`),
    # and `rewhead.outscale: 0.0` -- a zero-initialized output layer, so the
    # two-hot reward distribution starts uniform and predicts exactly 0.
    head_depth = cfg.networks.get("head_depth", 1)
    reward_mlp = _apply_jax_init(
        _norm_mlp(
            state_dim + cfg.networks.rnn_hidden_dim,
            cfg.networks.num_reward_bins,
            cfg,
            depth=head_depth,
        ),
        outscale=0.0,
    )
    # Continue (termination) head, shared by the world model (BCE against
    # 1 - terminated) and the imagination discount in the actor loss.
    # JAX conhead.outscale: 1.0.
    continue_mlp = _apply_jax_init(
        _norm_mlp(state_dim + cfg.networks.rnn_hidden_dim, 1, cfg, depth=head_depth)
    )
    return prior_net, reward_mlp, continue_mlp


def build_world_model(
    *,
    cfg: DictConfig,
    obs_dim: int,
    prior_net: RSSMPriorV3,
    reward_mlp: nn.Module,
    continue_mlp: nn.Module,
):
    """MLP encoder + RSSMRolloutV3 + MLP decoder + reward head.

    ``prior_net`` and ``reward_mlp`` are the shared modules from
    :func:`build_shared_modules`. Returns a TensorDictSequential whose forward
    consumes a trajectory batch and writes every key DreamerV3ModelLoss expects.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    encoder_net = SymlogEncoder(_apply_jax_init(_mlp_trunk(obs_dim, cfg)))
    encoder = TensorDictModule(
        encoder_net,
        in_keys=[("next", "observation")],
        out_keys=[("next", "encoded_latents")],
    )

    rssm_prior = TensorDictModule(
        prior_net,
        in_keys=["state", "belief", "action"],
        out_keys=[
            ("next", "prior_logits"),
            ("next", "state"),
            ("next", "belief"),
        ],
    )

    # JAX rssm.hidden (64 under size1m) is the width of the *hidden* layers in
    # the prior/posterior heads and the GRU input projections; rssm.deter (512)
    # is only the recurrent state width. Passing rnn_hidden_dim here made every
    # such layer 8x wider than the reference.
    posterior_net = RSSMPosteriorV3(
        hidden_dim=cfg.networks.hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        obs_embed_dim=cfg.networks.obs_embed_dim,
        unimix=cfg.networks.unimix,
        norm=cfg.networks.jax_core,
        norm_eps=_NORM_EPS,
    )
    _apply_jax_init(posterior_net)
    rssm_posterior = TensorDictModule(
        posterior_net,
        in_keys=[("next", "belief"), ("next", "encoded_latents")],
        out_keys=[("next", "posterior_logits"), ("next", "state")],
    )

    rollout = RSSMRolloutV3(rssm_prior, rssm_posterior)

    # The decoder reconstructs from the *full* model state: JAX concatenates
    # ``[stoch, deter]`` before the decoder MLP (``rssm.py`` ``Decoder.__call__``).
    # Feeding only the stochastic latent forces the posterior to carry
    # everything the belief already holds, inflating the representation KL.
    decoder = TensorDictModule(
        _apply_jax_init(
            _norm_mlp(state_dim + cfg.networks.rnn_hidden_dim, obs_dim, cfg)
        ),
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reco_pixels")],
    )

    reward_head = TensorDictModule(
        reward_mlp,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reward")],
    )

    # Continue (termination) head — a binary predictor trained against 1 - done.
    continue_head = TensorDictModule(
        continue_mlp,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "continue_pred")],
    )

    world_model = TensorDictSequential(
        encoder, rollout, decoder, reward_head, continue_head
    )
    # Also hand back the raw encoder + posterior nets so the real-env acting
    # policy can reuse them (shared params) to turn each observation into a
    # latent (state, belief) instead of acting on the zero-primed defaults.
    return world_model, encoder_net, posterior_net


class BoundedNormalActor(nn.Module):
    """DreamerV3 ``bounded_normal`` policy head (heads.py:bounded_normal).

    Emits ``loc = tanh(mean)`` and ``scale = (maxstd - minstd) * sigmoid(raw + 2)
    + minstd`` for a plain (diagonal) Normal. Unlike ``TanhNormal``, this gives an
    analytic entropy and a log-prob that does not blow up near the action bounds.
    """

    def __init__(
        self, *, in_features: int, action_dim: int, cfg: DictConfig,
        minstd: float, maxstd: float,
    ):
        super().__init__()
        # JAX policy.outscale: 0.01 -- a near-zero output layer, so the policy
        # starts close to ``tanh(0) = 0`` mean with the sigmoid(+2)-biased std.
        self.net = _apply_jax_init(
            _norm_mlp(in_features, 2 * action_dim, cfg), outscale=0.01
        )
        self.minstd = minstd
        self.maxstd = maxstd

    def forward(self, state: torch.Tensor, belief: torch.Tensor):
        mean, raw_std = self.net(torch.cat([state, belief], dim=-1)).chunk(2, dim=-1)
        loc = torch.tanh(mean)
        scale = (self.maxstd - self.minstd) * torch.sigmoid(raw_std + 2.0) + self.minstd
        return loc, scale


class TanhNormalActor(nn.Module):
    """DreamerV2-style tanh-squashed Normal policy head (ablation baseline).

    Emits ``(loc, scale)`` for a :class:`~torchrl.modules.distributions.TanhNormal`
    -- the pre-DreamerV3 continuous policy. Selected by ``networks.actor_dist=tanh``
    to contrast against the DreamerV3 ``bounded_normal`` head (V3-off ablation).
    """

    def __init__(self, *, in_features: int, action_dim: int, cfg: DictConfig):
        super().__init__()
        self.net = _norm_mlp(in_features, 2 * action_dim, cfg)

    def forward(self, state: torch.Tensor, belief: torch.Tensor):
        loc, raw_std = self.net(torch.cat([state, belief], dim=-1)).chunk(2, dim=-1)
        scale = nn.functional.softplus(raw_std) + 0.1
        return loc, scale


def build_actor(*, cfg: DictConfig, action_dim: int):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    in_features = state_dim + cfg.networks.rnn_hidden_dim
    actor_dist = cfg.networks.get("actor_dist", "bounded")
    if actor_dist == "bounded":
        actor_net = BoundedNormalActor(
            in_features=in_features,
            action_dim=action_dim,
            cfg=cfg,
            minstd=cfg.networks.actor_minstd,
            maxstd=cfg.networks.actor_maxstd,
        )
        distribution_class = IndependentNormal
    elif actor_dist == "tanh":
        actor_net = TanhNormalActor(
            in_features=in_features, action_dim=action_dim, cfg=cfg
        )
        distribution_class = TanhNormal
    else:
        raise ValueError(
            f"unknown networks.actor_dist={actor_dist!r}, expected 'bounded' or 'tanh'"
        )
    actor_model = ProbabilisticTensorDictSequential(
        TensorDictModule(
            actor_net,
            in_keys=["state", "belief"],
            out_keys=["loc", "scale"],
        ),
        ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            default_interaction_type=InteractionType.RANDOM,
            distribution_class=distribution_class,
            return_log_prob=True,
            log_prob_key="action_log_prob",
        ),
    )
    with torch.no_grad():
        actor_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, cfg.networks.rnn_hidden_dim),
                },
                [1],
            )
        )
    return actor_model, actor_net


class InitialBelief(nn.Module):
    """Form the RSSM initial belief at episode reset, matching the JAX reference.

    JAX ``rssm.py:_observe`` masks ``(deter, stoch, prevact)`` to zero on the
    first step of an episode and runs ``_core(0, 0, 0)`` to obtain the initial
    deterministic belief *before* the first posterior. This module reproduces
    that: on ``is_init`` steps it replaces the zero-primed ``belief`` with
    ``prior(0, 0, 0)``; otherwise it passes the carried belief through unchanged.
    """

    def __init__(
        self,
        prior_net: RSSMPriorV3,
        state_dim: int,
        belief_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.prior_net = prior_net
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.action_dim = action_dim

    def forward(
        self, belief: torch.Tensor, is_init: torch.Tensor
    ) -> torch.Tensor:
        zeros_state = belief.new_zeros(*belief.shape[:-1], self.state_dim)
        zeros_belief = belief.new_zeros(*belief.shape[:-1], self.belief_dim)
        zeros_action = belief.new_zeros(*belief.shape[:-1], self.action_dim)
        _, _, init_belief = self.prior_net(zeros_state, zeros_belief, zeros_action)
        mask = is_init.bool()
        if mask.dim() < belief.dim():
            mask = mask.unsqueeze(-1)
        return torch.where(mask, init_belief, belief)


def build_actor_realworld(
    *,
    cfg: DictConfig,
    action_dim: int,
    encoder_net: nn.Module,
    posterior_net: nn.Module,
    prior_net: RSSMPriorV3,
    actor_model: ProbabilisticTensorDictSequential,
):
    """Recurrent acting policy for the *real* environment.

    ``actor_model`` alone maps ``(state, belief) -> action`` and is correct only
    in imagination, where the world model supplies the latents. In the real env
    the latents must be *inferred from observations*, so we mirror the DreamerV3
    reference acting path (``agent.py:policy`` -> ``rssm.py:_observe``): per step,

    0. ``InitialBelief`` : on episode reset, belief <- ``prior(0, 0, 0)``
    1. ``encoder``       : observation -> encoded_latents
    2. ``posterior``     : (belief, encoded_latents) -> state
    3. ``actor``         : (state, belief) -> action   (the shared ``actor_model``)
    4. ``prior``         : (state, belief, action) -> ("next", "belief")

    Step 4 advances the deterministic belief for the *next* step; the env carries
    ``("next", "belief")`` to the next root ``belief`` (registered by the
    ``TensorDictPrimer``), giving the RSSM recurrence. Step 0 (requires an
    ``InitTracker`` on the env) forms the reference initial belief so the first
    posterior of each episode conditions on ``_core(0, 0, 0)`` rather than
    ``belief = 0``. Every module is a shared instance of the trained world model
    / actor, so no new parameters are added.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    return TensorDictSequential(
        TensorDictModule(
            InitialBelief(
                prior_net, state_dim, cfg.networks.rnn_hidden_dim, action_dim
            ),
            in_keys=["belief", "is_init"],
            out_keys=["belief"],
        ),
        TensorDictModule(
            encoder_net, in_keys=["observation"], out_keys=["encoded_latents"]
        ),
        TensorDictModule(
            posterior_net,
            in_keys=["belief", "encoded_latents"],
            out_keys=["posterior_logits", "state"],
        ),
        actor_model,
        TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=["_prior_logits", "_prior_state", ("next", "belief")],
        ),
    )


def build_value(*, cfg: DictConfig):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    # Two-hot critic (V3) outputs logits over num_value_bins; the scalar ablation
    # arm (networks.value_head=scalar) outputs a single symlog value.
    twohot = cfg.networks.get("value_head", "twohot") == "twohot"
    out_features = cfg.networks.num_value_bins if twohot else 1
    # JAX value.outscale: 0.0 -- the critic predicts exactly 0 at init (with the
    # symmetric two-hot grid, a uniform distribution decodes to 0).
    value_model = TensorDictModule(
        _apply_jax_init(
            _norm_mlp(state_dim + cfg.networks.rnn_hidden_dim, out_features, cfg),
            outscale=0.0,
        ),
        in_keys=["state", "belief"],
        out_keys=["state_value"],
    )
    with torch.no_grad():
        value_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, cfg.networks.rnn_hidden_dim),
                },
                [1],
            )
        )
    return value_model


def build_mb_env(
    *,
    cfg: DictConfig,
    real_env,
    prior_net: RSSMPriorV3,
    reward_mlp: nn.Module,
    reward_bins: torch.Tensor,
    bin_space: str = "reward",
):
    """Imagination env: DreamerEnv wrapping the **shared** V3 prior + reward head.

    ``prior_net`` and ``reward_mlp`` are the same module instances the world
    model trains (see :func:`build_shared_modules`), so imagination rolls out
    with the learned dynamics/reward. The two-hot reward head is wrapped in
    :class:`TwoHotRewardDecoder` to emit the scalar reward the env expects.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    primer_env = TransformedEnv(
        real_env,
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )
    transition_model = SafeSequential(
        TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=["_", "state", "belief"],
        )
    )
    reward_model = TensorDictModule(
        TwoHotRewardDecoder(reward_mlp, reward_bins, bin_space=bin_space),
        in_keys=["state", "belief"],
        out_keys=["reward"],
    )
    mb_env = DreamerEnv(
        world_model=WorldModelWrapper(transition_model, reward_model),
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([cfg.networks.rnn_hidden_dim]),
    )
    mb_env.set_specs_from_env(primer_env)
    with torch.no_grad():
        mb_env.rollout(3)
    return mb_env


@torch.no_grad()
def eval_episode_reward(
    env, actor, num_episodes: int, max_steps: int = 200
) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(
                max_steps=max_steps, policy=actor, break_when_any_done=True
            )
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)
    device = torch.device(cfg.env.device)
    num_envs = cfg.collector.get("num_envs", 1)
    if num_envs <= 0:
        raise ValueError(f"collector.num_envs must be positive, got {num_envs}.")
    if cfg.collector.frames_per_batch % num_envs:
        raise ValueError(
            "collector.frames_per_batch must be divisible by collector.num_envs, "
            f"got {cfg.collector.frames_per_batch} and {num_envs}."
        )

    real_env = make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    # Shared RSSM prior + reward head: the world model trains these modules and
    # the imagination env rolls out with the *same* instances, so imagination
    # tracks the learned dynamics/reward instead of a frozen random network.
    prior_net, reward_mlp, continue_mlp = build_shared_modules(
        cfg=cfg, action_dim=action_dim
    )
    # Two-hot bin space. The reference's ``symexp_twohot`` head places the bins
    # at ``symexp(linspace(-20, 20, N))`` and does *both* the two-hot
    # interpolation and the decode in reward space, so the decoded prediction is
    # E[reward]. The paper's formulation (bin_space=symlog) interpolates in
    # symlog space and decodes ``symexp(E[symlog reward])``: the bin centers
    # coincide, but a spread-out distribution decodes differently.
    bin_space = cfg.networks.get("bin_space", "reward")
    reward_bins = _default_bins(
        cfg.networks.num_reward_bins, device=device, bin_space=bin_space
    )

    world_model, encoder_net, posterior_net = build_world_model(
        cfg=cfg,
        obs_dim=obs_dim,
        prior_net=prior_net,
        reward_mlp=reward_mlp,
        continue_mlp=continue_mlp,
    )
    if cfg.optimization.get("compile_rssm", False):
        # Fuse the unrolled RSSM recurrence. Worth minutes of one-off compile
        # for a long run (~3x on the rollout), not for a smoke test.
        world_model[1].compile_scan()

    # Continue head wrapped for the imagination discount in the actor loss.
    continue_model = TensorDictModule(
        continue_mlp, in_keys=["state", "belief"], out_keys=["continue_pred"]
    )
    actor_model, _actor_net = build_actor(cfg=cfg, action_dim=action_dim)
    # Real-env acting policy: encodes observations into latents each step
    # (imagination uses the bare ``actor_model`` head, which is fed latents by
    # the world model). Shares all params with the trained modules.
    actor_realworld = build_actor_realworld(
        cfg=cfg,
        action_dim=action_dim,
        encoder_net=encoder_net,
        posterior_net=posterior_net,
        prior_net=prior_net,
        actor_model=actor_model,
    )
    value_model = build_value(cfg=cfg)
    # Move every network onto the training device *before* the imagination env is
    # built: DreamerEnv takes its specs from the (already on-device) real env and
    # rolls out at construction time, so its shared modules must match.
    for _module in (world_model, actor_model, actor_realworld, value_model):
        _module.to(device)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        prior_net=prior_net,
        reward_mlp=reward_mlp,
        reward_bins=reward_bins,
        bin_space=bin_space,
    )

    # JAX loss_scales.repval = 0.3 (replay critic loss). Set to 0 to disable.
    repval_scale = cfg.optimization.get("repval_scale", 0.0)
    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_alpha=cfg.optimization.kl_alpha,
        unimix=cfg.networks.unimix,
        kl_dyn_scale=cfg.optimization.kl_dyn_scale,
        kl_rep_scale=cfg.optimization.kl_rep_scale,
        lambda_continue=cfg.optimization.lambda_continue,
        continue_target_scale=(
            1.0 - 1.0 / cfg.optimization.horizon
            if cfg.optimization.contdisc
            else 1.0
        ),
        # DreamerV3 sums the reconstruction loss over the observation dims and
        # averages over batch/time; averaging over the observation dims instead
        # would under-weight it against the KL by the observation width (24 for
        # walker), which is the balance the loss_scales are tuned around.
        global_average=False,
        bin_space=bin_space,
        # The replay critic loss backpropagates through the world-model
        # features (JAX ``repval_grad: True``), so it needs the live output.
        detach_output=repval_scale <= 0,
    )
    model_loss.set_keys(pixels="observation")
    # V3 feature toggles (defaults preserve full DreamerV3 behavior; the
    # config_dmc_v3off ablation arm flips these to isolate the V3 feature set).
    twohot = cfg.networks.get("value_head", "twohot") == "twohot"
    use_slow = cfg.optimization.get("slow_value", True)
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        normalize_returns=cfg.optimization.get("retnorm", True),  # retnorm on/off
        use_analytic_entropy=cfg.networks.get("actor_dist", "bounded") == "bounded",
        num_value_bins=cfg.networks.num_value_bins if twohot else None,  # None -> scalar value
        continue_model=continue_model,
        imag_loss=cfg.optimization.imag_loss,
        horizon=cfg.optimization.horizon,
        lam=cfg.optimization.lmbda,
        contdisc=cfg.optimization.contdisc,
        bin_space=bin_space,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    # Slow (EMA) target critic — a frozen copy of the value net that trails it
    # and regularises the value loss (DreamerV3 SlowModel, rate 0.02). Disabled
    # by optimization.slow_value=false (V3-off ablation).
    slow_value_model = None
    if use_slow:
        slow_value_model = copy.deepcopy(value_model)
        for p in slow_value_model.parameters():
            p.requires_grad_(False)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot" if twohot else "symlog_mse",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_value_model=slow_value_model,
        slowreg=1.0 if use_slow else 0.0,
        slow_rate=0.02,
        bin_space=bin_space,
    )
    # The losses own device-sensitive buffers of their own (two-hot bin grids,
    # the return-normalizer percentile EMAs), so they need moving too.
    for _loss in (model_loss, actor_loss, value_loss):
        _loss.to(device)

    # Single joint optimizer over all modules (DreamerV3 co-trains the world
    # model, actor and critic in one step). The raw module params are disjoint —
    # the imagination env only *shares* world-model params, it does not add new
    # ones — so chaining them double-counts nothing. The frozen slow critic is
    # excluded.
    all_params = (
        list(world_model.parameters())
        + list(actor_model.parameters())
        + list(value_model.parameters())
    )
    if cfg.optimization.optimizer == "dreamerv3":
        opt = DreamerV3Optimizer(
            all_params,
            lr=cfg.optimization.lr,
            agc=cfg.optimization.agc,
            warmup=cfg.optimization.opt_warmup,
        )
    else:
        opt = torch.optim.Adam(all_params, lr=cfg.optimization.lr)

    # DreamerV3 collects from ``run.envs`` environments concurrently (16 for
    # dmc_proprio): each driver tick steps all of them and pushes one
    # transition per worker, so replay holds that many decorrelated trajectory
    # streams instead of one. With ``frames_per_batch = num_envs`` the
    # train_ratio is unchanged. SerialEnv keeps the environments in one process
    # while still batching the policy call over all workers.

    def _make_explore_env(seed_offset: int = 2):
        return TransformedEnv(
            make_env(cfg, cfg.env.seed + seed_offset),
            Compose(
                TensorDictPrimer(
                    random=False,
                    default_value=0,
                    state=Unbounded(state_dim),
                    belief=Unbounded(cfg.networks.rnn_hidden_dim),
                ),
                # is_init lets the acting policy form the reference initial
                # belief (prior(0,0,0)) on the first step of each episode.
                InitTracker(),
            ),
        )

    if num_envs > 1:
        explore_env = SerialEnv(
            num_envs,
            [
                (lambda i=i: _make_explore_env(2 + i))
                for i in range(num_envs)
            ],
        )
    else:
        explore_env = _make_explore_env()

    collector = Collector(
        explore_env,
        actor_realworld,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.env.device,
        exploration_type=ExplorationType.RANDOM
        if cfg.collector.exploration == "random"
        else ExplorationType.MODE,
    )

    # With several environments the collector's batch is [environment, time].
    # Extending along dim 1 transposes it to [time, environment] before writing,
    # so later batches append in time while each storage column remains one
    # contiguous environment stream for SliceSampler.
    rb = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.replay_buffer.buffer_size,
            ndim=2 if num_envs > 1 else 1,
            # Keep replay off the accelerator. The storage is lazy but
            # preallocates its full capacity on the first extend, and with
            # ``state``/``belief`` primed into every transition a 1e6-step
            # buffer is tens of GB -- 28 GB of an A6000 measured here. The
            # sampling loop moves each batch to the device anyway, and the
            # reference likewise keeps replay in host memory.
            device="cpu",
        ),
        dim_extend=1 if num_envs > 1 else 0,
        sampler=SliceSampler(
            slice_len=cfg.replay_buffer.seq_len,
            traj_key=("collector", "traj_ids"),
            # Trajectory boundaries are rescanned over the *whole* storage on
            # every sample() otherwise, which is O(buffer_size) and comes to
            # dominate the step time as the buffer fills. extend() erases the
            # cache, so the boundaries stay exact: this loop extends once and
            # then samples updates_per_batch times.
            cache_values=True,
        ),
        batch_size=cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len,
    )

    env_step = 0
    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_hist: dict[str, list[torch.Tensor]] = {
        "kl": [],
        "dyn": [],
        "rep": [],
        "con": [],
        "reco": [],
        "reward": [],
        "actor": [],
        "value": [],
        "repval": [],
    }
    next_eval = 0

    eval_env = TransformedEnv(
        make_env(cfg, cfg.env.seed + 100),
        Compose(
            TensorDictPrimer(
                random=False,
                default_value=0,
                state=Unbounded(state_dim),
                belief=Unbounded(cfg.networks.rnn_hidden_dim),
            ),
            InitTracker(),
        ),
    )

    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )

    for data in collector:
        rb.extend(data if num_envs > 1 else data.reshape(-1))
        env_step += data.numel()

        if len(rb) < warmup:
            continue

        for _ in range(cfg.optimization.updates_per_batch):
            sample = (
                rb.sample()
                .reshape(cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len)
                .to(device)
            )

            sample.set(
                "state",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    state_dim,
                    device=device,
                ),
            )
            sample.set(
                "belief",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    cfg.networks.rnn_hidden_dim,
                    device=device,
                ),
            )

            m_td, model_out = model_loss(sample)
            total_m = (
                m_td["loss_model_kl"]
                + m_td["loss_model_reco"]
                + m_td["loss_model_reward"]
                + m_td.get(
                    "loss_model_continue",
                    torch.zeros_like(m_td["loss_model_kl"]),
                )
            ).squeeze()

            # Imagine from posterior states (detached from the world model, so the
            # actor/critic gradients do not flow into it — DreamerV3 sg's the
            # imagined features).
            post_state = (
                model_out.get(("next", "state")).detach().reshape(-1, state_dim)
            )
            post_belief = (
                model_out.get(("next", "belief"))
                .detach()
                .reshape(-1, cfg.networks.rnn_hidden_dim)
            )
            actor_input = TensorDict(
                {"state": post_state, "belief": post_belief},
                [post_state.shape[0]],
            )
            a_td, fake_data = actor_loss(actor_input)
            v_td, _ = value_loss(fake_data.detach())

            # Replay critic loss (JAX ``repl_loss``, loss_scales.repval 0.3):
            # DreamerV3 also trains the critic along the *real* replay
            # sequences, with the imagination returns as the per-step bootstrap.
            # ``repval_grad: True`` -> the gradient is not stopped at the
            # world-model features, hence the un-detached ``model_out``.
            repval = None
            if repval_scale > 0:
                boot = (
                    fake_data.get("lambda_target")[..., 0, 0]
                    .reshape(
                        cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len
                    )
                    .detach()
                )
                replay_feat = TensorDict(
                    {
                        "state": model_out.get(("next", "state")),
                        "belief": model_out.get(("next", "belief")),
                    },
                    batch_size=[
                        cfg.replay_buffer.batch_size,
                        cfg.replay_buffer.seq_len,
                    ],
                )
                repval = value_loss.replay_value_loss(
                    replay_feat,
                    next_reward=model_out.get(("next", "true_reward")).squeeze(-1),
                    next_done=model_out.get(("next", "done")).squeeze(-1),
                    next_terminated=model_out.get(("next", "terminated")).squeeze(-1),
                    bootstrap=boot,
                    horizon=cfg.optimization.horizon,
                    lam=cfg.optimization.lmbda,
                )

            # One joint backward + step over WM + actor + critic. The three
            # sub-losses touch disjoint parameters (imagination holds out the WM
            # and critic), so a single optimizer trains each correctly.
            total = total_m + a_td["loss_actor"] + v_td["loss_value"]
            if repval is not None:
                total = total + repval_scale * repval
            opt.zero_grad(set_to_none=True)
            total.backward()
            # DreamerV3Optimizer applies AGC per parameter; this global-norm clip
            # is a loose safety net (mainly for the Adam fallback).
            torch.nn.utils.clip_grad_norm_(all_params, cfg.optimization.grad_clip)
            opt.step()
            value_loss.update_slow_value()

            loss_hist["kl"].append(m_td["loss_model_kl"].detach())
            loss_hist["dyn"].append(m_td["kl_dyn"].detach())
            loss_hist["rep"].append(m_td["kl_rep"].detach())
            loss_hist["con"].append(
                m_td.get(
                    "loss_model_continue", torch.zeros_like(m_td["loss_model_kl"])
                ).detach()
            )
            loss_hist["reco"].append(m_td["loss_model_reco"].detach())
            loss_hist["reward"].append(m_td["loss_model_reward"].detach())
            loss_hist["actor"].append(a_td["loss_actor"].detach())
            loss_hist["value"].append(v_td["loss_value"].detach())
            loss_hist["repval"].append(
                repval.detach()
                if repval is not None
                else torch.zeros((), device=device)
            )

        if env_step >= next_eval:
            r = eval_episode_reward(
                eval_env,
                actor_realworld,
                cfg.logger.eval_episodes,
                max_steps=cfg.logger.get("eval_max_steps", 200),
            )
            history_steps.append(env_step)
            history_eval.append(r)
            # Term names mirror DreamerV3's ``loss/*`` metrics (agent.py) so a run
            # can be diffed against the reference's metrics.jsonl at matched steps.
            torchrl_logger.info(
                "[env_step=%5d] eval_reward=%+.2f kl=%.3f dyn=%.3f rep=%.3f "
                "con=%.4f reco=%.3f reward=%.3f policy=%.3f value=%.3f "
                "repval=%.3f",
                env_step,
                r.item(),
                loss_hist["kl"][-1].item(),
                loss_hist["dyn"][-1].item(),
                loss_hist["rep"][-1].item(),
                loss_hist["con"][-1].item(),
                loss_hist["reco"][-1].item(),
                loss_hist["reward"][-1].item(),
                loss_hist["actor"][-1].item(),
                loss_hist["value"][-1].item(),
                loss_hist["repval"][-1].item(),
            )
            next_eval = env_step + cfg.logger.eval_every

    if cfg.logger.output_plot and _has_matplotlib:
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dep)

        eval_steps = history_steps
        eval_rewards = torch.stack(history_eval).cpu().numpy() if history_eval else []
        kl_vals = torch.stack(loss_hist["kl"]).cpu().numpy() if loss_hist["kl"] else []
        reco_vals = (
            torch.stack(loss_hist["reco"]).cpu().numpy() if loss_hist["reco"] else []
        )
        reward_vals = (
            torch.stack(loss_hist["reward"]).cpu().numpy()
            if loss_hist["reward"]
            else []
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(eval_steps, eval_rewards, marker="o")
        axes[0].set_title(f"{cfg.env.name} eval reward (real env)")
        axes[0].set_xlabel("env_step")
        axes[0].set_ylabel("avg episode return")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(reco_vals, label="reco", alpha=0.8)
        axes[1].plot(reward_vals, label="reward", alpha=0.8)
        axes[1].plot(kl_vals, label="kl", alpha=0.8)
        axes[1].set_title("World-model losses (update step)")
        axes[1].set_xlabel("update step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps, "
            f"{cfg.optimization.updates_per_batch} updates/batch"
        )
        fig.tight_layout()
        fig.savefig(cfg.logger.output_plot, dpi=120)
        torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
    elif cfg.logger.output_plot:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )


if __name__ == "__main__":
    main()
