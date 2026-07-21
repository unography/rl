# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 on Pendulum-v1 — minimal end-to-end training script.

State-based (not pixel-based) to keep the script compact — the 3-D obs is
treated as a flat feature vector with ``global_average=True`` in the model
loss. The wiring is still real:

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
    Compose,
    InitTracker,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeSequential, WorldModelWrapper
from torchrl.modules.distributions import IndependentNormal
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
    symexp,
    two_hot_decode,
)
from torchrl.objectives.utils import ValueEstimators

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


def make_env(env_name: str, seed: int = 0):
    env = GymEnv(env_name, device="cpu")
    env = TransformedEnv(env, StepCounter())
    # Normalize the action space to [-1, 1] (DreamerV3 wraps envs this way): the
    # policy emits [-1, 1] and it is rescaled to the env's native torque range,
    # so the agent gets full control authority and the RSSM action soft-clip is a
    # no-op. Actions stored in the buffer are the normalized [-1, 1] ones.
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

    The world-model reward head emits logits over ``num_reward_bins`` in symlog
    space (two-hot targets). The imagination :class:`DreamerEnv` needs a scalar
    reward, so we take the distribution's expectation and invert ``symlog``.
    Wrapping the *same* ``reward_mlp`` module keeps the imagined reward in
    lock-step with the trained world model.
    """

    def __init__(self, reward_mlp: nn.Module, reward_bins: torch.Tensor):
        super().__init__()
        self.reward_mlp = reward_mlp
        self.register_buffer("reward_bins", reward_bins)

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        logits = self.reward_mlp(torch.cat([state, belief], dim=-1))
        symlog_reward = two_hot_decode(logits, self.reward_bins)
        return symexp(symlog_reward).unsqueeze(-1)


def _norm_mlp(in_features: int, out_features: int, cfg: DictConfig):
    """MLP with SiLU activation + RMSNorm after each hidden layer (JAX-faithful).

    DreamerV3 uses ``act: silu, norm: rms`` throughout (configs.yaml); TorchRL's
    bare ``MLP`` defaults to Tanh and no norm.
    """
    return MLP(
        in_features=in_features,
        out_features=out_features,
        depth=cfg.networks.depth,
        num_cells=cfg.networks.hidden_dim,
        activation_class=nn.SiLU,
        norm_class=nn.RMSNorm,
        norm_kwargs={"normalized_shape": cfg.networks.hidden_dim},
    )


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
        hidden_dim=cfg.networks.rnn_hidden_dim,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        action_dim=action_dim,
        unimix=cfg.networks.unimix,
        jax_core=cfg.networks.jax_core,
        blocks=cfg.networks.blocks,
        norm=cfg.networks.jax_core,
    )
    reward_mlp = _norm_mlp(
        state_dim + cfg.networks.rnn_hidden_dim, cfg.networks.num_reward_bins, cfg
    )
    # Continue (termination) head, shared by the world model (BCE against 1-done)
    # and the imagination discount in the actor loss.
    continue_mlp = _norm_mlp(state_dim + cfg.networks.rnn_hidden_dim, 1, cfg)
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

    encoder_net = _norm_mlp(obs_dim, cfg.networks.obs_embed_dim, cfg)
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

    posterior_net = RSSMPosteriorV3(
        hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        obs_embed_dim=cfg.networks.obs_embed_dim,
        unimix=cfg.networks.unimix,
        norm=cfg.networks.jax_core,
    )
    rssm_posterior = TensorDictModule(
        posterior_net,
        in_keys=[("next", "belief"), ("next", "encoded_latents")],
        out_keys=[("next", "posterior_logits"), ("next", "state")],
    )

    rollout = RSSMRolloutV3(rssm_prior, rssm_posterior)

    decoder = TensorDictModule(
        _norm_mlp(state_dim, obs_dim, cfg),
        in_keys=[("next", "state")],
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
        self.net = _norm_mlp(in_features, 2 * action_dim, cfg)
        self.minstd = minstd
        self.maxstd = maxstd

    def forward(self, state: torch.Tensor, belief: torch.Tensor):
        mean, raw_std = self.net(torch.cat([state, belief], dim=-1)).chunk(2, dim=-1)
        loc = torch.tanh(mean)
        scale = (self.maxstd - self.minstd) * torch.sigmoid(raw_std + 2.0) + self.minstd
        return loc, scale


def build_actor(*, cfg: DictConfig, action_dim: int):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    actor_net = BoundedNormalActor(
        in_features=state_dim + cfg.networks.rnn_hidden_dim,
        action_dim=action_dim,
        cfg=cfg,
        minstd=cfg.networks.actor_minstd,
        maxstd=cfg.networks.actor_maxstd,
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
            distribution_class=IndependentNormal,
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
    value_model = TensorDictModule(
        _norm_mlp(
            state_dim + cfg.networks.rnn_hidden_dim, cfg.networks.num_value_bins, cfg
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
        TwoHotRewardDecoder(reward_mlp, reward_bins),
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
def eval_episode_reward(env, actor, num_episodes: int) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(max_steps=200, policy=actor, break_when_any_done=True)
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)

    real_env = make_env(cfg.env.name, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    # Shared RSSM prior + reward head: the world model trains these modules and
    # the imagination env rolls out with the *same* instances, so imagination
    # tracks the learned dynamics/reward instead of a frozen random network.
    prior_net, reward_mlp, continue_mlp = build_shared_modules(
        cfg=cfg, action_dim=action_dim
    )
    reward_bins = torch.linspace(-20.0, 20.0, cfg.networks.num_reward_bins)

    world_model, encoder_net, posterior_net = build_world_model(
        cfg=cfg,
        obs_dim=obs_dim,
        prior_net=prior_net,
        reward_mlp=reward_mlp,
        continue_mlp=continue_mlp,
    )
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
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg.env.name, cfg.env.seed + 1),
        prior_net=prior_net,
        reward_mlp=reward_mlp,
        reward_bins=reward_bins,
    )

    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_alpha=cfg.optimization.kl_alpha,
        unimix=cfg.networks.unimix,
        kl_dyn_scale=cfg.optimization.kl_dyn_scale,
        kl_rep_scale=cfg.optimization.kl_rep_scale,
        lambda_continue=cfg.optimization.lambda_continue,
        global_average=True,  # state-based obs, not (C, H, W) pixels
    )
    model_loss.set_keys(pixels="observation")
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        normalize_returns=True,
        use_analytic_entropy=True,
        num_value_bins=cfg.networks.num_value_bins,
        continue_model=continue_model,
        imag_loss=cfg.optimization.imag_loss,
        horizon=cfg.optimization.horizon,
        lam=cfg.optimization.lmbda,
        contdisc=cfg.optimization.contdisc,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    # Slow (EMA) target critic — a frozen copy of the value net that trails it
    # and regularises the value loss (DreamerV3 SlowModel, rate 0.02).
    slow_value_model = copy.deepcopy(value_model)
    for p in slow_value_model.parameters():
        p.requires_grad_(False)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_value_model=slow_value_model,
        slowreg=1.0,
        slow_rate=0.02,
    )

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

    explore_env = TransformedEnv(
        make_env(cfg.env.name, cfg.env.seed + 2),
        Compose(
            TensorDictPrimer(
                random=False,
                default_value=0,
                state=Unbounded(state_dim),
                belief=Unbounded(cfg.networks.rnn_hidden_dim),
            ),
            # is_init lets the acting policy form the reference initial belief
            # (prior(0,0,0)) on the first step of each episode.
            InitTracker(),
        ),
    )

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

    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.replay_buffer.buffer_size),
        sampler=SliceSampler(
            slice_len=cfg.replay_buffer.seq_len,
            traj_key=("collector", "traj_ids"),
        ),
        batch_size=cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len,
    )

    env_step = 0
    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_hist: dict[str, list[torch.Tensor]] = {
        "kl": [],
        "reco": [],
        "reward": [],
        "actor": [],
        "value": [],
    }
    next_eval = 0

    eval_env = TransformedEnv(
        make_env(cfg.env.name, cfg.env.seed + 100),
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
        rb.extend(data.reshape(-1))
        env_step += data.numel()

        if len(rb) < warmup:
            continue

        for _ in range(cfg.optimization.updates_per_batch):
            sample = rb.sample().reshape(
                cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len
            )

            sample.set(
                "state",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    state_dim,
                ),
            )
            sample.set(
                "belief",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    cfg.networks.rnn_hidden_dim,
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

            # One joint backward + step over WM + actor + critic. The three
            # sub-losses touch disjoint parameters (imagination holds out the WM
            # and critic), so a single optimizer trains each correctly.
            total = total_m + a_td["loss_actor"] + v_td["loss_value"]
            opt.zero_grad(set_to_none=True)
            total.backward()
            # DreamerV3Optimizer applies AGC per parameter; this global-norm clip
            # is a loose safety net (mainly for the Adam fallback).
            torch.nn.utils.clip_grad_norm_(all_params, cfg.optimization.grad_clip)
            opt.step()
            value_loss.update_slow_value()

            loss_hist["kl"].append(m_td["loss_model_kl"].detach())
            loss_hist["reco"].append(m_td["loss_model_reco"].detach())
            loss_hist["reward"].append(m_td["loss_model_reward"].detach())
            loss_hist["actor"].append(a_td["loss_actor"].detach())
            loss_hist["value"].append(v_td["loss_value"].detach())

        if env_step >= next_eval:
            r = eval_episode_reward(eval_env, actor_realworld, cfg.logger.eval_episodes)
            history_steps.append(env_step)
            history_eval.append(r)
            torchrl_logger.info(
                "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
                env_step,
                r.item(),
                loss_hist["kl"][-1].item(),
                loss_hist["reco"][-1].item(),
                loss_hist["reward"][-1].item(),
                loss_hist["actor"][-1].item(),
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
