# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 training script, reproducing the reference JAX implementation.

Proprioceptive (not pixel-based): observations are flat feature vectors whose
reconstruction loss sums the event dimension before averaging batch and time.
Metrics stream to JSONL on the same step axis the reference logs, and
``logger.diagnostics`` adds its ``train/*`` scalars for a term-by-term
comparison.

Usage::

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        collector.total_frames=5000 logger.eval_every=500

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        --config-name=config_dmc_walker
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import NamedTuple

import hydra
import torch

from dreamer_v3_agent import (
    build_actor,
    build_continuation_model,
    build_imagination_model,
    build_mb_env,
    build_real_world_actor,
    build_value,
    build_world_model,
    DreamerV3BehaviorPolicySync,
    DreamerV3Optimizer,
    DreamerV3SeededPolicy,
    make_env,
    make_primed_env,
)
from dreamer_v3_replay import (
    collector_action_budget,
    DreamerV3ReplayPipeline,
    DreamerV3ReplayRecordBuilder,
    DreamerV3ReplaySampler,
    DreamerV3ShiftedRecordExtender,
    DreamerV3UpdateRatio,
    driver_step_for_action,
)
from dreamer_v3_utils import (
    append_jsonl,
    eval_episode_reward,
    jax_torch_seed,
    latent_state_dim,
    LEARNER_RNG_STREAM,
    plot_enabled,
    reference_diagnostics,
    REPLAY_RNG_STREAM,
    save_run_plot,
    training_episode_returns,
)
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from torchrl import timeit
from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RoundRobinWriter
from torchrl.envs import SerialEnv
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators


class _Learner(NamedTuple):
    """The trained modules, their losses, and the shared optimizer."""

    world_model: TensorDictModuleBase
    model_loss: DreamerV3ModelLoss
    actor_loss: DreamerV3ActorLoss
    value_loss: DreamerV3ValueLoss
    value_target_updater: SoftUpdate
    optimizer: DreamerV3Optimizer
    real_world_actor: TensorDictModuleBase


def _validated_action_budget(cfg: DictConfig) -> int:
    """Return the control-transition budget for the collector."""
    num_envs = cfg.collector.num_envs
    if num_envs <= 0:
        raise ValueError(f"collector.num_envs must be positive, got {num_envs}.")
    if cfg.collector.frames_per_batch % num_envs:
        raise ValueError(
            "collector.frames_per_batch must be divisible by collector.num_envs, "
            f"got {cfg.collector.frames_per_batch} and {num_envs}."
        )
    collector_action_frames = (
        collector_action_budget(
            cfg.collector.total_frames,
            num_envs,
            cfg.env.max_episode_steps,
        )
        if cfg.collector.count_reset_records
        else cfg.collector.total_frames
    )
    if collector_action_frames % cfg.collector.frames_per_batch:
        raise ValueError(
            "The action budget derived from collector.total_frames must be "
            "divisible by collector.frames_per_batch, got "
            f"{collector_action_frames} and {cfg.collector.frames_per_batch}."
        )
    return collector_action_frames


def _build_learner(
    cfg: DictConfig, device: torch.device, obs_dim: int, action_dim: int
) -> _Learner:
    """Build the world model, actor, critic, losses and optimizer."""
    (
        world_model,
        prior_net,
        reward_net,
        reward_decoder,
        continuation_net,
    ) = build_world_model(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim)
    world_model = world_model.to(device)
    imagination_model = build_imagination_model(
        prior_net=prior_net,
        reward_net=reward_net,
        reward_decoder=reward_decoder,
        # Only under "scan", which already accepts a different random stream.
        compile_prior=cfg.optimization.compile_rssm == "scan",
    ).to(device)
    continuation_model = build_continuation_model(continuation_net=continuation_net).to(
        device
    )
    actor_model = build_actor(cfg=cfg, action_dim=action_dim).to(device)
    value_model = build_value(cfg=cfg).to(device)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        imagination_model=imagination_model,
        device=device,
    )

    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_mode="separate",
        lambda_dynamic=cfg.optimization.dynamic_loss_weight,
        lambda_representation=cfg.optimization.representation_loss_weight,
        unimix=cfg.networks.unimix,
        lambda_continue=1.0,
        continue_target_scale=1 - 1 / cfg.optimization.continuation_horizon,
        # DreamerV3 sums observation event dimensions and averages batch/time.
        global_average=False,
        detach_output=False,
    ).to(device)
    model_loss.set_keys(pixels="observation")
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        continuation_model=continuation_model,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        return_normalization_rate=cfg.optimization.return_normalization_rate,
        return_normalization_min_scale=cfg.optimization.return_normalization_min_scale,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    actor_loss.to(device)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_critic_regularization=cfg.optimization.slow_critic_regularization,
    ).to(device)
    value_target_updater = SoftUpdate(value_loss, tau=cfg.optimization.slow_critic_tau)

    trainable_parameters = (
        list(world_model.parameters())
        + list(actor_model.parameters())
        + list(value_loss.parameters())
    )
    optimizer = DreamerV3Optimizer(
        trainable_parameters,
        lr=cfg.optimization.lr,
        agc=cfg.optimization.adaptive_grad_clip,
        eps=cfg.optimization.adam_eps,
        warmup_steps=cfg.optimization.warmup_steps,
    )

    real_world_actor = build_real_world_actor(
        world_model=world_model,
        actor_model=actor_model,
        mixed_precision=cfg.optimization.mixed_precision,
    )
    return _Learner(
        world_model=world_model,
        model_loss=model_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        value_target_updater=value_target_updater,
        optimizer=optimizer,
        real_world_actor=real_world_actor,
    )


def _build_collection(
    cfg: DictConfig,
    device: torch.device,
    learner: _Learner,
    state_dim: int,
    action_dim: int,
    collector_action_frames: int,
) -> tuple[Collector, DreamerV3BehaviorPolicySync | None]:
    """Build the exploration environments, the acting policy and the collector."""
    num_envs = cfg.collector.num_envs
    real_world_actor = learner.real_world_actor
    if cfg.optimization.jax_behavior_policy_sync:
        collector_actor = copy.deepcopy(real_world_actor)
        # JAX's policy-key regex also shadows the decoder. Its carry is empty
        # for proprioception and it is action-dead, but include it for an exact
        # parameter-tree handoff rather than only action-equivalent behavior.
        behavior_decoder = copy.deepcopy(learner.world_model[2])
        learner_policy_tree = torch.nn.ModuleList(
            [real_world_actor, learner.world_model[2]]
        )
        behavior_policy_tree = torch.nn.ModuleList([collector_actor, behavior_decoder])
        behavior_policy_sync = DreamerV3BehaviorPolicySync(
            learner_policy_tree, behavior_policy_tree
        )
    else:
        collector_actor = real_world_actor
        behavior_policy_sync = None
    collector_policy = (
        DreamerV3SeededPolicy(collector_actor, cfg.env.seed)
        if cfg.optimization.separate_policy_rng
        else collector_actor
    )

    def make_explore_env(index: int):
        seed = cfg.env.seed + 2 + index if cfg.env.use_seed else None
        return make_primed_env(cfg, seed, state_dim, action_dim)

    if num_envs == 1:
        explore_env = make_explore_env(0)
    else:
        explore_env = SerialEnv(
            num_envs,
            [
                (lambda index=index: make_explore_env(index))
                for index in range(num_envs)
            ],
        )

    collector = Collector(
        explore_env,
        collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=collector_action_frames,
        policy_device=device,
        env_device="cpu",
        storing_device="cpu",
        exploration_type=ExplorationType.RANDOM
        if cfg.collector.exploration == "random"
        else ExplorationType.MODE,
    )
    if cfg.optimization.separate_policy_rng:
        # Collector construction probes policy output keys once. JAX initializes
        # those keys without consuming a real-action seed, so restart the
        # isolated policy stream before the first environment transition.
        collector_policy.reset_counter()
    return collector, behavior_policy_sync


def _build_replay(
    cfg: DictConfig, num_envs: int, replay_device: torch.device
) -> tuple[
    ReplayBuffer,
    DreamerV3ReplaySampler,
    DreamerV3ReplayRecordBuilder,
    DreamerV3ShiftedRecordExtender | None,
    DreamerV3ReplayPipeline,
]:
    """Build the replay buffer and its record stream, writeback and sampling."""
    replay_sampler = DreamerV3ReplaySampler(
        # One extra record is the destination slot for the final refreshed
        # posterior. The stream id remains constant across episode resets.
        slice_len=cfg.replay_buffer.seq_len + 1,
        online=cfg.replay_buffer.online,
    )
    rb = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.replay_buffer.buffer_size,
            ndim=2 if num_envs > 1 else 1,
            device=replay_device,
        ),
        dim_extend=1 if num_envs > 1 else 0,
        writer=RoundRobinWriter(track_generations=True),
        sampler=replay_sampler,
        batch_size=cfg.replay_buffer.batch_size * (cfg.replay_buffer.seq_len + 1),
        generator=torch.Generator().manual_seed(
            jax_torch_seed(cfg.env.seed, 0, REPLAY_RNG_STREAM)
        ),
    )
    replay_record_builder = DreamerV3ReplayRecordBuilder(num_envs)
    shifted_record_extender = (
        DreamerV3ShiftedRecordExtender(num_envs)
        if cfg.collector.count_reset_records
        else None
    )
    replay_pipeline = DreamerV3ReplayPipeline()
    return (
        rb,
        replay_sampler,
        replay_record_builder,
        shifted_record_extender,
        replay_pipeline,
    )


def _log_train_episodes(
    cfg: DictConfig,
    metrics_jsonl_path: Path | None,
    run_timer,
    completed_episodes: list[tuple[int, int, float]],
    batch_start_action_step: int,
) -> None:
    """Append one train_episode record per episode completed in this batch."""
    num_envs = cfg.collector.num_envs
    for time_index, env_index, score in completed_episodes:
        if cfg.collector.count_reset_records:
            action_index = batch_start_action_step // num_envs + time_index + 1
            episode_step = driver_step_for_action(
                action_index,
                env_index,
                num_envs,
                cfg.env.max_episode_steps,
            )
        else:
            episode_step = (
                batch_start_action_step + time_index * num_envs + env_index + 1
            )
        append_jsonl(
            metrics_jsonl_path,
            {
                "type": "train_episode",
                "environment_steps": episode_step,
                "action_steps": batch_start_action_step + (time_index + 1) * num_envs,
                "score": score,
                "elapsed_seconds": run_timer.elapsed(),
            },
        )


def _log_train_window(
    *,
    cfg: DictConfig,
    metrics_jsonl_path: Path | None,
    run_timer,
    learner: _Learner,
    sample: TensorDictBase,
    state_dim: int,
    use_bfloat16: bool,
    device: torch.device,
    record_step: int,
    action_step: int,
    update_step: int,
    loss_window_sum: torch.Tensor,
    loss_window_updates: int,
    last_log_seconds: float,
    last_log_updates: int,
) -> tuple[float, int]:
    """Append one train record; return the new last-log clock and update count."""
    # Read the clock before the optional diagnostics pass: it runs a
    # full world-model and imagination forward, and charging that to the
    # interval would deflate the very rate it is logged next to.
    elapsed_seconds = run_timer.elapsed()
    diagnostics = (
        reference_diagnostics(
            model_loss=learner.model_loss,
            actor_loss=learner.actor_loss,
            value_loss=learner.value_loss,
            sample=sample,
            state_dim=state_dim,
            rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
            use_bfloat16=use_bfloat16,
            device=device,
        )
        if cfg.logger.diagnostics
        else {}
    )
    batch_elements = cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len
    # ``training_fps`` mirrors the reference's ``fps/train``: batch
    # elements over wall-clock seconds since the previous log, so the
    # two runs' numbers can be compared directly.
    log_seconds = elapsed_seconds - last_log_seconds
    training_fps = (
        (update_step - last_log_updates) * batch_elements / log_seconds
        if log_seconds > 0
        else 0.0
    )
    append_jsonl(
        metrics_jsonl_path,
        {
            "type": "train",
            "environment_steps": record_step,
            "action_steps": action_step,
            "updates": update_step,
            # Averaged over every update since the previous log, as the
            # reference reports.
            "updates_in_window": loss_window_updates,
            **dict(
                zip(
                    (
                        "loss_dynamic_representation",
                        "loss_reconstruction",
                        "loss_reward",
                        "loss_actor",
                        "loss_value",
                        "loss_replay_value",
                    ),
                    (loss_window_sum / max(loss_window_updates, 1)).cpu().tolist(),
                )
            ),
            "training_fps": training_fps,
            "elapsed_seconds": elapsed_seconds,
            "bfloat16": use_bfloat16,
            **diagnostics,
        },
    )
    return run_timer.elapsed(), update_step


def _evaluate(
    *,
    cfg: DictConfig,
    device: torch.device,
    eval_env,
    real_world_actor: TensorDictModuleBase,
    metrics_jsonl_path: Path | None,
    run_timer,
    record_step: int,
    action_step: int,
    latest_losses: torch.Tensor,
) -> torch.Tensor:
    """Run the deterministic evaluation episodes and log their mean return."""
    # The RSSM samples its categorical latent on every step, including
    # under a deterministic policy. Fork the stream so the evaluation
    # cadence does not shift the training trajectory.
    with timeit("dreamer_v3/evaluation"), torch.random.fork_rng(
        devices=[device] if device.type == "cuda" else []
    ):
        r = eval_episode_reward(
            eval_env,
            real_world_actor,
            cfg.logger.eval_episodes,
            cfg.env.max_episode_steps,
        )
    torchrl_logger.info(
        "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
        record_step,
        r.item(),
        latest_losses[0].item(),
        latest_losses[1].item(),
        latest_losses[2].item(),
        latest_losses[3].item(),
    )
    append_jsonl(
        metrics_jsonl_path,
        {
            "type": "evaluation",
            "environment_steps": record_step,
            "action_steps": action_step,
            "return": r.item(),
            "episodes": cfg.logger.eval_episodes,
            "elapsed_seconds": run_timer.elapsed(),
        },
    )
    return r


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)

    device = (
        torch.device(cfg.optimization.device)
        if cfg.optimization.device
        else get_available_device()
    )
    replay_device = (
        torch.device(cfg.replay_buffer.device) if cfg.replay_buffer.device else device
    )
    num_envs = cfg.collector.num_envs
    count_reset_records = cfg.collector.count_reset_records
    collector_action_frames = _validated_action_budget(cfg)
    real_env = make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = latent_state_dim(cfg)
    metrics_jsonl_path = (
        Path(cfg.logger.metrics_jsonl).resolve() if cfg.logger.metrics_jsonl else None
    )
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("")
    timeit.reset()
    run_timer = timeit("dreamer_v3/run").start()

    learner = _build_learner(cfg, device, obs_dim, action_dim)
    model_loss = learner.model_loss
    actor_loss = learner.actor_loss
    value_loss = learner.value_loss
    optimizer = learner.optimizer
    value_target_updater = learner.value_target_updater

    collector, behavior_policy_sync = _build_collection(
        cfg, device, learner, state_dim, action_dim, collector_action_frames
    )
    (
        rb,
        replay_sampler,
        replay_record_builder,
        shifted_record_extender,
        replay_pipeline,
    ) = _build_replay(cfg, num_envs, replay_device)

    action_step = 0
    # JAX's driver emits one initial reset observation from every worker before
    # the first control transition. It counts those records on the curve axis.
    record_step = num_envs if count_reset_records else 0
    update_step = 0
    running_training_return = torch.zeros(num_envs)
    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_history: list[torch.Tensor] = []
    loss_window_sum = torch.zeros(6, device=device)
    loss_window_updates = 0
    record_loss_history = plot_enabled(cfg)
    next_eval = 0
    next_train_log = 0
    last_log_seconds = run_timer.elapsed()
    last_log_updates = 0

    eval_env = make_primed_env(cfg, cfg.env.seed + 100, state_dim, action_dim)

    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )
    warmup = max(warmup, num_envs * (cfg.replay_buffer.seq_len + 1))

    updates_per_batch = cfg.optimization.updates_per_batch
    update_ratio = (
        DreamerV3UpdateRatio(
            cfg.optimization.train_ratio
            / (cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len)
        )
        if cfg.optimization.train_ratio is not None
        else None
    )
    use_bfloat16 = cfg.optimization.mixed_precision and device.type == "cuda"

    def train_step(
        sample: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            model_loss_td, model_out = model_loss(sample)
            model_kl = (
                model_loss_td["loss_model_dynamic"]
                + model_loss_td["loss_model_representation"]
            )
            total_model_loss = (
                model_kl
                + model_loss_td["loss_model_reco"]
                + model_loss_td["loss_model_reward"]
                + model_loss_td["loss_model_continue"]
            ).squeeze()

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
            actor_loss_td, fake_data = actor_loss(actor_input)
            value_loss_td, _ = value_loss(fake_data.detach())

            replay_features = TensorDict(
                {
                    "state": model_out.get(("next", "state")),
                    "belief": model_out.get(("next", "belief")),
                    "bootstrap": fake_data.get("lambda_target")[..., 0, 0].reshape(
                        sample.batch_size
                    ),
                    "next": sample.get("next").select("reward", "done", "terminated"),
                },
                sample.batch_size,
            )
            replay_loss = value_loss.replay_value_loss(
                replay_features,
                horizon=cfg.optimization.continuation_horizon,
                lmbda=cfg.optimization.lmbda,
            )["loss_replay_value"]
            total_loss = (
                total_model_loss
                + actor_loss_td["loss_actor"]
                + value_loss_td["loss_value"]
                + cfg.optimization.replay_value_loss_weight * replay_loss
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        value_target_updater.step()
        metrics = torch.stack(
            (
                model_kl.detach().reshape(()),
                model_loss_td["loss_model_reco"].detach().reshape(()),
                model_loss_td["loss_model_reward"].detach().reshape(()),
                actor_loss_td["loss_actor"].detach().reshape(()),
                value_loss_td["loss_value"].detach().reshape(()),
                replay_loss.detach().reshape(()),
            )
        )
        return (
            metrics,
            model_out.get(("next", "state")).detach(),
            model_out.get(("next", "belief")).detach(),
        )

    if cfg.optimization.separate_policy_rng:
        # Model construction and policy inference use different random streams
        # in JAX. Start the learner stream independently after initialization,
        # on its own domain so it cannot alias the policy's first call.
        torch.manual_seed(jax_torch_seed(cfg.env.seed, 0, LEARNER_RNG_STREAM))

    for data in collector:
        # Collector data is yielded after its action was computed. Applying the
        # pending snapshot here therefore matches JAX's apply-after-action sync.
        if behavior_policy_sync is not None:
            behavior_policy_sync.apply_after_action()
        batch_start_action_step = action_step
        completed_episodes = training_episode_returns(
            data, running_training_return, num_envs
        )
        _log_train_episodes(
            cfg,
            metrics_jsonl_path,
            run_timer,
            completed_episodes,
            batch_start_action_step,
        )
        replay_data = replay_record_builder(data)
        with timeit("dreamer_v3/replay_extend"):
            if shifted_record_extender is not None:
                shifted_record_extender.extend(rb, replay_sampler, replay_data)
            else:
                replay_indices = rb.extend(
                    replay_data if num_envs > 1 else replay_data.reshape(-1)
                )
                replay_sampler.observe_extend(replay_indices, rb.storage)
        if (
            not replay_pipeline.has_prefetched
            and update_step == 0
            and len(rb) >= num_envs * (cfg.replay_buffer.seq_len + 1)
        ):
            # JAX starts a one-batch replay prefetch as soon as the first item
            # exists, well before the learner warmup gate. Cache the equivalent
            # initial batch instead of sampling it only when training begins.
            with timeit("dreamer_v3/replay_sample"):
                replay_pipeline.prefetch(rb)
        action_step += data.numel()
        record_step += replay_data.numel() if count_reset_records else data.numel()

        if len(rb) < warmup:
            continue

        batch_updates = (
            update_ratio(record_step) if update_ratio is not None else updates_per_batch
        )
        if not batch_updates:
            # The ratio scheduler is still paying off its remainder, so this
            # batch trains nothing. The reference likewise skips the record.
            continue

        if behavior_policy_sync is not None:
            # JAX stages the pre-update policy tree only once while a snapshot
            # is pending, even if several learner updates follow this record.
            behavior_policy_sync.stage_before_training()

        batch_losses = torch.empty((batch_updates, 6), device=device)
        for update_index in range(batch_updates):
            with timeit("dreamer_v3/replay_sample"):
                replay_sample, sample_info = replay_pipeline.take(rb)
                replay_sample = replay_sample.reshape(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len + 1,
                )
                sample = replay_sample[:, :-1].to(device)
            with timeit("dreamer_v3/replay_update"):
                # The successor is already sampled, so applying the older
                # refresh here preserves JAX's one-sample-ahead/one-output-
                # behind visibility.
                replay_pipeline.apply_pending_context(rb)
            with timeit("dreamer_v3/train_update"):
                (
                    update_losses,
                    refreshed_state,
                    refreshed_belief,
                ) = train_step(sample)
                batch_losses[update_index].copy_(update_losses)
                loss_window_sum += update_losses
                loss_window_updates += 1
            with timeit("dreamer_v3/replay_update"):
                replay_pipeline.stage_context(
                    sample_info,
                    refreshed_state,
                    refreshed_belief,
                )
            update_step += 1

        if record_loss_history:
            loss_history.append(batch_losses.cpu())

        train_log_due = bool(
            metrics_jsonl_path is not None
            and cfg.logger.train_every
            and (
                record_step >= next_train_log or action_step >= collector_action_frames
            )
        )
        eval_due = bool(cfg.logger.eval_every and record_step >= next_eval)
        latest_losses = batch_losses[-1].cpu() if eval_due else None
        if train_log_due:
            last_log_seconds, last_log_updates = _log_train_window(
                cfg=cfg,
                metrics_jsonl_path=metrics_jsonl_path,
                run_timer=run_timer,
                learner=learner,
                sample=sample,
                state_dim=state_dim,
                use_bfloat16=use_bfloat16,
                device=device,
                record_step=record_step,
                action_step=action_step,
                update_step=update_step,
                loss_window_sum=loss_window_sum,
                loss_window_updates=loss_window_updates,
                last_log_seconds=last_log_seconds,
                last_log_updates=last_log_updates,
            )
            loss_window_sum.zero_()
            loss_window_updates = 0
            next_train_log = record_step + cfg.logger.train_every

        if eval_due:
            r = _evaluate(
                cfg=cfg,
                device=device,
                eval_env=eval_env,
                real_world_actor=learner.real_world_actor,
                metrics_jsonl_path=metrics_jsonl_path,
                run_timer=run_timer,
                record_step=record_step,
                action_step=action_step,
                latest_losses=latest_losses,
            )
            history_steps.append(record_step)
            history_eval.append(r)
            next_eval = record_step + cfg.logger.eval_every

    if cfg.logger.output_plot:
        save_run_plot(cfg, history_steps, history_eval, loss_history)

    append_jsonl(
        metrics_jsonl_path,
        {
            "type": "summary",
            "backend": cfg.env.backend,
            "environment": cfg.env.name,
            "task": cfg.env.task,
            "seed": cfg.env.seed,
            "environment_seeded": bool(cfg.env.use_seed),
            "total_environment_steps": record_step,
            "total_action_steps": action_step,
            "updates": update_step,
            "elapsed_seconds": run_timer.elapsed(),
            "timings": timeit.todict(percall=False),
        },
    )
    if metrics_jsonl_path is not None:
        torchrl_logger.info("Saved run metrics to %s", metrics_jsonl_path)


if __name__ == "__main__":
    main()
