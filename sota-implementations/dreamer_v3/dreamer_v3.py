# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 training script, reproducing the reference JAX implementation.

Proprioceptive (not pixel-based): observations are flat feature vectors whose
reconstruction loss sums the event dimension before averaging batch and time.
Two configurations ship with it — a compact ``Pendulum-v1`` smoke test and
``config_dmc_walker``, which reproduces the reference DeepMind Control
``walker_walk`` setup at 640,867 trainable parameters.

Real collection and evaluation run on CPU; ``optimization.device`` selects where
the models, losses and policy run. Metrics stream to JSONL on the same step axis
the reference logs, and ``logger.diagnostics`` adds its ``train/*`` scalars for
a term-by-term comparison.

Usage::

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        collector.total_frames=5000 logger.eval_every=500

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        --config-name=config_dmc_walker
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import hydra
import torch

from dreamer_v3_utils import (
    _append_jsonl,
    _collector_action_budget,
    _DreamerV3BehaviorPolicySync,
    _DreamerV3Optimizer,
    _DreamerV3ReplayPipeline,
    _DreamerV3ReplayRecordBuilder,
    _DreamerV3ReplaySampler,
    _DreamerV3SeededPolicy,
    _DreamerV3ShiftedReplayWriter,
    _DreamerV3UpdateRatio,
    _driver_step_for_action,
    _jax_torch_seed,
    _LEARNER_RNG_STREAM,
    _reference_diagnostics,
    _REPLAY_RNG_STREAM,
    _training_episode_returns,
    build_actor,
    build_continuation_model,
    build_imagination_model,
    build_mb_env,
    build_real_world_actor,
    build_value,
    build_world_model,
    eval_episode_reward,
    make_env,
)
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase

from torchrl import timeit
from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


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
    if num_envs <= 0:
        raise ValueError(f"collector.num_envs must be positive, got {num_envs}.")
    if cfg.collector.frames_per_batch % num_envs:
        raise ValueError(
            "collector.frames_per_batch must be divisible by collector.num_envs, "
            f"got {cfg.collector.frames_per_batch} and {num_envs}."
        )
    count_reset_records = bool(cfg.collector.count_reset_records)
    collector_action_frames = (
        _collector_action_budget(
            cfg.collector.total_frames,
            num_envs,
            cfg.env.max_episode_steps,
        )
        if count_reset_records
        else cfg.collector.total_frames
    )
    if collector_action_frames % cfg.collector.frames_per_batch:
        raise ValueError(
            "The action budget derived from collector.total_frames must be "
            "divisible by collector.frames_per_batch, got "
            f"{collector_action_frames} and {cfg.collector.frames_per_batch}."
        )
    real_env = make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    metrics_jsonl_path = (
        Path(cfg.logger.metrics_jsonl).resolve() if cfg.logger.metrics_jsonl else None
    )
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("")
    timeit.reset()
    run_timer = timeit("dreamer_v3/run").start()

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
    optimizer = _DreamerV3Optimizer(
        trainable_parameters,
        lr=cfg.optimization.lr,
        agc=cfg.optimization.adaptive_grad_clip,
        beta1=0.9,
        beta2=0.999,
        eps=cfg.optimization.adam_eps,
        warmup_steps=cfg.optimization.warmup_steps,
    )

    real_world_actor = build_real_world_actor(
        world_model=world_model,
        actor_model=actor_model,
        mixed_precision=cfg.optimization.mixed_precision,
    )
    if cfg.optimization.jax_behavior_policy_sync:
        collector_actor = copy.deepcopy(real_world_actor)
        # JAX's policy-key regex also shadows the decoder. Its carry is empty
        # for proprioception and it is action-dead, but include it for an exact
        # parameter-tree handoff rather than only action-equivalent behavior.
        behavior_decoder = copy.deepcopy(world_model[2])
        learner_policy_tree = torch.nn.ModuleList([real_world_actor, world_model[2]])
        behavior_policy_tree = torch.nn.ModuleList([collector_actor, behavior_decoder])
        behavior_policy_sync = _DreamerV3BehaviorPolicySync(
            learner_policy_tree, behavior_policy_tree
        )
    else:
        collector_actor = real_world_actor
        behavior_policy_sync = None
    collector_policy = (
        _DreamerV3SeededPolicy(collector_actor, cfg.env.seed)
        if cfg.optimization.separate_policy_rng
        else collector_actor
    )

    def make_explore_env(index: int):
        seed = cfg.env.seed + 2 + index if cfg.env.use_seed else None
        return TransformedEnv(
            make_env(cfg, seed),
            TensorDictPrimer(
                random=False,
                default_value=0,
                state=Unbounded(state_dim),
                belief=Unbounded(cfg.networks.rnn_hidden_dim),
                previous_action=Unbounded(action_dim),
            ),
        )

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
    if isinstance(collector_policy, _DreamerV3SeededPolicy):
        # Collector construction probes policy output keys once. JAX initializes
        # those keys without consuming a real-action seed, so restart the
        # isolated policy stream before the first environment transition.
        collector_policy.reset_counter()

    replay_sampler = _DreamerV3ReplaySampler(
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
            _jax_torch_seed(cfg.env.seed, 0, _REPLAY_RNG_STREAM)
        ),
    )
    replay_record_builder = _DreamerV3ReplayRecordBuilder(num_envs)
    shifted_replay_writer = (
        _DreamerV3ShiftedReplayWriter(num_envs) if count_reset_records else None
    )
    replay_pipeline = _DreamerV3ReplayPipeline()

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
    record_loss_history = bool(cfg.logger.output_plot and _has_matplotlib)
    next_eval = 0
    next_train_log = 0
    # Interval anchors: rates are reported since the previous log, as
    # elements.FPS does.
    last_log_seconds = run_timer.elapsed()
    last_log_updates = 0

    eval_env = TransformedEnv(
        make_env(cfg, cfg.env.seed + 100),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
            previous_action=Unbounded(action_dim),
        ),
    )

    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )
    warmup = max(warmup, num_envs * (cfg.replay_buffer.seq_len + 1))

    updates_per_batch = cfg.optimization.updates_per_batch
    update_ratio = (
        _DreamerV3UpdateRatio(
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
        torch.manual_seed(_jax_torch_seed(cfg.env.seed, 0, _LEARNER_RNG_STREAM))

    for data in collector:
        # Collector data is yielded after its action was computed. Applying the
        # pending snapshot here therefore matches JAX's apply-after-action sync.
        if behavior_policy_sync is not None:
            behavior_policy_sync.apply_after_action()
        batch_start_action_step = action_step
        completed_episodes = _training_episode_returns(
            data, running_training_return, num_envs
        )
        for time_index, env_index, score in completed_episodes:
            if count_reset_records:
                action_index = batch_start_action_step // num_envs + time_index + 1
                episode_step = _driver_step_for_action(
                    action_index,
                    env_index,
                    num_envs,
                    cfg.env.max_episode_steps,
                )
            else:
                episode_step = (
                    batch_start_action_step + time_index * num_envs + env_index + 1
                )
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "type": "train_episode",
                    "environment_steps": episode_step,
                    "action_steps": batch_start_action_step
                    + (time_index + 1) * num_envs,
                    "score": score,
                    "elapsed_seconds": run_timer.elapsed(),
                },
            )
        replay_data = replay_record_builder(data)
        with timeit("dreamer_v3/replay_extend"):
            if shifted_replay_writer is not None:
                shifted_replay_writer.extend(rb, replay_sampler, replay_data)
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

        batch_losses = torch.empty(
            (batch_updates, 6) if record_loss_history else (6,),
            device=device,
        )
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
                if record_loss_history:
                    batch_losses[update_index].copy_(update_losses)
                else:
                    batch_losses.copy_(update_losses)
                loss_window_sum += update_losses.detach()
                loss_window_updates += 1
            with timeit("dreamer_v3/replay_update"):
                replay_pipeline.stage_context(
                    sample_info,
                    refreshed_state,
                    refreshed_belief,
                )
            update_step += 1

        if record_loss_history:
            loss_history.append(batch_losses.detach().cpu())

        train_log_due = bool(
            metrics_jsonl_path is not None
            and cfg.logger.train_every
            and (
                record_step >= next_train_log or action_step >= collector_action_frames
            )
        )
        eval_due = bool(cfg.logger.eval_every and record_step >= next_eval)
        latest_losses = (
            (batch_losses[-1] if record_loss_history else batch_losses).detach().cpu()
            if train_log_due or eval_due
            else None
        )
        if train_log_due:
            # Read the clock before the optional diagnostics pass: it runs a
            # full world-model and imagination forward, and charging that to the
            # interval would deflate the very rate it is logged next to.
            elapsed_seconds = run_timer.elapsed()
            diagnostics = (
                _reference_diagnostics(
                    model_loss=model_loss,
                    actor_loss=actor_loss,
                    value_loss=value_loss,
                    sample=sample,
                    state_dim=state_dim,
                    rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
                    use_bfloat16=use_bfloat16,
                    device=device,
                )
                if cfg.logger.get("diagnostics", False)
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
            last_log_seconds = run_timer.elapsed()
            last_log_updates = update_step
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "type": "train",
                    "environment_steps": record_step,
                    "action_steps": action_step,
                    "updates": update_step,
                    # Averaged over every update since the previous log, as
                    # the reference reports. A single update is a noisy sample.
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
                            (loss_window_sum / max(loss_window_updates, 1))
                            .detach()
                            .cpu()
                            .tolist(),
                        )
                    ),
                    "training_fps": training_fps,
                    "elapsed_seconds": elapsed_seconds,
                    "bfloat16": use_bfloat16,
                    **diagnostics,
                },
            )
            loss_window_sum.zero_()
            loss_window_updates = 0
            next_train_log = record_step + cfg.logger.train_every

        if eval_due:
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
            history_steps.append(record_step)
            history_eval.append(r)
            torchrl_logger.info(
                "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
                record_step,
                r.item(),
                latest_losses[0].item(),
                latest_losses[1].item(),
                latest_losses[2].item(),
                latest_losses[3].item(),
            )
            _append_jsonl(
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
            next_eval = record_step + cfg.logger.eval_every

    if cfg.logger.output_plot and _has_matplotlib:
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dep)

        eval_steps = history_steps
        eval_rewards = torch.stack(history_eval).cpu().numpy() if history_eval else []
        loss_curves = torch.cat(loss_history).numpy() if loss_history else None
        kl_vals = loss_curves[:, 0] if loss_curves is not None else []
        reco_vals = loss_curves[:, 1] if loss_curves is not None else []
        reward_vals = loss_curves[:, 2] if loss_curves is not None else []

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
            f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps"
        )
        fig.tight_layout()
        fig.savefig(cfg.logger.output_plot, dpi=120)
        torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
    elif cfg.logger.output_plot:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )

    _append_jsonl(
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
