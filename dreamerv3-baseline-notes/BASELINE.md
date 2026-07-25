# DreamerV3 baseline (control arm) — `dreamerv3-baseline-dmc`

Blunt. This branch exists to **prove the parity work is a real contribution** by
showing what torchrl `main` does on the same task, with no fixes.

## What this branch is
- `main` + **one** change: minimal env plumbing so the unmodified example can run
  DM Control (`make_env` dmc dispatch + CatTensors flatten + DoubleToFloat +
  config-driven eval length). **Nothing algorithmic touched.**
- Diff vs main: 1 commit, `sota-implementations/dreamer_v3/` only.

## What it still has (the parity branch fixes all of these)
- **Blind acting policy** — collector + eval use the bare actor with
  zero-initialized latents; the agent cannot perceive obs while acting. This
  alone prevents learning (proven on Pendulum: `[[dreamerv3-example-doesnt-learn-pendulum]]`).
- **No V3 ingredients** — no unimix, block-GRU, two-hot critic, EMA target,
  retnorm, bounded-normal actor; plain Adam (no AGC/warmup).
- **The two loss bugs** — per-categorical KL free-nats, reco double-symlog.

## Why it's a fair control
- Same env, same eval protocol (1000-step episodes), same model capacity
  (**size1m**, identical to the parity `config_dmc.yaml`), same training budget
  and `train_ratio` (1024). The **only** difference is the algorithm.

## Expectation
- Baseline does **not** match the JAX walker curve (final mean ~881). It should
  stay flat/low.
- If it *did* learn walker, the parity work would be redundant. It won't.

## Run it (GPU)
```bash
# baseline only, overlay vs JAX reference band:
.venv/bin/python dreamerv3-baseline-notes/scripts/run_baseline.py \
  --seeds 0 1 2 --total-frames 500000 --device cuda
# -> plots/dmc_walker_walk_baseline_vs_jax.png  + dmc_walker_walk_baseline.csv
```

## The money plot (three-way: JAX vs baseline vs parity)
After the parity branch has produced its curve (`dmc_walker_walk_parity.csv` via
`dreamerv3-dmc-notes/scripts/run_dmc_parity.py` in that worktree):
```bash
.venv/bin/python dreamerv3-baseline-notes/scripts/run_baseline.py \
  --seeds 0 1 2 --device cuda --skip-run \
  --parity-csv ../rl-parity-dmc/dreamerv3-dmc-notes/plots/dmc_walker_walk_parity.csv
# -> plots/dmc_walker_walk_three_way.png
```
(`--skip-run` re-plots from existing baseline logs. Point `--parity-csv` at the
parity worktree's CSV.) This single figure is the PR headline: JAX matches
parity, baseline does not.

## CPU smoke (proves the DMC path, not learning)
```bash
.venv/bin/python sota-implementations/dreamer_v3/dreamer_v3.py --config-name config_dmc \
  env.domain=cartpole env.task=balance env.device=cpu \
  networks.rnn_hidden_dim=32 networks.num_categoricals=4 networks.num_classes=4 \
  networks.hidden_dim=32 networks.obs_embed_dim=32 replay_buffer.seq_len=16 \
  collector.total_frames=1000 collector.frames_per_batch=200 \
  optimization.updates_per_batch=2 logger.eval_every=400 logger.eval_episodes=1 \
  logger.eval_max_steps=100
```

## References
- Parity branch (the fixes): `dreamerv3-jax-parity-dmc` + its `dreamerv3-dmc-notes/`.
- JAX curves: `_ref/dreamerv3/scores/dmc_proprio-dreamerv3.json.gz`.
- Paper: https://arxiv.org/abs/2301.04104. Reference: danijar/dreamerv3 @ e3f0224.
