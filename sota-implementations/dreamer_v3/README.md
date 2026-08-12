# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration
and a proprioceptive DeepMind Control Walker Walk reproduction configuration.

Run the small example with:

```bash
python sota-implementations/dreamer_v3/dreamer_v3.py
```

Run the full Walker Walk configuration with:

```bash
python sota-implementations/dreamer_v3/dreamer_v3.py \
  --config-name=config_dmc_walker
```

The Walker preset uses the 1M-parameter RSSM dimensions, batches of 16 sequences
of length 64, a replay ratio of 1024, and 1.1 million environment steps. It logs
evaluation return against environment steps to JSON so curves can be compared
without relying on wall-clock-dependent training iterations.

For a three-seed median and interquartile reproduction run:

```bash
python sota-implementations/dreamer_v3/benchmark.py \
  --seeds 0 1 2 \
  --output-dir dmc_walker_runs
```

The benchmark writes one metrics file per seed plus `summary.json` and checks a
minimum final median return of 700. Use `--minimum-final-return` to override the
acceptance threshold when evaluating a deliberately smaller ablation. Full
learning-curve runs are intended for scheduled or manual validation; pull-request
CI uses short smoke overrides.

## JAX versus Torch training curves

The Walker preset additionally writes `scores.jsonl` from the existing
stochastic collector. This is a read-only diagnostic: it does not modify the
actor, collector data, replay, or optimization. Each completed episode contains:

- `step` and `episode/score`, matching the JAX score schema;
- `optimizer_updates`, the updates completed before the ending collector batch;
- maximum absolute actor `state` and `belief`;
- temporal variation of observations and actor distribution parameters.

These fields test the observed Torch execution path: observations vary while
the actor's RSSM inputs and action distribution remain unchanged. The actor is
intended to consume RSSM state and belief; the diagnostic does not add raw
observations to it.

Run three fresh JAX jobs from the unchanged JAX checkout at revision
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. Replace `SEED` with 0, 1, and 2,
and never reuse a partial log directory:

```bash
cd /root/dreamerv3
run_dir=/root/logdir/jax-reference/walker_walk-seedSEED
test ! -e "$run_dir" || { echo "refusing to reuse $run_dir"; exit 1; }
mkdir -p "$run_dir"
set -o pipefail
WANDB_PROJECT=dreamerv3-jax-torch \
WANDB_RUN_GROUP=dmc-walker-walk-defaults \
WANDB_JOB_TYPE=jax-reference \
WANDB_DIR="$run_dir" \
.venv/bin/python dreamerv3/main.py \
  --configs dmc_proprio \
  --seed SEED \
  --logger.outputs jsonl scope wandb \
  --logdir "$run_dir" \
  2>&1 | tee "$run_dir/console.log"
```

Run three Torch jobs from this checkout. W&B is disabled by default and is
enabled only by this explicit command:

```bash
cd /root/rl
run_dir=/root/logdir/torch-existing/walker_walk-seedSEED
test ! -e "$run_dir" || { echo "refusing to reuse $run_dir"; exit 1; }
mkdir -p "$run_dir"
set -o pipefail
.venv/bin/python sota-implementations/dreamer_v3/dreamer_v3.py \
  --config-name=config_dmc_walker \
  env.seed=SEED \
  logger.wandb.enabled=true \
  hydra.run.dir="$run_dir" \
  2>&1 | tee "$run_dir/console.log"
```

Both commands log `episode/score` against W&B's global environment step in
project `dreamerv3-jax-torch`, group `dmc-walker-walk-defaults`. Set
`WANDB_ENTITY=<team>` for JAX and add `logger.wandb.entity=<team>` for Torch
when using a team project. W&B project visibility is controlled in the W&B UI.

Retain each run's `scores.jsonl`, resolved config, console output, Git revision,
and package versions. No post-run uploader is required. Compare the six
`episode/score` histories directly in W&B and inspect every Torch score record
for these invariants:

```text
collector/observation_temporal_std > 0
collector/state_abs_max == 0
collector/belief_abs_max == 0
collector/loc_temporal_range == 0
collector/scale_temporal_range == 0
optimizer_updates > 0 after warmup
```

This experiment demonstrates behavioral differences between the checked-in
defaults and separately demonstrates Torch's unchanged collector latents. It
does not prove that the latent issue caused the entire curve gap. JAX uses
bfloat16 CUDA and 16 environments, Torch uses its current single-environment
settings, and the JAX DMC environment is not seeded by `--seed`; the three
seeds are independent replications, not paired trajectories. Precision,
one-environment, random-agent, and corrected-collector controls are optional
follow-up runs, not part of the initial six jobs.
