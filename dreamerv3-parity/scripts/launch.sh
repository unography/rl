#!/usr/bin/env bash
# Launch the extended JAX-vs-Torch walker_walk curve comparison (200k steps).
set -u

RL=/home/ubuntu/rl
SCRATCH="$RL/dreamerv3-parity"
JAXSEED="$SCRATCH/scripts/run_jax_seed.py"
STEPS=200000

mkdir -p "$SCRATCH/logs"

for seed in 0 1 2; do
  out=/tmp/dv3-200k-seed$seed
  mkdir -p "$out"
  MUJOCO_GL=egl nohup "$RL/.venv/bin/python" \
    "$RL/sota-implementations/dreamer_v3/dreamer_v3.py" \
    --config-name=config_dmc_walker \
    env.seed=$seed \
    collector.total_frames=$STEPS \
    logger.train_every=4096 \
    logger.diagnostics=true \
    logger.metrics_json=$out/metrics.json \
    logger.metrics_jsonl=$out/metrics.jsonl \
    hydra.run.dir=$out/hydra \
    >"$SCRATCH/logs/torch-seed$seed.log" 2>&1 &
  echo "torch seed$seed pid=$!"
done

# Launch the JAX runs one at a time: two simultaneous startups race and one
# dies silently (empty log). Give each ~60s to get past init.
for seed in 1 2; do
  out=/tmp/jaxrun200k-seed$seed
  rm -rf "$out"
  MUJOCO_GL=egl nohup /home/ubuntu/dreamerv3/.venv/bin/python -u \
    "$JAXSEED" $seed $STEPS "$out" noprealloc \
    >"$SCRATCH/logs/jax-seed$seed.log" 2>&1 &
  echo "jax seed$seed pid=$!"
  sleep 60
done
