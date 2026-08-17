#!/usr/bin/env bash
# Run the JAX reference on walker_walk for seeds 0, 1, 2 -- one at a time, 200k
# env steps each -- and copy each run's logs into the repo as it finishes.
#
#   ./run_jax_serial.sh [steps] [seeds...]
#
# Serial by design: the reference curve in /home/ubuntu/logdir was produced by a
# solo run, so running one seed at a time reproduces its conditions. It also
# means preallocation can stay on (the reference default) instead of the
# noprealloc workaround the earlier concurrent runs needed.
#
# Only metrics.jsonl, scores.jsonl and config.yaml are copied into the repo.
# The ckpt/, replay/, scope/ and plugins/ directories are deliberately left in
# /tmp: checkpoints and replay are large binaries, and scope/ is a redundant
# binary re-encoding of the scalars already in metrics.jsonl.
set -u

STEPS="${1:-200000}"
shift || true
SEEDS=("${@:-0 1 2}")
[ "${#SEEDS[@]}" -eq 1 ] && read -r -a SEEDS <<< "${SEEDS[0]}"

RL=/home/ubuntu/rl
SCRATCH="$RL/dreamerv3-parity"
JAXSEED="$SCRATCH/scripts/run_jax_seed.py"
RESULTS="$SCRATCH/results"
JAXPY=/home/ubuntu/dreamerv3/.venv/bin/python

mkdir -p "$SCRATCH/logs" "$RESULTS"

for seed in "${SEEDS[@]}"; do
  out=/tmp/jaxrun200k-seed$seed
  rm -rf "$out"
  echo "[$(date -u +%H:%M:%S)] starting jax seed $seed -> $out ($STEPS steps)"
  MUJOCO_GL=egl "$JAXPY" -u "$JAXSEED" "$seed" "$STEPS" "$out" \
    > "$SCRATCH/logs/jax-serial-seed$seed.log" 2>&1
  status=$?
  echo "[$(date -u +%H:%M:%S)] jax seed $seed exited with status $status"

  dest="$RESULTS/jax-seed$seed"
  mkdir -p "$dest"
  for f in metrics.jsonl scores.jsonl config.yaml; do
    [ -f "$out/$f" ] && cp "$out/$f" "$dest/$f"
  done
  echo "[$(date -u +%H:%M:%S)] copied seed $seed logs to $dest"
done

echo "[$(date -u +%H:%M:%S)] all seeds done"
