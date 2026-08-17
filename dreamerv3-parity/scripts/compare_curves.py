"""Compare DreamerV3 walker_walk training curves and internals: JAX vs Torch."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROUNDS = [16000, 32000, 48000]


def jax_curve(path):
    rows = [json.loads(line) for line in open(path)]
    buckets = defaultdict(list)
    for row in rows:
        buckets[((row["step"] - 1) // 16000) * 16000].append(row["episode/score"])
    return {k: v for k, v in sorted(buckets.items())}


def torch_curve(path):
    rows = [json.loads(line) for line in open(path)]
    buckets = defaultdict(list)
    for row in rows:
        if row.get("type") == "train_episode":
            buckets[row["action_steps"]].append(row["score"])
    return {k: v for k, v in sorted(buckets.items())}


jax_runs, torch_runs = {}, {}
for name, path in [
    ("jax s0", "/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/scores.jsonl"),
    ("jax s1", "/tmp/jaxrun-seed1/scores.jsonl"),
    ("jax s2", "/tmp/jaxrun-seed2/scores.jsonl"),
]:
    if Path(path).exists() and Path(path).stat().st_size:
        jax_runs[name] = jax_curve(path)
for seed in (0, 1, 2):
    path = f"/tmp/dv3-seed{seed}/metrics.jsonl"
    if Path(path).exists() and Path(path).stat().st_size:
        torch_runs[f"torch s{seed}"] = torch_curve(path)

names = list(jax_runs) + list(torch_runs)
allruns = {**jax_runs, **torch_runs}
print("Episode return (mean over the 16 parallel episodes ending at that step)")
print(f"{'step':>7s} " + " ".join(f"{n:>9s}" for n in names))
for step in ROUNDS:
    cells = []
    for n in names:
        v = allruns[n].get(step)
        cells.append(f"{statistics.mean(v):9.1f}" if v else f"{'-':>9s}")
    print(f"{step:>7d} " + " ".join(cells))

print()
for label, runs in (("JAX", jax_runs), ("Torch", torch_runs)):
    print(f"{label} across seeds:")
    for step in ROUNDS:
        vals = [statistics.mean(r[step]) for r in runs.values() if r.get(step)]
        if len(vals) >= 2:
            print(
                f"  {step:>6d}: n={len(vals)} "
                f"mean={statistics.mean(vals):7.1f} "
                f"min={min(vals):7.1f} max={max(vals):7.1f} "
                f"sd={statistics.stdev(vals):7.1f}"
            )
        elif vals:
            print(f"  {step:>6d}: n=1 value={vals[0]:7.1f}")
