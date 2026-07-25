"""Extract a DreamerV3 reference learning curve from danijar/dreamerv3 scores.

The reference repo ships published curves as gzipped JSON under ``scores/``.
``scores/dmc_proprio-dreamerv3.json.gz`` holds every DMC-proprio task, 5 seeds
each, as records ``{task, method, seed, xs, ys}`` (xs = env steps, ys = eval
return). This script pulls out one task, writes a tidy per-seed CSV plus a
mean/min/max CSV, and renders the reference band -- so the GPU parity overlay
never needs the JAX repo checked out.

Usage (from repo root):
    .venv/bin/python dreamerv3-dmc-notes/scripts/extract_jax_curve.py \
        --scores /Users/dhruv/Documents/coding/uni/_ref/dreamerv3/scores/dmc_proprio-dreamerv3.json.gz \
        --task dmc_walker_walk

Everything is local. Nothing is pushed.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "dreamerv3-dmc-notes" / "reference"
PLOTDIR = REPO / "dreamerv3-dmc-notes" / "plots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="path to *-dreamerv3.json.gz")
    ap.add_argument("--task", default="dmc_walker_walk")
    args = ap.parse_args()

    records = json.load(gzip.open(args.scores))
    rows = [r for r in records if r.get("task") == args.task]
    if not rows:
        tasks = sorted({r.get("task") for r in records})
        raise SystemExit(f"task {args.task!r} not found. Available: {tasks}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    # Per-seed tidy CSV.
    per_seed = OUTDIR / f"{args.task}_dreamerv3.csv"
    with per_seed.open("w") as fh:
        fh.write("seed,step,return\n")
        for r in rows:
            for x, y in zip(r["xs"], r["ys"]):
                fh.write(f"{r['seed']},{int(x)},{y}\n")

    # Align seeds on the common step grid and reduce to mean/min/max.
    grids = [set(r["xs"]) for r in rows]
    common = sorted(set.intersection(*grids))
    by_seed = {r["seed"]: dict(zip(r["xs"], r["ys"])) for r in rows}
    mean_csv = OUTDIR / f"{args.task}_dreamerv3_mean.csv"
    means = []
    with mean_csv.open("w") as fh:
        fh.write("step,mean,min,max,n\n")
        for x in common:
            vals = [by_seed[s][x] for s in by_seed if x in by_seed[s]]
            m = sum(vals) / len(vals)
            means.append((x, m, min(vals), max(vals)))
            fh.write(f"{int(x)},{m},{min(vals)},{max(vals)},{len(vals)}\n")

    print(f"task={args.task}  seeds={sorted(by_seed)}  steps {common[0]:.0f}..{common[-1]:.0f}")
    print(f"final mean return @ {common[-1]:.0f}: {means[-1][1]:.1f}")
    print(f"wrote {per_seed}")
    print(f"wrote {mean_csv}")

    # Reference-band plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [m[0] for m in means]
    mu = [m[1] for m in means]
    lo = [m[2] for m in means]
    hi = [m[3] for m in means]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(xs, lo, hi, alpha=0.2, color="#1b9e77", label="JAX seed range")
    ax.plot(xs, mu, color="#1b9e77", lw=2, label="JAX DreamerV3 (mean)")
    ax.set(
        xlabel="env steps",
        ylabel="eval return",
        title=f"{args.task}: DreamerV3 reference (danijar, {len(by_seed)} seeds)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = PLOTDIR / f"{args.task}_reference.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
