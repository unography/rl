"""Run the BASELINE DreamerV3 (main + DMC plumbing) on walker over N seeds.

This is the control arm: main's algorithm, no parity fixes. It overlays the
baseline eval curve on the committed JAX reference band; the expectation is that
the baseline does NOT match (stays flat/low), which is what makes the parity
work a real contribution.

Pass --parity-csv <path> to add every arm in the parity CSV (parity branch +
the V3-off ablation) and produce the combined money plot (JAX vs main-baseline
vs parity vs V3-off). The parity CSV is the `<task>_parity.csv` written by
dreamerv3-dmc-notes/scripts/run_dmc_parity.py on the parity branch (schema:
arm,step,mean,min,max) -- point at that worktree.

Usage (from repo root, on GPU):
    .venv/bin/python dreamerv3-baseline-notes/scripts/run_baseline.py \
        --seeds 0 1 2 --total-frames 500000 --device cuda

    # three-way, after both branches have run:
    .venv/bin/python dreamerv3-baseline-notes/scripts/run_baseline.py \
        --seeds 0 1 2 --device cuda \
        --parity-csv ../rl-parity-dmc/dreamerv3-dmc-notes/plots/dmc_walker_walk_parity.csv

Everything is local; nothing is pushed.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "bin" / "python"
EXAMPLE = REPO / "sota-implementations" / "dreamer_v3" / "dreamer_v3.py"
OUTDIR = REPO / "dreamerv3-baseline-notes" / "plots"
REFDIR = REPO / "dreamerv3-baseline-notes" / "reference"

LINE_RE = re.compile(r"env_step=\s*(?P<step>\d+)\].*?eval_reward=(?P<rew>[-+]?[\d.]+)")


def parse_log(path: Path) -> list[tuple[int, float]]:
    out = []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            out.append((int(m["step"]), float(m["rew"])))
    return out


def mean_min_max(curves):
    grids = [set(s for s, _ in c) for c in curves if c]
    if not grids:
        return [], [], [], []
    common = sorted(set.intersection(*grids))
    maps = [dict(c) for c in curves if c]
    xs, mu, lo, hi = [], [], [], []
    for x in common:
        vals = [m[x] for m in maps if x in m]
        xs.append(x); mu.append(sum(vals) / len(vals))
        lo.append(min(vals)); hi.append(max(vals))
    return xs, mu, lo, hi


def load_csv_band(path: Path, step="step", mean="mean", lo="min", hi="max", where=None):
    xs, mu, l, h = [], [], [], []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if where and not where(r):
                continue
            xs.append(int(r[step])); mu.append(float(r[mean]))
            l.append(float(r[lo])); h.append(float(r[hi]))
    return xs, mu, l, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--task", default="dmc_walker_walk")
    ap.add_argument("--domain", default="walker")
    ap.add_argument("--task-name", default="walk")
    ap.add_argument("--total-frames", type=int, default=500000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--parity-csv", default=None, help="parity branch <task>_parity.csv for the three-way plot")
    ap.add_argument("--skip-run", action="store_true", help="re-plot from existing logs")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = [
        f"env.domain={args.domain}", f"env.task={args.task_name}",
        f"env.device={args.device}",
        f"collector.total_frames={args.total_frames}",
        f"logger.eval_every={args.eval_every}",
    ]

    curves = []
    for seed in args.seeds:
        log = OUTDIR / f"{args.task}_baseline_seed{seed}.log"
        if not args.skip_run:
            cmd = [str(PY), str(EXAMPLE), "--config-name", "config_dmc",
                   f"env.seed={seed}", *base]
            sys.stderr.write(f"[baseline] seed {seed}: {' '.join(cmd)}\n")
            with log.open("w") as fh:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               env=dict(os.environ), check=True)
        curves.append(parse_log(log))
    bxs, bmu, blo, bhi = mean_min_max(curves)

    # baseline CSV.
    bcsv = OUTDIR / f"{args.task}_baseline.csv"
    with bcsv.open("w") as fh:
        fh.write("step,mean,min,max\n")
        for x, m, l, h in zip(bxs, bmu, blo, bhi):
            fh.write(f"{x},{m},{l},{h}\n")
    sys.stderr.write(f"[baseline] wrote {bcsv}\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ref = REFDIR / f"{args.task}_dreamerv3_mean.csv"
    if ref.exists():
        rxs, rmu, rlo, rhi = load_csv_band(ref)
        ax.fill_between(rxs, rlo, rhi, alpha=0.15, color="#1b9e77")
        ax.plot(rxs, rmu, color="#1b9e77", lw=2.5, label="JAX DreamerV3 (ref, 5 seeds)")
    if bxs:
        ax.fill_between(bxs, blo, bhi, alpha=0.15, color="#d95f02")
        ax.plot(bxs, bmu, color="#d95f02", lw=2, label=f"torchrl main baseline ({len(args.seeds)} seeds)")
    title = f"{args.task}: baseline (main) vs JAX"
    if args.parity_csv:
        # Plot every arm present in the parity CSV (e.g. branch=all V3 features,
        # v3off=features removed) for the full comparison.
        import collections

        arm_styles = {
            "branch": ("#7570b3", "torchrl parity (all V3 features)"),
            "v3off": ("#e7298a", "torchrl V3-off (features removed)"),
        }
        rows = collections.defaultdict(list)
        with Path(args.parity_csv).open() as fh:
            for r in csv.DictReader(fh):
                rows[r.get("arm", "branch")].append(r)
        for arm, rlist in rows.items():
            xs = [int(r["step"]) for r in rlist]
            mu = [float(r["mean"]) for r in rlist]
            lo = [float(r["min"]) for r in rlist]
            hi = [float(r["max"]) for r in rlist]
            color, label = arm_styles.get(arm, ("#666666", f"torchrl {arm}"))
            ax.fill_between(xs, lo, hi, alpha=0.15, color=color)
            ax.plot(xs, mu, color=color, lw=2, label=label)
        title = f"{args.task}: JAX vs main-baseline vs parity vs V3-off"
    ax.set(xlabel="env steps", ylabel="eval return", title=title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = OUTDIR / (f"{args.task}_combined.png" if args.parity_csv else f"{args.task}_baseline_vs_jax.png")
    fig.savefig(png, dpi=130)
    sys.stderr.write(f"[baseline] wrote {png}\n")

    if bxs:
        print(f"baseline: final mean return @ {bxs[-1]} = {bmu[-1]:.1f}  (JAX ~881)")


if __name__ == "__main__":
    main()
