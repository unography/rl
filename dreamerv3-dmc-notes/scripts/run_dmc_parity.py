"""Run the DreamerV3 example on a DMC task over N seeds and overlay vs JAX.

Intended for the GPU box. For each seed it runs
``sota-implementations/dreamer_v3/dreamer_v3.py --config-name config_dmc``,
captures the eval log, then plots this branch's mean/min/max eval-return curve
against the committed JAX reference band
(``dreamerv3-dmc-notes/reference/<task>_dreamerv3_mean.csv``, produced by
``extract_jax_curve.py`` -- no JAX repo needed at run time).

Usage (from repo root, on GPU):
    .venv/bin/python dreamerv3-dmc-notes/scripts/run_dmc_parity.py \
        --seeds 0 1 2 --total-frames 500000 --device cuda

Add ablation arms with --extra-arm NAME "override=val override2=val" to run,
e.g., the buggy KL for a mechanism A/B. Everything is local; nothing is pushed.
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
OUTDIR = REPO / "dreamerv3-dmc-notes" / "plots"
REFDIR = REPO / "dreamerv3-dmc-notes" / "reference"

LINE_RE = re.compile(r"env_step=\s*(?P<step>\d+)\].*?eval_reward=(?P<rew>[-+]?[\d.]+)")


def parse_log(path: Path) -> list[tuple[int, float]]:
    rows = []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            rows.append((int(m["step"]), float(m["rew"])))
    return rows


def run_seed(seed: int, arm_overrides: list[str], base_overrides: list[str], log_path: Path) -> None:
    cmd = [str(PY), str(EXAMPLE), "--config-name", "config_dmc",
           f"env.seed={seed}", *base_overrides, *arm_overrides]
    sys.stderr.write(f"[dmc-parity] {log_path.name}: {' '.join(cmd)}\n")
    with log_path.open("w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                       env=dict(os.environ), check=True)


def load_reference(task: str):
    csv_path = REFDIR / f"{task}_dreamerv3_mean.csv"
    if not csv_path.exists():
        return None
    xs, mu, lo, hi = [], [], [], []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            xs.append(int(r["step"])); mu.append(float(r["mean"]))
            lo.append(float(r["min"])); hi.append(float(r["max"]))
    return xs, mu, lo, hi


def mean_min_max(curves: list[list[tuple[int, float]]]):
    # Align on steps present in every seed's curve.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--task", default="dmc_walker_walk", help="for the reference overlay + labels")
    ap.add_argument("--domain", default="walker")
    ap.add_argument("--task-name", default="walk")
    ap.add_argument("--total-frames", type=int, default=500000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--extra-arm", nargs=2, action="append", default=[],
                    metavar=("NAME", "OVERRIDES"), help='e.g. --extra-arm buggykl ""')
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = [
        f"env.domain={args.domain}", f"env.task={args.task_name}",
        f"env.device={args.device}",
        f"collector.total_frames={args.total_frames}",
        f"logger.eval_every={args.eval_every}",
    ]

    arms = {"branch": []}
    for name, ov in args.extra_arm:
        arms[name] = ov.split() if ov else []

    # Run every arm x seed; collect curves.
    data = {}
    for arm, arm_ov in arms.items():
        curves = []
        for seed in args.seeds:
            log = OUTDIR / f"{args.task}_{arm}_seed{seed}.log"
            run_seed(seed, arm_ov, base, log)
            curves.append(parse_log(log))
        data[arm] = mean_min_max(curves)

    # CSV.
    csv_path = OUTDIR / f"{args.task}_parity.csv"
    with csv_path.open("w") as fh:
        fh.write("arm,step,mean,min,max\n")
        for arm, (xs, mu, lo, hi) in data.items():
            for x, m, l, h in zip(xs, mu, lo, hi):
                fh.write(f"{arm},{x},{m},{l},{h}\n")
    sys.stderr.write(f"[dmc-parity] wrote {csv_path}\n")

    # Overlay plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ref = load_reference(args.task)
    if ref:
        xs, mu, lo, hi = ref
        ax.fill_between(xs, lo, hi, alpha=0.15, color="#1b9e77")
        ax.plot(xs, mu, color="#1b9e77", lw=2.5, label="JAX DreamerV3 (ref, 5 seeds)")
    palette = ["#d95f02", "#7570b3", "#e7298a"]
    for i, (arm, (xs, mu, lo, hi)) in enumerate(data.items()):
        if not xs:
            continue
        c = palette[i % len(palette)]
        ax.fill_between(xs, lo, hi, alpha=0.15, color=c)
        ax.plot(xs, mu, color=c, lw=2, label=f"torchrl {arm} ({len(args.seeds)} seeds)")
    ax.set(xlabel="env steps", ylabel="eval return",
           title=f"{args.task}: torchrl DreamerV3 vs JAX reference")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = OUTDIR / f"{args.task}_parity.png"
    fig.savefig(png, dpi=130)
    sys.stderr.write(f"[dmc-parity] wrote {png}\n")

    for arm, (xs, mu, _, _) in data.items():
        if xs:
            print(f"{arm:8s}: final mean return @ {xs[-1]} = {mu[-1]:.1f}")


if __name__ == "__main__":
    main()
