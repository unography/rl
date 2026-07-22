"""Orchestrate the KL free-nats A/B on Pendulum and plot the two learning curves.

Runs the dreamer_v3 example twice through ``kl_ablation_entry.py`` -- once with the
branch's fixed ``categorical_kl_balanced`` (summed-then-clamped free nats) and once
with the origin/main buggy version (per-categorical clamp) -- with identical config
and seed. Parses eval_reward / kl per env step from each run's log and writes an
overlaid plot plus a CSV.

Everything is local. Nothing is pushed.

Usage (from repo root):
    .venv/bin/python dreamerv3-parity-notes/run_kl_ablation.py \
        --frames 50000 --seed 0 --updates 16 --eval-every 2000

Add --reuse-log fixed=<path> to skip re-running an arm whose log you already have
(e.g. the calibration run).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "dreamerv3-parity-notes" / "kl_ablation_entry.py"
PY = REPO / ".venv" / "bin" / "python"
OUTDIR = REPO / "dreamerv3-parity-notes" / "results"

LINE_RE = re.compile(
    r"env_step=\s*(?P<step>\d+)\].*?eval_reward=(?P<rew>-?[\d.]+).*?kl=(?P<kl>-?[\d.]+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            rows.append(
                {
                    "step": int(m["step"]),
                    "eval_reward": float(m["rew"]),
                    "kl": float(m["kl"]),
                }
            )
    return rows


def run_arm(mode: str, overrides: list[str], log_path: Path) -> None:
    env = dict(os.environ, KL_MODE=mode)
    cmd = [str(PY), str(ENTRY), *overrides]
    sys.stderr.write(f"[ablation] running {mode}: {' '.join(cmd)}\n")
    with log_path.open("w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--cat", type=int, default=32)
    ap.add_argument("--classes", type=int, default=32)
    ap.add_argument("--rnn-hidden", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--reuse-log", action="append", default=[], help="mode=path")
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    overrides = [
        f"networks.num_categoricals={args.cat}",
        f"networks.num_classes={args.classes}",
        f"networks.rnn_hidden_dim={args.rnn_hidden}",
        f"networks.hidden_dim={args.hidden}",
        f"optimization.updates_per_batch={args.updates}",
        f"collector.total_frames={args.frames}",
        f"logger.eval_every={args.eval_every}",
        f"env.seed={args.seed}",
    ]

    reuse = dict(kv.split("=", 1) for kv in args.reuse_log)
    logs = {}
    for mode in ("fixed", "buggy"):
        if mode in reuse:
            logs[mode] = Path(reuse[mode])
            sys.stderr.write(f"[ablation] reusing {mode} log: {logs[mode]}\n")
        else:
            logs[mode] = OUTDIR / f"kl_{mode}.log"
            run_arm(mode, overrides, logs[mode])

    data = {mode: parse_log(path) for mode, path in logs.items()}

    # CSV
    csv_path = OUTDIR / "kl_ablation.csv"
    with csv_path.open("w") as fh:
        fh.write("mode,step,eval_reward,kl\n")
        for mode, rows in data.items():
            for r in rows:
                fh.write(f"{mode},{r['step']},{r['eval_reward']},{r['kl']}\n")
    sys.stderr.write(f"[ablation] wrote {csv_path}\n")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fixed": "#1b9e77", "buggy": "#d95f02"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, rows in data.items():
        if not rows:
            continue
        xs = [r["step"] for r in rows]
        ax0.plot(xs, [r["eval_reward"] for r in rows], label=mode, color=colors[mode], lw=2)
        ax1.plot(xs, [r["kl"] for r in rows], label=mode, color=colors[mode], lw=2)
    ax0.set(xlabel="env steps", ylabel="eval return", title="Pendulum-v1 eval return")
    ax1.set(xlabel="env steps", ylabel="KL loss (nats)", title="world-model KL")
    ax1.axhline(1.0, ls="--", c="gray", lw=1, label="free_bits=1.0")
    for ax in (ax0, ax1):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"DreamerV3 KL free-nats ablation (cat={args.cat}, cls={args.classes}, seed={args.seed})"
    )
    fig.tight_layout()
    png = OUTDIR / "kl_ablation.png"
    fig.savefig(png, dpi=130)
    sys.stderr.write(f"[ablation] wrote {png}\n")

    # Console summary
    for mode, rows in data.items():
        if rows:
            final = rows[-1]
            best = max(r["eval_reward"] for r in rows)
            print(f"{mode:6s}: final_return={final['eval_reward']:.1f}  best={best:.1f}  final_kl={final['kl']:.3f}")


if __name__ == "__main__":
    main()
