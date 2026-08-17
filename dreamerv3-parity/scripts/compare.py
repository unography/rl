#!/usr/bin/env python
"""Compare DreamerV3 walker_walk runs: TorchRL vs the JAX reference.

Reads whatever run directories exist and prints (a) the episode-return curve
and (b) a term-by-term comparison of the reference ``train/*`` scalars against
the Torch ``logger.diagnostics=true`` output.

Usage::

    .venv/bin/python dreamerv3-parity/scripts/compare.py
    .venv/bin/python dreamerv3-parity/scripts/compare.py --bucket 25000
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# (label, path). Missing paths are skipped, so this list can name both the
# finished 50k runs and the longer 200k runs at once.
JAX_RUNS = [
    # The 200k serial runs are the reference of record; results/ holds the copy
    # committed to the repo, /tmp holds the live run in progress.
    ("jax s0", "dreamerv3-parity/results/jax-seed0"),
    ("jax s1", "dreamerv3-parity/results/jax-seed1"),
    ("jax s2", "dreamerv3-parity/results/jax-seed2"),
    ("jaxlive s0", "/tmp/jaxrun200k-seed0"),
    ("jaxlive s1", "/tmp/jaxrun200k-seed1"),
    ("jaxlive s2", "/tmp/jaxrun200k-seed2"),
    # The original 1.1M-budget reference run (reached ~438k steps).
    ("jaxref s0", "/home/ubuntu/logdir/jax-dmc/walker_walk-seed0"),
    # Earlier partial runs, kept for their 16k-48k reference buckets.
    ("jax50k s1", "/tmp/jaxrun-seed1"),
    ("jax50k s2", "/tmp/jaxrun-seed2"),
    ("jaxpart s1", "/tmp/jaxrun-partial-seed1"),
    ("jaxpart s2", "/tmp/jaxrun-partial-seed2"),
]
TORCH_RUNS = [
    ("torch s0", "/tmp/dv3-200k-seed0/metrics.jsonl"),
    ("torch s1", "/tmp/dv3-200k-seed1/metrics.jsonl"),
    ("torch s2", "/tmp/dv3-200k-seed2/metrics.jsonl"),
    # Stopped at ~67.6k to hand the GPU to the serial JAX runs.
    ("torchpart s0", "/tmp/dv3-partial-seed0/metrics.jsonl"),
    ("torchpart s1", "/tmp/dv3-partial-seed1/metrics.jsonl"),
    ("torchpart s2", "/tmp/dv3-partial-seed2/metrics.jsonl"),
    ("torch50k s0", "/tmp/dv3-seed0/metrics.jsonl"),
    ("torch50k s1", "/tmp/dv3-seed1/metrics.jsonl"),
    ("torch50k s2", "/tmp/dv3-seed2/metrics.jsonl"),
]

# JAX ``train/*`` key -> Torch diagnostics key. World-model loss terms are
# reported unweighted on both sides (see _reference_diagnostics).
PAIRS = [
    ("train/loss/dyn", "loss_dynamic"),
    ("train/loss/rep", "loss_representation"),
    ("train/loss/con", "loss_continue"),
    ("train/loss/rew", "loss_reward"),
    ("train/loss/policy", "loss_actor"),
    ("train/loss/value", "loss_value"),
    ("train/loss/repval", "loss_replay_value"),
    ("train/val", "val"),
    ("train/slowval", "slowval"),
    ("train/ret", "ret"),
    ("train/ret_max", "ret_max"),
    ("train/ret_rate", "ret_rate"),
    ("train/adv", "adv"),
    ("train/adv_mag", "adv_mag"),
    ("train/adv_std", "adv_std"),
    ("train/ent/action", "ent_action"),
    ("train/weight", "weight"),
    ("train/con", "con"),
    ("train/rew", "rew"),
]


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially flushed final line while a run is live
    return rows


def jax_scores(rundir):
    """Episode returns keyed by env step. Prefers scores.jsonl, falls back."""
    for name, key in (
        ("scores.jsonl", "episode/score"),
        ("metrics.jsonl", "episode/score"),
    ):
        path = Path(rundir) / name
        if path.exists() and path.stat().st_size:
            rows = [r for r in read_jsonl(path) if key in r and "step" in r]
            if rows:
                return sorted((int(r["step"]), r[key]) for r in rows)
    return []


def torch_scores(path):
    """Keyed on ``environment_steps``, the 1-based driver-record counter.

    That is the axis JAX logs as ``step``; ``action_steps`` counts control
    transitions only and runs one bucket behind at episode boundaries.
    """
    rows = read_jsonl(path)
    return sorted(
        (int(r["environment_steps"]), r["score"])
        for r in rows
        if r.get("type") == "train_episode" and "environment_steps" in r
    )


def scalar(rows, stepkey, key, predicate=lambda r: True):
    return sorted(
        (int(r[stepkey]), r[key])
        for r in rows
        if stepkey in r
        and key in r
        and isinstance(r[key], (int, float))
        and predicate(r)
    )


def bucket(series, width):
    out = defaultdict(list)
    for step, value in series:
        out[(step - 1) // width * width].append(value)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=int, default=16000)
    args = parser.parse_args()

    curves = {}
    for label, rundir in JAX_RUNS:
        series = jax_scores(rundir)
        if series:
            curves[label] = bucket(series, args.bucket)
    for label, path in TORCH_RUNS:
        if Path(path).exists() and Path(path).stat().st_size:
            series = torch_scores(path)
            if series:
                curves[label] = bucket(series, args.bucket)

    if curves:
        steps = sorted({b for c in curves.values() for b in c})
        names = list(curves)
        print(f"Episode return, mean over episodes in each {args.bucket}-step bucket")
        print(f"{'step':>8} " + " ".join(f"{n:>11}" for n in names))
        for step in steps:
            cells = []
            for name in names:
                values = curves[name].get(step)
                cells.append(
                    f"{statistics.mean(values):11.1f}" if values else f"{'-':>11}"
                )
            print(f"{step:>8} " + " ".join(cells))
        print()

    jax_rows, torch_rows = {}, {}
    for label, rundir in JAX_RUNS:
        path = Path(rundir) / "metrics.jsonl"
        if path.exists() and path.stat().st_size:
            jax_rows[label] = read_jsonl(path)
    for label, path in TORCH_RUNS:
        if Path(path).exists() and Path(path).stat().st_size:
            torch_rows[label] = read_jsonl(path)
    if not (jax_rows and torch_rows):
        return

    print("Reference train/* scalars: JAX mean | Torch mean | ratio, per bucket")
    print("(Torch side needs logger.diagnostics=true; ratios near 1.0 mean parity)")

    def collect(rows_by_label, stepkey, key, predicate):
        merged = defaultdict(list)
        for rows in rows_by_label.values():
            for step, value in scalar(rows, stepkey, key, predicate):
                merged[(step - 1) // args.bucket * args.bucket].append(value)
        return merged

    is_train = lambda r: r.get("type") == "train"  # noqa: E731
    buckets = None
    lines = []
    for jax_key, torch_key in PAIRS:
        jb = collect(jax_rows, "step", jax_key, lambda r: True)
        tb = collect(torch_rows, "environment_steps", torch_key, is_train)
        shared = sorted(set(jb) & set(tb))
        if buckets is None:
            buckets = shared
        cells = []
        for step in buckets:
            if step not in jb or step not in tb:
                cells.append(f"{'-':>24}")
                continue
            jm, tm = statistics.mean(jb[step]), statistics.mean(tb[step])
            ratio = tm / jm if abs(jm) > 1e-9 else float("nan")
            cells.append(f"{jm:8.3f} {tm:8.3f} {ratio:6.2f}")
        lines.append(f"{jax_key:>22} | " + " | ".join(cells))

    if buckets:
        header = f"{'metric':>22} | " + " | ".join(
            f"{step // 1000:>5}k jax    torch  ratio"[:24] for step in buckets
        )
        print(header)
        print("-" * len(header))
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
