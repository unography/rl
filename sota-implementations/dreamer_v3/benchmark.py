# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run and aggregate DreamerV3 DMC Walker learning curves."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from torchrl._utils import logger as torchrl_logger


def _read_run(path: Path) -> dict:
    """Fold one run's jsonl into the fields the aggregation needs.

    The example writes a single jsonl per run: one ``train_episode`` record per
    finished episode, and a closing ``summary`` record with the totals.
    """
    episode_steps: list[int] = []
    episode_returns: list[float] = []
    summary: dict | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record["type"] == "train_episode":
            episode_steps.append(record["environment_steps"])
            episode_returns.append(record["score"])
        elif record["type"] == "summary":
            summary = record
    if summary is None:
        raise ValueError(
            f"{path} has no summary record; the run did not finish, so its "
            f"total step count is unknown."
        )
    return {
        "seed": summary["seed"],
        "total_environment_steps": summary["total_environment_steps"],
        "training_episode_steps": episode_steps,
        "training_episode_returns": episode_returns,
    }


def aggregate_runs(paths: list[Path], window_size: int = 50_000) -> dict:
    """Aggregate stochastic training returns into fixed-step median/IQR bands.

    Returns ``environment_steps`` and, aligned with it, ``median_return``,
    ``lower_quartile_return`` and ``upper_quartile_return`` across seeds, plus
    ``per_seed_window_median``, ``window_size`` and ``seeds``.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    runs = [_read_run(path) for path in paths]
    total_steps = min(run["total_environment_steps"] for run in runs)
    steps = list(range(window_size, total_steps + 1, window_size))
    if not steps:
        raise ValueError(f"Runs must contain at least {window_size} environment steps.")
    window_medians = []
    for run in runs:
        episode_steps = run["training_episode_steps"]
        episode_returns = run["training_episode_returns"]
        medians = []
        for stop in steps:
            start = stop - window_size
            values = [
                score
                for step, score in zip(episode_steps, episode_returns)
                if start < step <= stop
            ]
            if not values:
                raise ValueError(
                    f"Seed {run['seed']} has no completed training episode in "
                    f"the ({start}, {stop}] window."
                )
            medians.append(torch.tensor(values, dtype=torch.float64).median())
        window_medians.append(torch.stack(medians))
    returns = torch.stack(window_medians)
    return {
        "environment_steps": steps,
        "median_return": returns.median(0).values.tolist(),
        "lower_quartile_return": torch.quantile(returns, 0.25, dim=0).tolist(),
        "upper_quartile_return": torch.quantile(returns, 0.75, dim=0).tolist(),
        "per_seed_window_median": returns.tolist(),
        "window_size": window_size,
        "seeds": [run["seed"] for run in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("dmc_walker_runs"))
    parser.add_argument("--minimum-final-return", type=float, default=900.0)
    parser.add_argument("--window-size", type=int, default=50_000)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("dreamer_v3.py")
    metrics_paths = []
    for seed in args.seeds:
        metrics_jsonl_path = args.output_dir / f"seed_{seed}.jsonl"
        command = [
            sys.executable,
            str(script),
            "--config-name=config_dmc_walker",
            f"env.seed={seed}",
            f"logger.metrics_jsonl={metrics_jsonl_path}",
            "logger.output_plot=null",
            *args.overrides,
        ]
        torchrl_logger.info("Running DMC Walker seed %d", seed)
        subprocess.run(command, check=True)
        metrics_paths.append(metrics_jsonl_path)

    summary = aggregate_runs(metrics_paths, window_size=args.window_size)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    final_median = summary["median_return"][-1]
    if final_median < args.minimum_final_return:
        raise RuntimeError(
            "Final median DMC Walker return "
            f"{final_median:.1f} is below {args.minimum_final_return:.1f}."
        )
    torchrl_logger.info(
        "Saved DMC Walker median/IQR curve to %s (final median %.1f)",
        summary_path,
        final_median,
    )


if __name__ == "__main__":
    main()
