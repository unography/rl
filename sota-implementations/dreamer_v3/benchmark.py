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
from collections.abc import Sequence
from pathlib import Path

import torch
from omegaconf import OmegaConf

from torchrl._utils import logger as torchrl_logger

CONFIG_PATH = Path(__file__).with_name("config_dmc_walker.yaml")


def benchmark_settings(overrides: Sequence[str] = ()) -> dict:
    """Read the ``benchmark`` block of the walker preset.

    The block holds the reproduction protocol: the seeds to run, the window the
    training returns are aggregated over, and the final-window median the run
    must reach. Hydra overrides passed through to the example are applied here
    too, so ``benchmark.window_size=1000`` means the same thing on both sides.
    """
    config = OmegaConf.load(CONFIG_PATH)
    dotlist = [override for override in overrides if override.startswith("benchmark.")]
    if dotlist:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))
    return OmegaConf.to_container(config.benchmark, resolve=True)


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


def aggregate_runs(paths: list[Path], window_size: int) -> dict:
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
    parser.add_argument("--output-dir", type=Path, default=Path("dmc_walker_runs"))
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides for the example. Those under benchmark.* also "
            f"override the {CONFIG_PATH.name} block this script reads."
        ),
    )
    args = parser.parse_args()

    settings = benchmark_settings(args.overrides)
    seeds = settings["seeds"]
    window_size = settings["window_size"]
    minimum_final_return = settings["minimum_final_median_return"]

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("dreamer_v3.py")
    metrics_paths = []
    for seed in seeds:
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

    summary = aggregate_runs(metrics_paths, window_size=window_size)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    final_median = summary["median_return"][-1]
    if final_median < minimum_final_return:
        raise RuntimeError(
            "Final median DMC Walker return "
            f"{final_median:.1f} is below {minimum_final_return:.1f}."
        )
    torchrl_logger.info(
        "Saved DMC Walker median/IQR curve to %s (final median %.1f)",
        summary_path,
        final_median,
    )


if __name__ == "__main__":
    main()
