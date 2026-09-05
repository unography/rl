# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run and aggregate multi-seed DreamerV3 learning curves of one preset."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from torchrl._utils import logger as torchrl_logger

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "config_dmc_walker"
# Set for each run below. A caller override would break the seed loop.
_RESERVED_OVERRIDES = ("env.seed", "logger.metrics_jsonl")


def _quantile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated ``q`` quantile of ``values``."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def crafter_score(success_rates: Sequence[float]) -> float:
    """Return the official Crafter score of per-achievement success rates.

    The score is the geometric mean of ``1 + s_i`` minus one, with ``s_i``
    the percentage, from 0 to 100, of episodes that unlock achievement
    ``i``: ``exp(mean(log(1 + s_i))) - 1``. It ranges from 0 to 100 and is
    not the episode return.

    Args:
        success_rates (Sequence[float]): One percentage per achievement.
    """
    if not success_rates:
        raise ValueError("The Crafter score needs at least one achievement.")
    if any(rate < 0 or rate > 100 for rate in success_rates):
        raise ValueError(f"Success rates are percentages, got {success_rates}.")
    mean_log = sum(math.log(1 + rate) for rate in success_rates) / len(success_rates)
    return math.exp(mean_log) - 1


def achievement_success_rates(
    episodes: Sequence[dict], names: Sequence[str], action_budget: int | None
) -> tuple[list[float], int]:
    """Return the success rate of each achievement, and the episodes counted.

    An episode counts when it ends within ``action_budget`` environment
    actions; ``None`` counts every episode. An achievement succeeds in an
    episode when its count is positive.

    Args:
        episodes (Sequence[dict]): Training-episode records with
            ``action_steps`` and an ``achievements`` mapping.
        names (Sequence[str]): The achievement names, in output order.
        action_budget (int or None): The action budget of the evaluation.
    """
    counted = [
        episode
        for episode in episodes
        if action_budget is None or episode["action_steps"] <= action_budget
    ]
    if not counted:
        return [0.0] * len(names), 0
    rates = [
        100.0
        * sum(episode["achievements"][name] > 0 for episode in counted)
        / len(counted)
        for name in names
    ]
    return rates, len(counted)


def _override_key(override: str) -> str:
    """Return the config key a Hydra override addresses."""
    return override.split("=", 1)[0].lstrip("+~").strip()


def effective_config(
    config_name: str = DEFAULT_CONFIG_NAME, overrides: Sequence[str] = ()
) -> DictConfig:
    """Compose a preset as Hydra will, with the caller's overrides."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def episode_cycle(config: DictConfig) -> int:
    """Return the environment steps between episode-completion bursts.

    Workers run to the same time limit, so episodes finish one episode apart.
    """
    num_envs = config.collector.num_envs
    if config.collector.count_reset_records:
        # The driver axis also counts the reset record of each episode.
        return (config.env.max_episode_steps + 1) * num_envs
    return config.env.max_episode_steps * num_envs


def validate_window_size(
    window_size: int,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: Sequence[str] = (),
) -> None:
    """Refuse, before the runs start, a window too narrow for one episode."""
    config = effective_config(config_name, overrides)
    cycle = episode_cycle(config)
    if window_size < cycle:
        raise ValueError(
            f"benchmark.window_size={window_size} is below the {cycle}-step "
            f"episode cycle ({config.collector.num_envs} envs x "
            f"{config.env.max_episode_steps}-step episodes), so most windows "
            "would hold no completed episode. Shorten collector.total_frames "
            "to run a smaller ablation, and leave the window alone."
        )


def benchmark_settings(
    config_name: str = DEFAULT_CONFIG_NAME, overrides: Sequence[str] = ()
) -> dict:
    """Read the ``benchmark`` block of a preset, overrides applied.

    A missing or null ``minimum_final_median_return`` disables the threshold.
    """
    config = effective_config(config_name, overrides)
    if "benchmark" not in config:
        raise ValueError(f"{config_name} has no benchmark block.")
    settings = OmegaConf.to_container(config.benchmark, resolve=True)
    settings.setdefault("minimum_final_median_return", None)
    settings.setdefault("crafter_action_budget", None)
    return settings


def reject_reserved_overrides(overrides: Sequence[str]) -> None:
    """Refuse overrides of the keys this script sets per run.

    Hydra takes the last of a duplicated key, so ``env.seed`` would train one
    trajectory and report it under every seed's name.
    """
    for override in overrides:
        key = _override_key(override)
        if key in _RESERVED_OVERRIDES:
            raise ValueError(
                f"{key} is set per run by this script and cannot be overridden. "
                "Use benchmark.seeds to choose the seeds and --output-dir to "
                "choose where their metrics land."
            )


def _read_run(path: Path) -> dict:
    """Fold one run's jsonl into the fields the aggregation needs."""
    episodes: list[dict] = []
    summary: dict | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record["type"] == "train_episode":
            episodes.append(record)
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
        "achievement_names": summary.get("achievement_names"),
        "training_episodes": episodes,
        "training_episode_steps": [e["environment_steps"] for e in episodes],
        "training_episode_returns": [e["episode_return"] for e in episodes],
    }


def _crafter_summary(runs: Sequence[dict], action_budget: int | None) -> dict | None:
    """Aggregate the achievement records of Crafter runs, or None for others."""
    names = runs[0]["achievement_names"]
    if names is None:
        return None
    rates_per_seed = []
    episodes_per_seed = []
    for run in runs:
        rates, counted = achievement_success_rates(
            run["training_episodes"], names, action_budget
        )
        rates_per_seed.append(rates)
        episodes_per_seed.append(counted)
    scores = [crafter_score(rates) for rates in rates_per_seed]
    return {
        "action_budget": action_budget,
        "achievement_names": list(names),
        "episodes_per_seed": episodes_per_seed,
        "success_rates_per_seed": rates_per_seed,
        "crafter_score_per_seed": scores,
        "crafter_score_median": _quantile(scores, 0.5),
    }


def aggregate_runs(
    paths: Sequence[Path],
    window_size: int,
    *,
    crafter_action_budget: int | None = None,
    **manifest: object,
) -> dict:
    """Aggregate stochastic training returns into fixed-step median/IQR bands.

    Returns ``environment_steps`` with ``median_return``,
    ``lower_quartile_return``, ``upper_quartile_return`` and
    ``per_seed_window_median`` aligned to it, plus ``window_size``, ``seeds``
    and the ``manifest`` entries, such as the config name and the task.
    Runs that record achievements also get a ``crafter`` entry: the success
    rate of each achievement and the Crafter score of each seed, from the
    episodes that end within ``crafter_action_budget`` actions.
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
            medians.append(_quantile(values, 0.5))
        window_medians.append(medians)
    across_seeds = list(zip(*window_medians))
    summary = {
        "environment_steps": steps,
        "median_return": [_quantile(window, 0.5) for window in across_seeds],
        "lower_quartile_return": [_quantile(window, 0.25) for window in across_seeds],
        "upper_quartile_return": [_quantile(window, 0.75) for window in across_seeds],
        "per_seed_window_median": window_medians,
        "window_size": window_size,
        "seeds": [run["seed"] for run in runs],
        **manifest,
    }
    crafter = _crafter_summary(runs, crafter_action_budget)
    if crafter is not None:
        summary["crafter"] = crafter
    return summary


def default_output_dir(config_name: str) -> Path:
    """Return ``<preset>_runs`` for ``config_<preset>``."""
    return Path(config_name.removeprefix("config_") + "_runs")


def task_name(config: DictConfig) -> str:
    """Return the environment name with its task, if the backend has one."""
    return (
        f"{config.env.name}/{config.env.task}" if config.env.task else config.env.name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="The preset in this directory to run, without the .yaml suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the metrics land. Defaults to <preset>_runs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides for the example. Those under benchmark.* also "
            "override the preset block this script reads."
        ),
    )
    args = parser.parse_args()
    config_name = args.config_name
    output_dir = args.output_dir or default_output_dir(config_name)

    reject_reserved_overrides(args.overrides)
    settings = benchmark_settings(config_name, args.overrides)
    seeds = settings["seeds"]
    window_size = settings["window_size"]
    minimum_final_return = settings["minimum_final_median_return"]
    validate_window_size(window_size, config_name, args.overrides)
    task = task_name(effective_config(config_name, args.overrides))

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train.py")
    metrics_paths = []
    for seed in seeds:
        metrics_jsonl_path = output_dir / f"seed_{seed}.jsonl"
        command = [
            sys.executable,
            str(script),
            f"--config-name={config_name}",
            f"env.seed={seed}",
            f"logger.metrics_jsonl={metrics_jsonl_path}",
            "logger.output_plot=null",
            *args.overrides,
        ]
        torchrl_logger.info("Running %s (%s) seed %d", config_name, task, seed)
        subprocess.run(command, check=True)
        metrics_paths.append(metrics_jsonl_path)

    summary = aggregate_runs(
        metrics_paths,
        window_size=window_size,
        crafter_action_budget=settings["crafter_action_budget"],
        config_name=config_name,
        task=task,
        minimum_final_median_return=minimum_final_return,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    final_median = summary["median_return"][-1]
    if minimum_final_return is not None and final_median < minimum_final_return:
        raise RuntimeError(
            f"Final median {task} return {final_median:.1f} is below "
            f"{minimum_final_return:.1f}."
        )
    torchrl_logger.info(
        "Saved %s median/IQR curve to %s (final median %.1f, threshold %s)",
        task,
        summary_path,
        final_median,
        "none" if minimum_final_return is None else f"{minimum_final_return:.1f}",
    )
    if "crafter" in summary:
        torchrl_logger.info(
            "Crafter score per seed %s (median %.2f) over %s episodes within "
            "%s actions",
            summary["crafter"]["crafter_score_per_seed"],
            summary["crafter"]["crafter_score_median"],
            summary["crafter"]["episodes_per_seed"],
            summary["crafter"]["action_budget"],
        )


if __name__ == "__main__":
    main()
