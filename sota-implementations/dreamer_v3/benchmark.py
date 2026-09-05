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

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase

from torchrl._utils import logger as torchrl_logger

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "config_dmc_walker"
# Set for each run below. A caller override would break the seed loop.
_RESERVED_OVERRIDES = ("env.seed", "logger.metrics_jsonl")


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
    episodes: TensorDictBase, names: Sequence[str], action_budget: int | None
) -> tuple[torch.Tensor, int]:
    """Return the success rate of each achievement, and the episodes counted.

    An episode counts when it ends within ``action_budget`` environment
    actions; ``None`` counts every episode. An achievement succeeds in an
    episode when its count is positive.

    Args:
        episodes (TensorDictBase): Training episodes with ``action_steps`` and
            an ``achievements`` vector.
        names (Sequence[str]): The achievement names, in output order.
        action_budget (int or None): The action budget of the evaluation.
    """
    achievements = episodes.get("achievements")
    if achievements.shape[-1] != len(names):
        raise ValueError(
            f"Expected {len(names)} achievements, got {achievements.shape[-1]}."
        )
    mask = torch.ones(episodes.batch_size, dtype=torch.bool)
    if action_budget is not None:
        mask &= episodes.get("action_steps") <= action_budget
    counted = int(mask.sum())
    if not counted:
        raise ValueError("No completed Crafter episode is inside the action budget.")
    rates = achievements[mask].gt(0).double().mean(0).mul(100)
    return rates, counted


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


def _read_run(path: Path) -> TensorDict:
    """Fold one run's jsonl into the fields the aggregation needs."""
    episodes: list[TensorDict] = []
    episode_achievement_names: tuple[str, ...] | None = None
    summary: dict | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record["type"] == "train_episode":
            row = TensorDict(
                {
                    "environment_steps": torch.tensor(
                        record["environment_steps"], dtype=torch.int64
                    ),
                    "action_steps": torch.tensor(
                        record["action_steps"], dtype=torch.int64
                    ),
                    "episode_return": torch.tensor(
                        record["episode_return"], dtype=torch.float64
                    ),
                },
                [],
            )
            achievements = record.get("achievements")
            if achievements is not None:
                names = tuple(achievements)
                if (
                    episode_achievement_names is not None
                    and names != episode_achievement_names
                ):
                    raise ValueError(
                        f"{path} changes achievement names between episodes."
                    )
                episode_achievement_names = names
                row.set(
                    "achievements",
                    torch.tensor(
                        [achievements[name] for name in names], dtype=torch.int64
                    ),
                )
            episodes.append(row)
        elif record["type"] == "summary":
            summary = record
    if summary is None:
        raise ValueError(
            f"{path} has no summary record; the run did not finish, so its "
            f"total step count is unknown."
        )
    if "total_action_steps" not in summary:
        raise ValueError(f"{path} does not record total_action_steps.")
    achievement_names = summary.get("achievement_names")
    achievement_names = (
        tuple(achievement_names) if achievement_names is not None else None
    )
    if episode_achievement_names != achievement_names and episodes:
        raise ValueError(
            f"{path} has different achievement names in episodes and its summary."
        )
    if episodes:
        episode_data = torch.stack(episodes)
    else:
        episode_data = TensorDict(
            {
                "environment_steps": torch.empty(0, dtype=torch.int64),
                "action_steps": torch.empty(0, dtype=torch.int64),
                "episode_return": torch.empty(0, dtype=torch.float64),
            },
            [0],
        )
        if achievement_names is not None:
            episode_data.set(
                "achievements",
                torch.empty((0, len(achievement_names)), dtype=torch.int64),
            )
    run = TensorDict(
        {
            "seed": torch.tensor(summary["seed"], dtype=torch.int64),
            "total_environment_steps": torch.tensor(
                summary["total_environment_steps"], dtype=torch.int64
            ),
            "total_action_steps": torch.tensor(
                summary["total_action_steps"], dtype=torch.int64
            ),
            "training_episodes": episode_data,
        },
        [],
    )
    run.set_non_tensor("achievement_names", achievement_names)
    return run


def _crafter_summary(
    runs: Sequence[TensorDictBase], action_budget: int | None
) -> dict | None:
    """Aggregate the achievement records of Crafter runs, or None for others."""
    names = runs[0].get_non_tensor("achievement_names")
    if names is None:
        if any(run.get_non_tensor("achievement_names") is not None for run in runs[1:]):
            raise ValueError("Runs disagree on whether they contain Crafter metrics.")
        return None
    if any(run.get_non_tensor("achievement_names") != names for run in runs[1:]):
        raise ValueError("Crafter runs have different achievement names.")
    seed_results = []
    for run in runs:
        if action_budget is not None and int(run["total_action_steps"]) < action_budget:
            raise ValueError(
                f"Seed {int(run['seed'])} stopped at {int(run['total_action_steps'])} "
                f"actions, below the {action_budget}-action Crafter budget."
            )
        rates, counted = achievement_success_rates(
            run.get("training_episodes"), names, action_budget
        )
        seed_results.append(
            TensorDict(
                {
                    "seed": run["seed"],
                    "episodes": torch.tensor(counted, dtype=torch.int64),
                    "success_rates": rates,
                    "crafter_score": torch.tensor(
                        crafter_score(rates.tolist()), dtype=torch.float64
                    ),
                },
                [],
            )
        )
    results = torch.stack(seed_results)
    scores = results["crafter_score"]
    return {
        "action_budget": action_budget,
        "achievement_names": list(names),
        "episodes_per_seed": results["episodes"].tolist(),
        "success_rates_per_seed": results["success_rates"].tolist(),
        "crafter_score_per_seed": scores.tolist(),
        "crafter_score_median": float(torch.quantile(scores, 0.5)),
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
    if not paths:
        raise ValueError("At least one completed run is required.")
    runs = [_read_run(path) for path in paths]
    seeds = torch.stack([run["seed"] for run in runs])
    if seeds.unique().numel() != seeds.numel():
        raise ValueError(f"Run seeds must be unique, got {seeds.tolist()}.")
    total_steps = min(int(run["total_environment_steps"]) for run in runs)
    steps = list(range(window_size, total_steps + 1, window_size))
    if not steps:
        raise ValueError(f"Runs must contain at least {window_size} environment steps.")
    seed_curves = []
    for run in runs:
        episodes = run.get("training_episodes")
        episode_steps = episodes["environment_steps"]
        episode_returns = episodes["episode_return"]
        medians = []
        for stop in steps:
            start = stop - window_size
            values = episode_returns[(start < episode_steps) & (episode_steps <= stop)]
            if not values.numel():
                raise ValueError(
                    f"Seed {int(run['seed'])} has no completed training episode in "
                    f"the ({start}, {stop}] window."
                )
            medians.append(torch.quantile(values, 0.5))
        seed_curves.append(
            TensorDict({"seed": run["seed"], "window_median": torch.stack(medians)}, [])
        )
    curves = torch.stack(seed_curves)
    window_medians = curves["window_median"]
    summary = {
        "environment_steps": steps,
        "median_return": torch.quantile(window_medians, 0.5, dim=0).tolist(),
        "lower_quartile_return": torch.quantile(window_medians, 0.25, dim=0).tolist(),
        "upper_quartile_return": torch.quantile(window_medians, 0.75, dim=0).tolist(),
        "per_seed_window_median": window_medians.tolist(),
        "window_size": window_size,
        "seeds": curves["seed"].tolist(),
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
