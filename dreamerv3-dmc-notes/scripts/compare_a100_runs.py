"""Compare local TorchRL walker logs with the committed JAX references.

The script understands both the parity branch's rich metric line and the
minimal main-control line. It writes one tidy CSV with the nearest JAX curve
point and, for parity metrics, the nearest JAX loss checkpoint.

Example:
    .venv/bin/python dreamerv3-dmc-notes/scripts/compare_a100_runs.py \
        --parity-log dreamerv3-dmc-notes/plots/a100_parity_seed7_10240.log \
        --baseline-log ../rl-baseline/dreamerv3-baseline-notes/a100_baseline_seed7_10240.log
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "dreamerv3-dmc-notes" / "reference"
LINE_RE = re.compile(r"env_step=\s*(?P<step>\d+)\].*?eval_reward=(?P<return>[-+\d.]+)(?P<metrics>.*)")
METRIC_RE = re.compile(r"(?P<name>kl|dyn|rep|con|reco|reward|policy|value|repval|actor)=(?P<value>[-+\d.]+)")


def parse_log(path: Path, implementation: str) -> list[dict[str, str | float | int]]:
    rows = []
    for line in path.read_text().splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        row: dict[str, str | float | int] = {
            "implementation": implementation,
            "run": path.stem,
            "env_step": int(match["step"]),
            "eval_return": float(match["return"]),
        }
        row.update(
            (metric["name"], float(metric["value"]))
            for metric in METRIC_RE.finditer(match["metrics"])
        )
        rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_jax_metrics(path: Path) -> list[dict[str, str | float | int]]:
    rows = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if "train/loss/dyn" not in raw:
            continue
        rows.append(
            {
                "env_step": int(raw["step"]),
                "dyn": raw["train/loss/dyn"],
                "rep": raw["train/loss/rep"],
                "rew": raw["train/loss/rew"],
                "con": raw["train/loss/con"],
                "policy": raw["train/loss/policy"],
                "value": raw["train/loss/value"],
                "repval": raw["train/loss/repval"],
                "recon_sum": sum(
                    value
                    for key, value in raw.items()
                    if key.startswith("train/loss/")
                    and key
                    not in {
                        "train/loss/dyn",
                        "train/loss/rep",
                        "train/loss/rew",
                        "train/loss/con",
                        "train/loss/policy",
                        "train/loss/value",
                        "train/loss/repval",
                    }
                ),
            }
        )
    return rows


def nearest(rows: list[dict[str, str]], step: int) -> dict[str, str]:
    return min(rows, key=lambda row: abs(int(row["step" if "step" in row else "env_step"]) - step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-log", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-log", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--v3off-log",
        type=Path,
        nargs="+",
        default=[],
        help="measured config_dmc_v3off logs to include as an ablation arm",
    )
    parser.add_argument(
        "--jax-metrics",
        type=Path,
        help="optional JAX metrics.jsonl; defaults to the committed loss CSV",
    )
    parser.add_argument(
        "--output", type=Path, default=REFERENCE / "a100_seed7_comparison.csv"
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()

    curve = read_csv(REFERENCE / "dmc_walker_walk_dreamerv3_mean.csv")
    losses = (
        read_jax_metrics(args.jax_metrics)
        if args.jax_metrics
        else read_csv(REFERENCE / "jax_walker_walk_losses_a6000.csv")
    )
    torch_rows = [
        row
        for path in args.parity_log
        for row in parse_log(path, "torchrl-parity")
    ] + [
        row
        for path in args.baseline_log
        for row in parse_log(path, "torchrl-main-control")
    ] + [
        row
        for path in args.v3off_log
        for row in parse_log(path, "torchrl-v3off")
    ]
    output_rows = []
    for row in torch_rows:
        loss_row = nearest(losses, int(row["env_step"]))
        result = dict(row)
        # The published curve has no pre-10k checkpoint. Do not misleadingly
        # compare a 1k shakeout return to the 10k JAX distribution.
        if int(row["env_step"]) >= int(curve[0]["step"]):
            curve_row = nearest(curve, int(row["env_step"]))
            result.update(
                jax_curve_step=int(curve_row["step"]),
                jax_return_mean=float(curve_row["mean"]),
                jax_return_min=float(curve_row["min"]),
                jax_return_max=float(curve_row["max"]),
                return_abs_error=abs(
                    float(row["eval_return"]) - float(curve_row["mean"])
                ),
                return_in_jax_band=(
                    float(curve_row["min"])
                    <= float(row["eval_return"])
                    <= float(curve_row["max"])
                ),
            )
        if row["implementation"] in ("torchrl-parity", "torchrl-v3off"):
            result["jax_loss_step"] = int(loss_row["env_step"])
            for torch_name, jax_name in (
                ("dyn", "dyn"),
                ("rep", "rep"),
                ("reco", "recon_sum"),
                ("reward", "rew"),
                ("con", "con"),
                ("policy", "policy"),
                ("value", "value"),
                ("repval", "repval"),
            ):
                if torch_name in row:
                    result[f"{torch_name}_jax"] = float(loss_row[jax_name])
                    result[f"{torch_name}_abs_error"] = abs(
                        float(row[torch_name]) - float(loss_row[jax_name])
                    )
        output_rows.append(result)

    fieldnames = sorted({key for row in output_rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} ({len(output_rows)} checkpoints)")

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in torch_rows:
        grouped[(str(row["implementation"]), int(row["env_step"]))].append(
            float(row["eval_return"])
        )
    summary_rows = []
    for (implementation, step), values in sorted(grouped.items()):
        result = {
            "implementation": implementation,
            "env_step": step,
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if step >= int(curve[0]["step"]):
            curve_row = nearest(curve, step)
            result.update(
                jax_step=int(curve_row["step"]),
                jax_mean=float(curve_row["mean"]),
                jax_min=float(curve_row["min"]),
                jax_max=float(curve_row["max"]),
                median_abs_error=abs(
                    statistics.median(values) - float(curve_row["mean"])
                ),
                seeds_in_jax_band=sum(
                    float(curve_row["min"]) <= value <= float(curve_row["max"])
                    for value in values
                ),
            )
        summary_rows.append(result)

    summary_output = args.summary_output or args.output.with_name(
        f"{args.output.stem}_summary.csv"
    )
    summary_fields = sorted({key for row in summary_rows for key in row})
    with summary_output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=summary_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_output} ({len(summary_rows)} summary rows)")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "torchrl-parity": "#7570b3",
        "torchrl-main-control": "#d95f02",
        "torchrl-v3off": "#e7298a",
    }
    labels = {
        "torchrl-parity": "TorchRL parity",
        "torchrl-main-control": "TorchRL main",
        "torchrl-v3off": "TorchRL V3-off",
    }
    measured_steps = [int(row["env_step"]) for row in summary_rows]
    measured_max = max(measured_steps, default=None)
    for implementation in colors:
        rows = [row for row in summary_rows if row["implementation"] == implementation]
        if not rows:
            continue
        ax.fill_between(
            [row["env_step"] for row in rows],
            [row["min"] for row in rows],
            [row["max"] for row in rows],
            alpha=0.15,
            color=colors[implementation],
        )
        ax.plot(
            [row["env_step"] for row in rows],
            [row["median"] for row in rows],
            marker="o",
            color=colors[implementation],
            label=(
                f"{labels[implementation]} (1 seed)"
                if max(int(row["n"]) for row in rows) == 1
                else f"{labels[implementation]} median"
            ),
        )
    # Crop before plotting, rather than only setting xlim afterward: Matplotlib
    # otherwise autoscales y from the hidden 490k portion of the JAX curve and
    # compresses a matched 51k comparison.
    plot_curve = (
        [row for row in curve if int(row["step"]) <= measured_max * 1.05]
        if measured_max is not None
        else curve
    )
    jax_steps = [int(row["step"]) for row in plot_curve]
    jax_mean = [float(row["mean"]) for row in plot_curve]
    jax_min = [float(row["min"]) for row in plot_curve]
    jax_max = [float(row["max"]) for row in plot_curve]
    ax.fill_between(jax_steps, jax_min, jax_max, alpha=0.15, color="#1b9e77")
    ax.plot(
        jax_steps,
        jax_mean,
        color="#1b9e77",
        linewidth=2,
        label="JAX DreamerV3 mean/range (5 seeds)",
    )
    ax.set(
        xlabel="environment steps",
        ylabel="evaluation return",
        title="DMC walker_walk: JAX vs TorchRL implementations",
    )
    # Keep the comparison readable when the committed JAX reference extends far
    # beyond a shorter local evidence budget (for example, 51.2k vs 490k).
    # The full JAX rows remain in the source CSV; the plot shows the matched
    # horizon covered by at least one measured TorchRL arm.
    if measured_max is not None:
        ax.set_xlim(0, measured_max * 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_output = args.plot_output or args.output.with_name(
        f"{args.output.stem}_returns.png"
    )
    fig.savefig(plot_output, dpi=130)
    print(f"wrote {plot_output}")


if __name__ == "__main__":
    main()
