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
import re
from pathlib import Path


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


def nearest(rows: list[dict[str, str]], step: int) -> dict[str, str]:
    return min(rows, key=lambda row: abs(int(row["step" if "step" in row else "env_step"]) - step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=REFERENCE / "a100_seed7_comparison.csv"
    )
    args = parser.parse_args()

    curve = read_csv(REFERENCE / "dmc_walker_walk_dreamerv3_mean.csv")
    losses = read_csv(REFERENCE / "jax_walker_walk_losses_a6000.csv")
    torch_rows = parse_log(args.parity_log, "torchrl-parity") + parse_log(
        args.baseline_log, "torchrl-main-control"
    )
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
        if row["implementation"] == "torchrl-parity":
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
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} ({len(output_rows)} checkpoints)")


if __name__ == "__main__":
    main()
