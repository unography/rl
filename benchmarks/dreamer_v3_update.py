# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark the complete DreamerV3 DMC Walker learner update."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


_ROOT = Path(__file__).parents[1]
_EXAMPLE = _ROOT / "sota-implementations/dreamer_v3"
sys.path.insert(0, str(_EXAMPLE))

import train as dreamer_train  # noqa: E402
from dreamer_v3_benchmark import make_replay_sample  # noqa: E402
from dreamer_v3_utils import latent_state_dim  # noqa: E402


def load_config(*, compiled: bool, mixed_precision: bool) -> DictConfig:
    """Load the DMC Walker configuration."""
    configs = []
    for name in ("config.yaml", "config_dmc_walker.yaml"):
        config = OmegaConf.load(_EXAMPLE / name)
        config.pop("defaults", None)
        configs.append(config)
    cfg = OmegaConf.merge(*configs)
    cfg.optimization.compile_rssm = None
    cfg.optimization.compile_learner = compiled
    cfg.optimization.mixed_precision = mixed_precision
    return cfg


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    *,
    compiled: bool,
    device: torch.device,
    precision: str,
    warmup_updates: int,
    windows: int,
    updates_per_window: int,
) -> dict[str, object]:
    """Measure repeated windows of complete learner updates."""
    use_bfloat16 = precision == "bfloat16" and device.type == "cuda"
    cfg = load_config(compiled=compiled, mixed_precision=use_bfloat16)
    cfg.optimization.device = str(device)
    torch.manual_seed(cfg.env.seed)
    real_env = dreamer_train.make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    real_env.close()
    learner = dreamer_train._build_learner(cfg, device, obs_dim, action_dim)
    sample = make_replay_sample(
        cfg, device=device, obs_dim=obs_dim, action_dim=action_dim
    )

    def update() -> None:
        dreamer_train._learner_update(
            sample,
            learner=learner,
            cfg=cfg,
            state_dim=latent_state_dim(cfg),
            device=device,
            use_bfloat16=use_bfloat16,
        )

    _synchronize(device)
    first_start = time.perf_counter()
    update()
    _synchronize(device)
    first_update = time.perf_counter() - first_start
    for _ in range(warmup_updates):
        update()
    _synchronize(device)

    milliseconds = []
    for _ in range(windows):
        start = time.perf_counter()
        for _ in range(updates_per_window):
            update()
        _synchronize(device)
        milliseconds.append((time.perf_counter() - start) * 1_000 / updates_per_window)

    median = statistics.median(milliseconds)
    parameters = {
        id(parameter): parameter
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "mode": "compiled" if compiled else "eager",
        "precision": "bfloat16" if use_bfloat16 else "float32",
        "precision_requested": precision,
        "batch_size": cfg.replay_buffer.batch_size,
        "sequence_length": cfg.replay_buffer.seq_len,
        "imagination_horizon": cfg.optimization.imagination_horizon,
        "parameter_count": sum(parameter.numel() for parameter in parameters.values()),
        "first_update_seconds": first_update,
        "warmup_updates": warmup_updates,
        "windows": windows,
        "updates_per_window": updates_per_window,
        "milliseconds_per_update": milliseconds,
        "median_milliseconds_per_update": median,
        "range_milliseconds_per_update": [min(milliseconds), max(milliseconds)],
        "updates_per_second": 1_000 / median,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--updates-per-window", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_updates < 0 or args.windows < 1 or args.updates_per_window < 1:
        parser.error("update counts must be positive, except warmup may be zero")
    result = run_benchmark(
        compiled=args.mode == "compiled",
        device=torch.device(args.device),
        precision=args.precision,
        warmup_updates=args.warmup_updates,
        windows=args.windows,
        updates_per_window=args.updates_per_window,
    )
    output = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
