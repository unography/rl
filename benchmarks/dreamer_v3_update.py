# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark the complete DreamerV3 DMC Walker learner update.

This uses the same callable as the training example, including backward, the
optimizer step, and the slow-critic update. CUDA synchronization occurs once at
each timing-window boundary rather than once per learner update.

Run from the repository root::

    python benchmarks/dreamer_v3_update.py \
        --compile-rssm none
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict


_REPO_ROOT = Path(__file__).parents[1]
_EXAMPLE_DIR = _REPO_ROOT / "sota-implementations/dreamer_v3"
sys.path.insert(0, str(_EXAMPLE_DIR))

import train as dreamer_train  # noqa: E402
from dreamer_v3_utils import latent_state_dim  # noqa: E402
from torchrl import timeit  # noqa: E402


CompileMode = Literal["none", "step", "scan"]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _load_config(
    compile_mode: CompileMode,
) -> DictConfig:
    cfg = OmegaConf.merge(
        OmegaConf.load(_EXAMPLE_DIR / "config.yaml"),
        OmegaConf.load(_EXAMPLE_DIR / "config_dmc_walker.yaml"),
    )
    cfg.optimization.compile_rssm = None if compile_mode == "none" else compile_mode
    return cfg


def _make_sample(
    cfg: DictConfig,
    *,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
) -> TensorDict:
    batch_size = cfg.replay_buffer.batch_size
    sequence_length = cfg.replay_buffer.seq_len
    state_dim = latent_state_dim(cfg)
    generator = torch.Generator(device=device).manual_seed(cfg.env.seed + 11)
    is_init = torch.zeros(
        batch_size,
        sequence_length,
        1,
        dtype=torch.bool,
        device=device,
    )
    # Exercise both sequence-start and intermediate-reset behavior.
    is_init[::2, 0] = True
    is_init[1::4, sequence_length // 2] = True
    done = torch.zeros_like(is_init)
    terminated = torch.zeros_like(is_init)
    done[1::4, sequence_length // 2 - 1] = True
    terminated[1::4, sequence_length // 2 - 1] = True
    return TensorDict(
        {
            "state": torch.zeros(
                batch_size,
                sequence_length,
                state_dim,
                device=device,
            ),
            "belief": torch.zeros(
                batch_size,
                sequence_length,
                cfg.networks.rnn_hidden_dim,
                device=device,
            ),
            "action": torch.randn(
                batch_size,
                sequence_length,
                action_dim,
                device=device,
                generator=generator,
            ).clamp_(-1, 1),
            "is_init": is_init,
            "next": TensorDict(
                {
                    "observation": torch.randn(
                        batch_size,
                        sequence_length,
                        obs_dim,
                        device=device,
                        generator=generator,
                    ),
                    "reward": torch.randn(
                        batch_size,
                        sequence_length,
                        1,
                        device=device,
                        generator=generator,
                    ),
                    "done": done,
                    "terminated": terminated,
                },
                [batch_size, sequence_length],
            ),
        },
        [batch_size, sequence_length],
    )


def _parameter_count(learner: dreamer_train._Learner) -> int:
    parameters = {
        id(parameter): parameter
        for group in learner.optimizer.param_groups
        for parameter in group["params"]
    }
    return sum(parameter.numel() for parameter in parameters.values())


def _run_benchmark(
    *,
    compile_mode: CompileMode,
    device: torch.device,
    warmup_updates: int,
    windows: int,
    updates_per_window: int,
    obs_dim: int = 24,
    action_dim: int = 6,
) -> dict[str, Any]:
    """Run the benchmark and return machine-readable timing and memory results."""
    cfg = _load_config(compile_mode)
    torch.manual_seed(cfg.env.seed)
    learner = dreamer_train._build_learner(cfg, device, obs_dim, action_dim)
    sample = _make_sample(
        cfg,
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    state_dim = latent_state_dim(cfg)
    use_bfloat16 = bool(cfg.optimization.mixed_precision) and device.type == "cuda"

    def update() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return dreamer_train._learner_update(
            sample,
            learner=learner,
            cfg=cfg,
            state_dim=state_dim,
            device=device,
            use_bfloat16=use_bfloat16,
        )

    _synchronize(device)
    with timeit("dreamer_v3_benchmark/first_update") as first_update_timer:
        result = update()
        _synchronize(device)
        first_update_seconds = first_update_timer.elapsed()
    for _ in range(warmup_updates):
        result = update()
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    window_seconds = []
    for _ in range(windows):
        _synchronize(device)
        with timeit("dreamer_v3_benchmark/window") as window_timer:
            for _ in range(updates_per_window):
                result = update()
            _synchronize(device)
            window_seconds.append(window_timer.elapsed())

    # Keep update outputs alive through the final synchronization, matching the
    # trainer's replay-context staging lifetime.
    del result
    milliseconds_per_update = [
        seconds * 1_000 / updates_per_window for seconds in window_seconds
    ]
    median_milliseconds = statistics.median(milliseconds_per_update)
    result_data: dict[str, Any] = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "compile_rssm": compile_mode,
        "mixed_precision": use_bfloat16,
        "batch_size": cfg.replay_buffer.batch_size,
        "sequence_length": cfg.replay_buffer.seq_len,
        "imagination_horizon": cfg.optimization.imagination_horizon,
        "belief_dim": cfg.networks.rnn_hidden_dim,
        "state_dim": state_dim,
        "parameter_count": _parameter_count(learner),
        "warmup_updates": warmup_updates,
        "windows": windows,
        "updates_per_window": updates_per_window,
        "first_update_seconds": first_update_seconds,
        "window_milliseconds_per_update": milliseconds_per_update,
        "median_milliseconds_per_update": median_milliseconds,
        "lower_quartile_milliseconds_per_update": _percentile(
            milliseconds_per_update, 0.25
        ),
        "upper_quartile_milliseconds_per_update": _percentile(
            milliseconds_per_update, 0.75
        ),
        "updates_per_second": 1_000 / median_milliseconds,
    }
    if device.type == "cuda":
        result_data.update(
            {
                "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
                "max_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                    device
                ),
            }
        )
    return result_data


def main() -> None:
    """Parse command-line options and report one benchmark run as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile-rssm",
        choices=("none", "step", "scan"),
        default="none",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--warmup-updates", type=int, default=4)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--updates-per-window", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_updates < 0:
        parser.error("--warmup-updates must be nonnegative")
    if args.windows <= 0:
        parser.error("--windows must be positive")
    if args.updates_per_window <= 0:
        parser.error("--updates-per-window must be positive")
    results = _run_benchmark(
        compile_mode=args.compile_rssm,
        device=torch.device(args.device),
        warmup_updates=args.warmup_updates,
        windows=args.windows,
        updates_per_window=args.updates_per_window,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
