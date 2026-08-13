from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


def temporal_range(value: torch.Tensor) -> float:
    value = value.float().reshape(value.shape[0], -1)
    return (value.amax(0) - value.amin(0)).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="pytorch/rl checkout to probe (defaults to this checkout).",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    from torchrl.collectors import Collector
    from torchrl.data import Unbounded
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import TensorDictPrimer

    example_path = repo / "sota-implementations/dreamer_v3/dreamer_v3.py"
    config_dir = example_path.parent
    example = runpy.run_path(example_path, run_name="dreamer_v3_latent_repro")

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    cfg.env.max_episode_steps = 8
    torch.manual_seed(cfg.env.seed)

    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    env = TransformedEnv(
        example["make_env"](cfg, cfg.env.seed),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )
    actor = example["build_actor"](
        cfg=cfg,
        action_dim=env.action_spec.shape[-1],
    )
    collector = Collector(
        env,
        actor,
        frames_per_batch=32,
        total_frames=32,
        device="cpu",
    )
    try:
        batch = next(iter(collector))
    finally:
        collector.shutdown()

    is_init = batch["is_init"].squeeze(-1)
    evidence = {
        "observation_temporal_range": temporal_range(batch["observation"]),
        "state_abs_max": batch["state"].abs().max().item(),
        "belief_abs_max": batch["belief"].abs().max().item(),
        "loc_temporal_range": temporal_range(batch["loc"]),
        "scale_temporal_range": temporal_range(batch["scale"]),
        "sampled_action_temporal_range": temporal_range(batch["action"]),
        "episode_start_count": int(is_init.sum().item()),
        "episode_start_belief_abs_max": batch["belief"][is_init].abs().max().item(),
        "noninitial_belief_abs_max": batch["belief"][~is_init].abs().max().item(),
    }
    sys.stdout.write(
        "\n".join(f"{key}={value}" for key, value in evidence.items()) + "\n"
    )
    sys.stdout.flush()

    assert evidence["observation_temporal_range"] > 0
    assert (
        evidence["state_abs_max"] > 0
    ), "DreamerV3 collector never computed an observation-conditioned state"
    assert (
        evidence["belief_abs_max"] > 0
    ), "DreamerV3 collector never advanced its recurrent belief"
    assert (
        evidence["loc_temporal_range"] > 0
    ), "Actor distribution remained fixed while observations changed"
    assert evidence["episode_start_count"] >= 2
    assert (
        evidence["episode_start_belief_abs_max"] == 0
    ), "Recurrent belief was not reset at a new episode"
    assert evidence["noninitial_belief_abs_max"] > 0


if __name__ == "__main__":
    main()
