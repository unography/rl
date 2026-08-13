from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


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
    example = runpy.run_path(example_path, run_name="dreamer_v3_cuda_repro")

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config_dmc_walker")

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
        device=cfg.env.device,
    )
    try:
        batch = next(iter(collector))
    finally:
        collector.shutdown()

    actor_devices = sorted({str(parameter.device) for parameter in actor.parameters()})
    output = [
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device_count={torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        output.append(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    output.extend(
        [
            f"resolved_walker_env_device={cfg.env.device}",
            f"actor_parameter_devices={actor_devices}",
            f"collector_observation_device={batch['observation'].device}",
            f"collector_action_device={batch['action'].device}",
        ]
    )
    sys.stdout.write("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
