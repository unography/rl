"""Build the Torch DreamerV3 nets with the DMC walker config and dump init stats."""
from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

ROOT = Path("/home/ubuntu/rl")
SOTA = ROOT / "sota-implementations/dreamer_v3"

mod = runpy.run_path(str(SOTA / "dreamer_v3.py"))

base = OmegaConf.load(SOTA / "config.yaml")
walker = OmegaConf.load(SOTA / "config_dmc_walker.yaml")
walker.pop("defaults", None)
cfg = OmegaConf.merge(base, walker)

torch.manual_seed(cfg.env.seed)

obs_dim = 24
action_dim = 6

(world_model, prior_net, reward_net, reward_decoder, continuation_net,) = mod[
    "build_world_model"
](cfg=cfg, obs_dim=obs_dim, action_dim=action_dim)
actor_model = mod["build_actor"](cfg=cfg, action_dim=action_dim)
value_model = mod["build_value"](cfg=cfg)

groups = {
    "world_model": world_model,
    "actor": actor_model,
    "value": value_model,
}

out = {}
total = 0
for group_name, module in groups.items():
    for name, param in module.named_parameters():
        key = f"{group_name}/{name}"
        arr = param.detach().double()
        total += arr.numel()
        out[key] = {
            "shape": list(arr.shape),
            "size": int(arr.numel()),
            "mean": float(arr.mean()),
            "std": float(arr.std()) if arr.numel() > 1 else 0.0,
            "absmax": float(arr.abs().max()),
        }

print("TOTAL TRAINABLE PARAMS (world_model + actor + value):", total)
for key, stat in out.items():
    print(
        f"{key:<62s} {str(stat['shape']):<18s} "
        f"mean={stat['mean']:+.6f} std={stat['std']:.6f} absmax={stat['absmax']:.6f}"
    )

with open(
    "/tmp/claude-1001/-home-ubuntu-rl/"
    "c57965a0-f02e-4e35-9698-498a529f8120/scratchpad/torch_params.json",
    "w",
) as f:
    json.dump(out, f, indent=1)
