"""Instantiate the reference DreamerV3 agent and dump initial parameter stats."""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ubuntu/dreamerv3")

import elements  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3.main import make_agent  # noqa: E402

cfg_path = "/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/config.yaml"
raw = yaml.YAML(typ="safe").load(open(cfg_path).read())
raw["logdir"] = "/tmp/jaxdump"
config = elements.Config(raw)

agent = make_agent(config)

import jax  # noqa: E402

with jax.transfer_guard("allow"):
    host_params = {k: np.asarray(jax.device_get(v)) for k, v in agent.params.items()}

out = {}
total = 0
for key, value in sorted(host_params.items()):
    arr = np.asarray(value, np.float64)
    total += arr.size
    out[key] = {
        "shape": list(arr.shape),
        "size": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "absmax": float(np.abs(arr).max()),
        "dtype": str(np.asarray(value).dtype),
    }
print("TOTAL PARAMS", total)
with open(
    "/tmp/claude-1001/-home-ubuntu-rl/"
    "c57965a0-f02e-4e35-9698-498a529f8120/scratchpad/jax_params.json",
    "w",
) as f:
    json.dump(out, f, indent=1)
for key, stat in out.items():
    print(
        f"{key:<48s} {str(stat['shape']):<18s} "
        f"mean={stat['mean']:+.6f} std={stat['std']:.6f} "
        f"absmax={stat['absmax']:.6f}"
    )
