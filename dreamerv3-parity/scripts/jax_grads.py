"""Extract per-loss-key gradient norms from the JAX reference on a fixed batch.

Step 1 of the gradient-equivalence plan in ``PROVING-PARITY.md``: build the
reference agent with the walker config, dump its parameters so the Torch side
can load exactly the same weights, and evaluate ``report()`` with
``report_gradnorms`` enabled on a deterministic synthetic batch.

``replay_context`` is set to 0. The context mechanism is a data-pipeline
concern, checked separately; carrying it here would mean synthesising the
enc/dyn/dec entry tensors a previous training pass would have produced.

Usage::

    /home/ubuntu/dreamerv3/.venv/bin/python jax_grads.py <outdir> [batch] [length]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/home/ubuntu/dreamerv3")

import elements  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3.main import make_agent  # noqa: E402

outdir = elements.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/jaxgrads")
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 4
length = int(sys.argv[3]) if len(sys.argv) > 3 else 16

raw = yaml.YAML(typ="safe").load(
    open("/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/config.yaml").read()
)
raw["logdir"] = str(outdir)
raw["seed"] = 0
raw["batch_size"] = batch
raw["batch_length"] = length
raw["report_length"] = length
raw["replay_context"] = 0
raw["agent"]["report_gradnorms"] = True
raw["jax"]["prealloc"] = False
config = elements.Config(raw)
outdir.mkdir()

agent = make_agent(config)
print("obs_space:")
for k, v in agent.obs_space.items():
    print(f"  {k:16s} {v}")
print("act_space:")
for k, v in agent.act_space.items():
    print(f"  {k:16s} {v}")
# The agent installs a transfer guard; opt out for the dumps.
with jax.transfer_guard("allow"):
    model_params = {
        k: np.asarray(v) for k, v in agent.params.items() if not k.startswith("opt/")
    }
print(
    f"params: {len(agent.params)} tensors total, "
    f"{len(model_params)} model tensors, "
    f"{sum(int(v.size) for v in model_params.values())} model scalars"
)

# A deterministic batch, identical on both sides. Built from agent.spaces so
# the key set matches exactly -- report() asserts on it.
rng = np.random.default_rng(0)
B, T = batch, length
print("spaces:")
for k, v in agent.spaces.items():
    print(f"  {k:16s} {v}")


def make(space):
    shape = (B, T, *space.shape)
    if space.dtype == bool:
        return np.zeros(shape, bool)
    if np.issubdtype(space.dtype, np.integer):
        return np.zeros(shape, space.dtype)
    if space.low.size and np.isfinite(space.low).all():
        return rng.uniform(space.low, space.high, shape).astype(space.dtype)
    return rng.normal(size=shape).astype(space.dtype)


data = {k: make(v) for k, v in agent.spaces.items()}
data["is_first"][:, 0] = True

np.savez(str(outdir / "batch.npz"), **data)
np.savez(str(outdir / "params.npz"), **model_params)
print(f"wrote batch.npz and params.npz to {outdir}")

carry = agent.init_report(B)
data["seed"] = agent._seeds(0, agent.train_mirrored)
with jax.transfer_guard("allow"):
    _, metrics = agent.report(carry, data)
    print("\nall metric keys:")
    for k in sorted(metrics):
        print("   ", k)
    grads = {k: float(v) for k, v in metrics.items() if "gradnorm" in k}
    losses = {k: float(v) for k, v in metrics.items() if "loss" in k}
print("\ngradient global norms per loss key:")
for k in sorted(grads):
    print(f"  {k:24s} {grads[k]:.8e}")
print("\nloss values:")
for k in sorted(losses):
    print(f"  {k:24s} {losses[k]:.8e}")
import json

(outdir / "gradnorms.json").write(
    json.dumps({"grads": grads, "losses": losses}, indent=2)
)
