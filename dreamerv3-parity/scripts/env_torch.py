"""Roll the TorchRL DMC walker stack with the same fixed action sequence."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchrl.envs import (  # noqa: E402  # noqa: E402
    CatTensors,
    DoubleToFloat,
    InitTracker,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv  # noqa: E402
from torchrl.envs.transforms import ClipTransform  # noqa: E402

base_env = DMControlEnv("walker", "walk", device="cpu", _seed=12345)
env = TransformedEnv(base_env)
env.append_transform(
    CatTensors(in_keys=sorted(base_env.observation_spec.keys()), out_key="observation")
)
env.append_transform(ClipTransform(in_keys_inv=["action"], low=-1.0, high=1.0))
env.append_transform(DoubleToFloat())
env.append_transform(StepCounter(max_steps=1000))
env.append_transform(InitTracker())

print("obs keys sorted:", sorted(base_env.observation_spec.keys()))
print("action spec:", env.action_spec)

rng = np.random.default_rng(0)
actions = rng.uniform(-1.5, 1.5, size=(60, 6)).astype(np.float32)

td = env.reset()
obs0 = td["observation"].numpy()
print("obs0 (sorted cat) [:5]:", np.round(obs0[:5], 6).tolist())
print("obs0 len:", obs0.shape)

rewards = []
for i in range(60):
    td["action"] = torch.as_tensor(actions[i])
    td = env.step(td)
    rewards.append(float(td["next", "reward"].item()))
    td = td["next"].clone()
print("rewards:", np.round(rewards, 8).tolist())
print("sum:", float(np.sum(rewards)))
print("final obs [:5]:", np.round(td["observation"].numpy()[:5], 6).tolist())
