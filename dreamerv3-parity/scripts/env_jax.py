"""Roll the reference DMC walker wrapper with a fixed action sequence."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ubuntu/dreamerv3")

import elements  # noqa: E402

import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402
from dm_control import suite  # noqa: E402
from dreamerv3.main import wrap_env  # noqa: E402
from embodied.envs.dmc import DMC  # noqa: E402

raw = yaml.YAML(typ="safe").load(
    open("/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/config.yaml").read()
)
raw["logdir"] = "/tmp/jaxdump"
config = elements.Config(raw)

dmenv = suite.load("walker", "walk", task_kwargs={"random": 12345})
env = DMC(dmenv, repeat=1, proprio=True, image=False)
env = wrap_env(env, config)

rng = np.random.default_rng(0)
actions = rng.uniform(-1.5, 1.5, size=(60, 6)).astype(np.float32)

obs = env.step({"action": np.zeros(6, np.float32), "reset": True})
print("keys:", sorted(k for k in obs if not k.startswith("log/")))
print("is_first", obs["is_first"], "reward", obs["reward"])
print(
    "obs0 height",
    np.asarray(obs["height"]).round(6),
    "orient[:3]",
    np.asarray(obs["orientations"])[:3].round(6),
    "vel[:3]",
    np.asarray(obs["velocity"])[:3].round(6),
)

rewards = []
for i in range(60):
    obs = env.step({"action": actions[i], "reset": False})
    rewards.append(float(obs["reward"]))
print("rewards:", np.round(rewards, 8).tolist())
print("sum:", float(np.sum(rewards)))
print(
    "final height",
    float(np.asarray(obs["height"])),
    "orient[:3]",
    np.asarray(obs["orientations"])[:3].round(6).tolist(),
    "vel[:3]",
    np.asarray(obs["velocity"])[:3].round(6).tolist(),
)
print("act_space:", env.act_space)
