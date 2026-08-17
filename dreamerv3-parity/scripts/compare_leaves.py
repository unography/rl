"""Compare the JAX param-tree leaf multiset with the torch optimizer leaf multiset."""
import pickle
from collections import Counter

import numpy as np

d = pickle.load(
    open(
        "/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/ckpt/"
        "20260808T203817F087503/agent.pkl",
        "rb",
    )
)
p = d["params"]
jax_leaves = {}
for k, v in p.items():
    if k.startswith("opt/") or k.startswith("slowval") or k.startswith("retnorm"):
        continue
    jax_leaves[k] = tuple(np.asarray(v).shape)


def canon(shape):
    # 2-D matrices differ by a transpose between JAX (in, out) and torch (out, in);
    # everything else (biases, norm scales, block kernels) is stored identically.
    if len(shape) == 2:
        return tuple(sorted(shape))
    return shape


jc = Counter(canon(s) for s in jax_leaves.values())
print(
    "JAX trainable leaves:",
    len(jax_leaves),
    "params:",
    sum(int(np.prod(s)) for s in jax_leaves.values()),
)
np.save(
    "/tmp/claude-1001/-home-ubuntu-rl/c57965a0-f02e-4e35-9698-498a529f8120/"
    "scratchpad/jax_leaf_shapes.npy",
    np.array(sorted(map(str, jax_leaves.values()))),
    allow_pickle=True,
)
import json

json.dump(
    {k: list(v) for k, v in jax_leaves.items()},
    open(
        "/tmp/claude-1001/-home-ubuntu-rl/c57965a0-f02e-4e35-9698-498a529f8120/"
        "scratchpad/jax_leaves.json",
        "w",
    ),
)
print("wrote jax_leaves.json")
