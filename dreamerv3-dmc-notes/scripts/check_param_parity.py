"""Compare the torchrl DreamerV3 example's parameter counts against the JAX reference.

The JAX run prints an exact per-module parameter budget at startup::

    Optimizer opt has 640,867 params:
           364,416 dyn
            66,111 val
            57,663 rew
            51,096 dec
            50,316 pol
            41,153 con
            10,112 enc

Those numbers pin down every width, layer count and input wiring in the
``dmc_proprio`` / ``size1m`` preset, so matching them module-by-module is a
cheap, exact check on the port's architecture -- much sharper than eyeballing a
config table. Run from the repo root::

    .venv/bin/python dreamerv3-dmc-notes/scripts/check_param_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO / "sota-implementations" / "dreamer_v3"
sys.path.insert(0, str(EXAMPLE_DIR))

# JAX reference: dmc_proprio + size1m, walker_walk (24-dim obs, 6-dim action).
JAX_PARAMS = {
    "dyn": 364_416,
    "val": 66_111,
    "rew": 57_663,
    "dec": 51_096,
    "pol": 50_316,
    "con": 41_153,
    "enc": 10_112,
}


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> int:
    from dreamer_v3 import (  # noqa: PLC0415  (script-local import of the example)
        build_actor,
        build_shared_modules,
        build_value,
        build_world_model,
    )

    with initialize_config_dir(config_dir=str(EXAMPLE_DIR), version_base=None):
        cfg = compose(config_name="config_dmc")

    obs_dim, action_dim = 24, 6
    prior_net, reward_mlp, continue_mlp = build_shared_modules(
        cfg=cfg, action_dim=action_dim
    )
    world_model, encoder_net, posterior_net = build_world_model(
        cfg=cfg,
        obs_dim=obs_dim,
        prior_net=prior_net,
        reward_mlp=reward_mlp,
        continue_mlp=continue_mlp,
    )
    actor_model, _ = build_actor(cfg=cfg, action_dim=action_dim)
    value_model = build_value(cfg=cfg)

    # ``dyn`` in JAX is the whole RSSM: GRU core + prior head + posterior head.
    decoder = world_model[2]
    ours = {
        "dyn": count(prior_net) + count(posterior_net),
        "val": count(value_model),
        "rew": count(reward_mlp),
        "dec": count(decoder),
        "pol": count(actor_model),
        "con": count(continue_mlp),
        "enc": count(encoder_net),
    }

    print(f"{'module':>8} {'torchrl':>10} {'jax':>10} {'delta':>10}")
    ok = True
    for key, expected in JAX_PARAMS.items():
        got = ours[key]
        delta = got - expected
        ok &= delta == 0
        print(f"{key:>8} {got:>10,} {expected:>10,} {delta:>+10,}")
    total, jax_total = sum(ours.values()), sum(JAX_PARAMS.values())
    print(f"{'TOTAL':>8} {total:>10,} {jax_total:>10,} {total - jax_total:>+10,}")

    # The heads should also start at exactly zero (JAX outscale: 0.0).
    with torch.no_grad():
        feat = (torch.zeros(2, 128), torch.zeros(2, 512))
        rew_logits = reward_mlp(torch.cat(feat, -1))
        print(f"\nreward logits at init: max|logit| = {rew_logits.abs().max():.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
