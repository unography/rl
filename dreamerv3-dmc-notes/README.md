# DreamerV3 DMC parity — `dreamerv3-jax-parity-dmc` branch

Bring TorchRL DreamerV3 to parity with danijar's JAX reference on **DM Control
`walker_walk`** (proprioceptive). This branch = `main` + 18 validated parity
commits + the DMC-generalization commits made here.

**End goal:** torchrl's `walker_walk` eval curve lands inside the JAX 5-seed band
(final ~881; reaches ~800 by 200k env steps).

## Status
- Example now runs Gym **and** DMC via `env.backend` (one script). Verified on CPU.
- `sota-implementations/dreamer_v3/config_dmc.yaml` = JAX `dmc_proprio` size1m preset.
- JAX reference curve extracted + committed (no JAX repo needed to overlay).
- **Real training + overlay pending a GPU** — see `GPU_PARITY_PLAN.md`.

## Read in this order
| File | What |
|---|---|
| `LEARNING.md` | Blunt bulleted explanation of every change + concepts + JAX refs. |
| `GPU_PARITY_PLAN.md` | Step-by-step plan for the GPU-resume agent: VERIFY items, runs, acceptance table, debug tree. |
| `reference/dmc_walker_walk_dreamerv3_mean.csv` | JAX reference curve (5 seeds, mean/min/max) — the overlay target. |
| `reference/dmc_walker_walk_dreamerv3.csv` | Same, per-seed tidy rows. |
| `plots/dmc_walker_walk_reference.png` | JAX reference band, rendered. |
| `scripts/extract_jax_curve.py` | Pull any task's curve out of `scores/*.json.gz` -> CSV + plot. |
| `scripts/run_dmc_parity.py` | Run N seeds on GPU, overlay vs the reference band. |

## Quick commands
```bash
# Local CPU smoke (proves the DMC path, not learning):
.venv/bin/python sota-implementations/dreamer_v3/dreamer_v3.py --config-name config_dmc \
  env.domain=cartpole env.task=balance env.device=cpu \
  networks.rnn_hidden_dim=32 networks.num_categoricals=4 networks.num_classes=4 \
  networks.hidden_dim=32 networks.obs_embed_dim=32 replay_buffer.seq_len=16 \
  collector.total_frames=1000 collector.frames_per_batch=200 \
  optimization.updates_per_batch=2 logger.eval_every=400 logger.eval_episodes=1 \
  logger.eval_max_steps=100

# GPU parity run (real):
.venv/bin/python dreamerv3-dmc-notes/scripts/run_dmc_parity.py \
  --seeds 0 1 2 --total-frames 500000 --device cuda
```

## References
- Paper: Hafner et al. 2023, https://arxiv.org/abs/2301.04104
- Reference code: `danijar/dreamerv3 @ e3f0224`
- Companion branches: `dreamerv3-loss-bugfixes` (minimal library PR),
  `dreamerv3-jax-parity` (Pendulum working example + deep-dive analyses).
