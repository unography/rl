# DreamerV3 DMC parity — `dreamerv3-jax-parity-dmc` branch

Bring TorchRL DreamerV3 to parity with danijar's JAX reference on **DM Control
`walker_walk`** (proprioceptive). This branch = `main` + 18 validated parity
commits + the DMC-generalization commits made here.

**End goal:** torchrl's `walker_walk` eval curve lands inside the JAX 5-seed band
(final ~881; reaches ~800 by 200k env steps).

## Status
- Example runs Gym **and** DMC via `env.backend` (one script).
- `sota-implementations/dreamer_v3/config_dmc.yaml` = JAX `dmc_proprio` size1m preset.
- JAX reference curve extracted + committed (no JAX repo needed to overlay).
- The JAX reference itself is now **runnable side-by-side** (own venv), so
  per-term losses can be regenerated, not just quoted.
- Architecture matches the reference **exactly**, module by module
  (640,867 params; `scripts/check_param_parity.py`). Six architecture bugs and
  five loss bugs found and fixed — see `RESULTS.md` Steps 1c/1d.
- **Full-length curves still pending** — the run is dispatch-bound; see the
  throughput section of `RESULTS.md`.

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
| `scripts/check_param_parity.py` | Assert per-module parameter counts against the reference budget. |
| `reference/jax_walker_walk_losses_a6000.csv` | JAX per-term losses to 20k steps, measured locally. |

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

# GPU parity run (real), with the V3-off ablation arm overlaid:
.venv/bin/python dreamerv3-dmc-notes/scripts/run_dmc_parity.py \
  --seeds 0 1 2 --total-frames 500000 --device cuda \
  --arm v3off config_dmc_v3off
```

Arms: `branch` (`config_dmc`, all V3 features on -> should match JAX) and
`v3off` (`config_dmc_v3off`, V3 feature set off, acting-policy + loss fixes on
-> should learn but underperform). The `dreamerv3-baseline-dmc` branch adds the
flat main-algorithm control.

## References
- Paper: Hafner et al. 2023, https://arxiv.org/abs/2301.04104
- Reference code: `danijar/dreamerv3 @ e3f0224`
- Companion branches: `dreamerv3-loss-bugfixes` (minimal library PR),
  `dreamerv3-jax-parity` (Pendulum working example + deep-dive analyses).
