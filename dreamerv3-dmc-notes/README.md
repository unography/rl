# DreamerV3 DMC parity — `dreamerv3-jax-parity-dmc` branch

Bring TorchRL DreamerV3 toward parity with danijar's JAX reference on **DM
Control `walker_walk`** (proprioceptive). This branch = `main` + 18 initial core
parity commits + later DMC, parity, evidence, and performance commits.

**End goal:** torchrl's `walker_walk` eval curve lands inside the JAX 5-seed band
(final ~881; reaches ~800 by 200k env steps).

## Status
- Example runs Gym **and** DMC via `env.backend` (one script).
- `sota-implementations/dreamer_v3/config_dmc.yaml` maps the main model and loss
  settings from the JAX `dmc_proprio` size1m preset. It keeps documented runtime
  differences; see `CHANGE_GUIDE.md` Section 7.
- JAX reference curve extracted + committed (no JAX repo needed to overlay).
- The JAX reference itself is now **runnable side-by-side** (own venv), so
  per-term losses can be regenerated, not just quoted.
- Parameter counts match the reference module by module (640,867 params;
  `scripts/check_param_parity.py`). Source review found and fixed six
  architecture bugs and five loss bugs. Equal counts are a strong check, but
  are not by themselves proof of identical architecture. See `RESULTS.md`
  Steps 1c/1d.
- **Full-length curves still pending** — the run is dispatch-bound; see the
  throughput section of `RESULTS.md`.
- Fresh three-seed A100 minimum evidence is recorded in `RESULTS.md`: at ~10k,
  parity has 2/3 seeds inside the JAX band versus 0/3 for main, a closer median,
  2.83x main's mean return, and a same-seed loss vector close to local JAX.
  Those runs predate the current 16-environment collection schedule; fresh
  full-length curves are still required for a parity claim.

## Read in this order
| File | What |
|---|---|
| [CHANGE_GUIDE.md](CHANGE_GUIDE.md) | Current Torch-to-JAX line map, simple explanations, tests, limits, and commit ledger. |
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
  networks.rnn_hidden_dim=16 networks.num_categoricals=2 networks.num_classes=2 \
  networks.hidden_dim=8 networks.obs_embed_dim=8 networks.depth=1 \
  networks.num_reward_bins=11 networks.num_value_bins=11 replay_buffer.seq_len=4 \
  replay_buffer.buffer_size=256 collector.total_frames=144 \
  optimization.updates_per_batch=1 logger.eval_every=144 logger.eval_episodes=1 \
  logger.eval_max_steps=10 logger.output_plot=

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
