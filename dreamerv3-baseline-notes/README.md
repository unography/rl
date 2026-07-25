# `dreamerv3-baseline-dmc` — the control arm

torchrl `main` + minimal DMC plumbing, no fixes. Runs walker on GPU to show that
without the parity work it does **not** match the JAX curve. See **`BASELINE.md`**.

| Path | What |
|---|---|
| `BASELINE.md` | What this branch is, why it's a fair control, how to run, the three-way money plot. |
| `scripts/run_baseline.py` | Run N seeds, overlay vs JAX; `--parity-csv` adds the parity curve (three-way). |
| `reference/*.csv` | JAX walker_walk reference curve (5 seeds), extracted from the scores json. |
| `plots/` | Filled on GPU: `*_baseline_vs_jax.png`, `*_three_way.png`. |

Companion branches: `dreamerv3-jax-parity-dmc` (the fixes + parity target),
`dreamerv3-loss-bugfixes` (minimal library PR).
