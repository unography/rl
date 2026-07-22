# DreamerV3 loss bugfixes — `dreamerv3-loss-bugfixes` branch

This branch is the **minimal, PR-ready fix** for two bugs in TorchRL's merged DreamerV3
world-model loss (`torchrl/objectives/dreamer_v3.py`, added in #3621). It is `main` plus exactly
two commits, touching only that one file:

```
838df378c [BugFix] DreamerV3 reco loss: don't symlog the prediction
98f62e881 [BugFix] DreamerV3 KL: apply free-nats to KL summed over categoricals
```

`git diff main` = 1 file, 2 hunks. All 41 tests in `test/objectives/test_dreamer_v3.py` pass.

## The two bugs (both are deviations from Hafner et al. 2023 / danijar/dreamerv3)

1. **KL free-nats clamped per categorical.** TorchRL floored `free_bits` on *each* categorical
   (`kl.clamp_min(free_bits).mean()`) instead of on the KL **summed over categoricals**
   (reference `rssm.py`: `Agg(OneHot(...), 1, jnp.sum)` then `max(kl, free_nats)`). Because
   `clamp_min` has zero gradient below the floor, per-categorical clamping freezes the KL at the
   floor and the world-model dynamics never train. Fix: `kl.sum(-1).clamp_min(free_bits).mean()`.

2. **Reconstruction double-symlogs the prediction.** TorchRL computed
   `(symlog(target) - symlog(pred))²`; the reference `symlog_mse` head compares the *raw*
   prediction to the symlog target: `(target_symlog - pred)²`. The decoder already predicts in
   symlog space, so symlog-ing it again double-compresses. Fix: drop the inner `symlog`.

## Documents here

| File | What |
|---|---|
| `GH_ISSUE.md` | Paste-ready GitHub issue: both bugs, current code vs. JAX (permalinked lines), fixes, measured impact. |
| `ANALYSIS.md` | Deep dive: full call chains on both sides, the math for why each bug breaks learning, file/line inventory. |
| `RESULTS.md` | The empirical A/B results (KL: clear performance delta; reco: muted on Pendulum). |
| `plots/kl_ablation.png` | KL bug A/B: fixed learns Pendulum (~-90), buggy stays flat (~-1400); KL pinned at the 1.10 floor. |
| `plots/reco_ablation.png` | Reco bug A/B: muted on Pendulum (both solve; small-obs env, symlog≈identity). |
| `scripts/` | The ablation harness (`run_kl_ablation.py`, `kl_ablation_entry.py`). |

## Important: reproducing the *performance* plots

The evidence is two-tier:

- **Code / reference argument** — self-contained on this branch: the fixed loss matches the JAX
  reference line-for-line (see `ANALYSIS.md`). This alone justifies both fixes.
- **Performance A/B** (`plots/kl_ablation.png`) — was produced on the **companion branch
  `dreamerv3-jax-parity`**, which carries the full working DreamerV3 example. This branch's
  example is `main`'s original, which does not learn (it has a separate blind-acting-policy bug,
  fixed on the companion branch), so the ablation scripts here will not reproduce the curves
  *on this branch*. Run them on `dreamerv3-jax-parity`.

## Verify the fixes

```bash
.venv/bin/python -m pytest test/objectives/test_dreamer_v3.py -q   # 41 passed
git diff main -- torchrl/objectives/dreamer_v3.py                  # the two hunks
```

## References

- Paper: Hafner et al. 2023, *Mastering Diverse Domains through World Models*,
  https://arxiv.org/abs/2301.04104
- Reference code: danijar/dreamerv3 @ `e3f0224` (`dreamerv3/rssm.py`, `embodied/jax/{outs,heads,nets}.py`)
- TorchRL base: `main` @ `ae421b98d`
