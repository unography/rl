# DreamerV3 JAX-parity — `dreamerv3-jax-parity` branch

This branch carries the **full working DreamerV3** (world model + actor-critic example) brought
into parity with danijar's JAX reference. Unlike `main`, the example here actually **learns**:
on `Pendulum-v1` eval return climbs from ~-1350 (random) to ~-90 (near-optimal).

It contains three bugfixes and the feature set that, together, make DreamerV3 correct and
performant. The two **library** bugfixes also live alone on the companion branch
`dreamerv3-loss-bugfixes` (the minimal, PR-ready torchrl fix).

## The three bugs fixed

| Bug | File | Evidence |
|---|---|---|
| **KL free-nats clamped per-categorical** | `torchrl/objectives/dreamer_v3.py` | Mechanism (KL pinned at 1.10 floor) **and** performance (fixed learns ~-90, buggy flat ~-1400). `plots/kl_ablation.png` |
| **Reconstruction double-symlog** | `torchrl/objectives/dreamer_v3.py` | Code-vs-JAX airtight; performance muted on Pendulum's small obs. `plots/reco_ablation.png` |
| **Blind real-env acting policy** | `sota-implementations/dreamer_v3/dreamer_v3.py` | Example went from never-learning to solving Pendulum; the fix that unblocked everything. |

## Documents

| File | What |
|---|---|
| `DRAFT_GH_ISSUE.md` | GitHub issue for the two **library** bugs (KL + reco), code vs JAX, measured impact. |
| `DRAFT_GH_ISSUE_ACTING_POLICY.md` | GitHub issue for the **example** acting-policy bug. |
| `DETAILED_ANALYSIS.md` | Deep dive on the KL + reco bugs: call chains, math, file/line inventory. |
| `ACTING_POLICY_FIX.md` | Deep dive on the blind-acting-policy bug: rollout probes, the fix, JAX parity. |
| `PENDULUM_ABLATION_RESULTS.md` | All empirical results (KL performance delta, reco muted). |
| `plots/kl_ablation.png` | KL A/B: eval return + world-model KL, fixed vs buggy. |
| `plots/reco_ablation.png` | Reco A/B on Pendulum (muted). |
| `plots/kl_ablation_mechanism_blindpolicy.png` | Earlier KL-mechanism-only plot (before the acting-policy fix). |
| `scripts/` | Ablation harness: `run_kl_ablation.py`, `kl_ablation_entry.py`. |

## How the fixes relate

The blind-acting-policy bug (example) was masking the library bugs: until the policy could
perceive observations, nothing learned, so the KL bug could only be shown at the mechanism level.
Fixing the acting policy gave a learnable baseline, which then let the KL bug show a clean
performance delta (fixed learns Pendulum, buggy does not).

## Reproduce

```bash
# Train (learns Pendulum ~-90 by ~16-34k frames; seed-dependent breakthrough timing):
.venv/bin/python sota-implementations/dreamer_v3/dreamer_v3.py \
  networks.num_categoricals=32 networks.num_classes=32 \
  networks.rnn_hidden_dim=256 networks.hidden_dim=256 \
  optimization.updates_per_batch=16 collector.total_frames=50000 \
  logger.eval_every=2000 env.seed=0

# KL A/B (fixed vs buggy KL, same acting fix):
.venv/bin/python dreamerv3-parity-notes/scripts/run_kl_ablation.py
```

## Companion branch

`dreamerv3-loss-bugfixes` = `main` + only the two library bugfixes (KL + reco), 2-hunk diff,
41/41 objective tests pass — the clean PR target for the torchrl library.

## References

- Paper: Hafner et al. 2023, https://arxiv.org/abs/2301.04104
- Reference code: danijar/dreamerv3 @ `e3f0224`
- TorchRL base: `main` @ `ae421b98d`
