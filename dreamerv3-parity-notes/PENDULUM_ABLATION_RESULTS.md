# KL free-nats bug: Pendulum-v1 ablation results

Controlled A/B on `Pendulum-v1`, run locally on Apple M4 CPU via `.venv`. Both arms use the
branch's `sota-implementations/dreamer_v3` example **with the acting-policy fix applied** (see
below), identical config and seed; the **only** difference is `categorical_kl_balanced`:

- **fixed** — branch code: free-nats floor on the KL *summed over categoricals*
  (`kl_term.sum(-1).clamp_min(free_bits)`).
- **buggy** — origin/main behavior reproduced by monkeypatch: floor applied *per categorical*
  (`kl_term.clamp_min(free_bits)`), everything else (unimix, two-scale) identical.

Config: `num_categoricals=32, num_classes=32, rnn_hidden=256, hidden=256, updates_per_batch=16,
total_frames=50000, seed=0, free_bits=1.0`. Reproduce with
`dreamerv3-parity-notes/run_kl_ablation.py` (+ `kl_ablation_entry.py`).

> **Prerequisite: the acting-policy fix.** These results depend on a separate fix to the example
> (a recurrent real-env acting policy; see `ACTING_POLICY_FIX.md`). *Without* it the example does
> not learn Pendulum with either KL, so the KL bug was only visible at the mechanism level
> (KL pinned vs. alive) and produced no performance delta. *With* it the example learns, and the
> KL bug's downstream harm becomes measurable.

## Result (with acting-policy fix)

| arm | KL (nats, mean over run) | eval return >=14k (mean / best) | final |
|---|---|---|---|
| **fixed** | ~2.7 (rises to 4.5) | **-244 / -86** | -252 |
| **buggy** | **1.11** (pinned at floor) | **-1387 / -1061** | -1357 |

Pendulum random ~= -1300, solved ~= -150. Plot: `dreamerv3-parity-notes/results/kl_ablation.png`.
(Earlier mechanism-only plot, before the acting fix: `results/kl_ablation_mechanism_blindpolicy.png`.)

## Interpretation

**Mechanism (right panel).** The buggy per-categorical clamp pins the world-model KL at ~1.10
nats for the entire run -- the dead-gradient signature: `clamp_min` has zero gradient below the
floor, so applied per categorical it freezes the dynamics/representation KL. The fixed version's
KL climbs to 2-4.5 nats (gradient flows because the *summed* KL clears the floor).

**Performance (left panel).** With a learnable baseline in place, the fixed KL breaks through at
~14k env steps and reaches near-optimal (~-90 to -250); the buggy KL never learns (flat ~-1400).
The dead KL gradient means the prior never learns to predict the posterior, so the imagined
rollouts the actor-critic trains on do not match reality -- the policy optimizes a broken dream.
Reconstruction still drops in the buggy arm (the posterior can encode the obs), which is why the
bug is invisible on world-model reco loss alone and only shows up in dynamics/imagination/return.

## Takeaway (KL bug)

The KL free-nats bug is now demonstrated at **both** levels on Pendulum: mechanism (KL pinned at
the floor) **and** task performance (fixed learns ~-90, buggy stays ~-1400). This is a clean,
single-variable A/B suitable as the empirical figure in the GH issue.

---

# Reco double-symlog bug: Pendulum ablation (muted, as predicted)

Same setup, toggling the reconstruction bug instead (fixed-KL + acting fix in both arms; the
only difference is whether the decoder prediction is symlog'd again). Reproduced by temporarily
env-gating the reco line (`RECO_DOUBLE_SYMLOG=1`), reverted after via `git checkout`.
Plot: `results/reco_ablation.png`.

| arm | eval return >=14k (mean / best) |
|---|---|
| fixed reco | -244 / -86 |
| buggy reco (double-symlog) | -289 / -47 |

**Both learn Pendulum to ~-90 -- no measurable performance difference.** As predicted: Pendulum's
observations are small (`|obs| < ~5`), and `symlog` is near-identity there, so
`symlog(symlog(x)) ~= symlog(x)` barely distorts the target. The reco bug's *code* deviation from
JAX is unambiguous (see `DETAILED_ANALYSIS.md` sec 2), but its *performance* impact needs a
larger-observation environment (e.g. dm_control proprio, or any env with wide-range / unbounded
observations where symlog compression is significant). Pendulum cannot show it.
