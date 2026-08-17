# What would prove TorchRL's DreamerV3 matches the JAX reference

Companion to `STATUS.md`, which records where the investigation *is*. This
records what "matching" would take, what is already established, and the one
test that would settle it.

Written 2026-08-17. The short version: **everything on either side of the
backward pass is verified; the gradients themselves have never been compared.**

---

## 1. The evidence hierarchy

Ranked by what each can actually establish, strongest first.

| # | Evidence | What it proves | Status |
|---|---|---|---|
| 1 | Per-parameter **gradients** agree on shared weights and shared data | The two implementations are the same algorithm | **missing** |
| 2 | One full learner step agrees: losses, gradients, parameter delta | As above, plus the optimizer wiring | **missing** |
| 3 | N-step trajectory agrees | As above, plus stateful pieces (EMAs, normalisers) | **missing** |
| 4 | Component forward passes agree on shared weights | Each function computes the reference's formula | **done** |
| 5 | Closed-form maths agrees (returns, weights, targets) | The derivations match | **done** |
| 6 | Optimizer agrees on shared gradients | The update rule matches | **done** |
| 7 | Architecture and cadence agree | Same model, same update schedule | **done** |
| 8 | Per-term `train/*` scalars agree during training | Weak: confounded by different data streams | partial |
| 9 | Learning curve overlaps | Nothing, at n=3 — see §5 | in progress |

The important structural point: items 4-7 verify the pieces *before* the
backward pass (the loss functions) and *after* it (the optimizer). Nothing
verifies the backward pass itself. A misplaced `stop_gradient`, a reduction over
the wrong axis, or a `.detach()` in the wrong branch passes every check we have
and produces exactly the symptom we cannot resolve: a slightly worse curve.

---

## 2. What is already established

All on shared inputs, against the reference at `/home/ubuntu/dreamerv3`.

**Forward passes** — `scripts/numeric_check.py` loads the JAX checkpoint's
weights into the TorchRL modules and compares in float64:

| component | max abs diff | value scale |
|---|---|---|
| block-GRU core | 1.34e-07 | 0.46 |
| prior logits | 8.74e-07 | 2.13 |
| encoder | 3.39e-07 | 0.45 |
| posterior logits | 3.97e-06 | 3.37 |
| decoder | 4.11e-07 | 0.78 |
| reward logits | 5.16e-06 | 48.2 |
| policy mean | 4.35e-07 | 0.80 |
| policy std | 1.27e-08 | 0.13 |

**Closed-form maths** — `scripts/cmp.py`, against a transcription of
`agent.py:lambda_return`: lambda return 9.5e-07, discount weight 8.9e-08,
advantage 9.5e-07, replay-value target exactly 0.

**Optimizer** — 1200 shared synthetic gradient steps through our
`_DreamerV3Optimizer` and through optax's chain: max abs diff 1.1e-08 across
1200x4 recorded update norms. Covers AGC, Adam, bias correction and warmup.

**Architecture** — all seven `size1m` leaf counts recomputed independently from
`configs.yaml`; 640,867 trainable parameters, exact
(`test_dreamer_v3_dmc_parameter_parity`).

**Cadence** — 1.0 learner update per env step after warmup on both sides;
measured slopes JAX 0.9976, Torch 1.0010.

---

## 3. The decisive test: gradient equivalence

### The reference already exposes the hook

`dreamerv3/agent.py:263` implements gradient-norm reporting, gated by
`report_gradnorms` (default `false`):

```python
lossfn = lambda data, carry: self.loss(
    carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
```

So the reference will already produce a **per-loss-key global gradient norm**
for every key in `self.scales` (`dyn`, `rep`, `rec`, `rew`, `con`, `policy`,
`value`, `repval`). That is the cheap first cut. `nj.grad(lossfn, self.modules)`
is the same hook for full per-parameter gradients.

`agent.py:156`, `def loss(self, carry, obs, prevact, training)`, is the single
entry point that produces every loss term.

### The procedure

1. **Shared weights.** Load the JAX checkpoint into both sides. The loader
   exists in `scripts/numeric_check.py`; extend it to cover every module, not
   just the eight it currently checks.
2. **Shared data.** Hand both implementations the *same* 16x64 window as raw
   tensors, bypassing both data pipelines. This deliberately does not test the
   pipelines — see §4.
3. **Deterministic sampling.** Both sides sample: the categorical
   straight-through in the RSSM, and the policy in imagination. Patch both to
   argmax, or inject a fixed noise tensor. The Torch test suite already does
   this with `deterministic_categorical` in `test_rssm_rollout_v3_forward`.
4. **Float32 both sides.** Disable bf16 autocast for the comparison; mixed
   precision is a separate question from algorithmic equivalence.
5. **Compare, in increasing strength:**
   - every loss term (`dyn`, `rep`, `rec`, `rew`, `con`, `policy`, `value`,
     `repval`) — should already pass, given §2
   - per-loss-key global gradient norm, via `report_gradnorms`
   - **per-parameter gradients**, via `nj.grad` against Torch's `.grad`
   - the parameter delta after one `optimizer.step()`
6. **Then iterate ~50 steps** on the same fixed window and check the parameter
   trajectories do not drift. This is what catches the stateful pieces a single
   step cannot: the return-normalisation EMA (`return_low`/`return_high`), the
   slow-critic tau update, and the free-nats clamp interacting with running
   values.

If per-parameter gradients agree to float32 across all 640,867 parameters, the
implementations are the same algorithm and the learning curve stops being
relevant to the question.

### Known obstacles

- The JAX side needs driving outside its training loop. `run_jax_seed.py`
  already constructs an agent from the reference config; the loss is reachable
  from there.
- Gradient sign and layout conventions differ (JAX kernels are `[in, out]`,
  Torch `[out, in]`); `numeric_check.py` already handles the transpose for
  weights and the same applies to gradients.
- Block-linear kernels are 3-D `(blocks, in/blocks, out/blocks)` on both sides
  and need no transpose — `numeric_check.py:setblock`.
- The two implementations disagree on RNG stream *by construction* even when
  outputs agree, which is why step 3 is mandatory rather than optional. See
  `test_rssm_rollout_v3_forward`, which pins that divergence.

---

## 4. The second gap: data pipelines

Gradient equivalence on a shared window says nothing about whether the two
pipelines *produce* equivalent windows. Ours re-encodes the collector stream
through a shifted writer (action `a_i` with observation `o_{i+1}`, plus an
explicit terminal-to-reset edge and a replay-context record); the reference
writes into a chunk store.

The check is cheaper than §3: fix a seed, dump the actual record stream each
side produces for the first few thousand steps, and diff them field by field.

This matters because sequence misalignment is precisely the class of bug that
survives every test we have and shows up only as a slightly worse curve.
Alignment has been verified **by reading the code**, not by comparing data.

---

## 5. What the learning curve can and cannot do

It is a smoke test that the assembled system trains. It is not parity evidence,
and no amount of additional compute at three seeds a side changes that.

From `STATUS.md` §1: the reference's own three seeds span 397 return units at
176k, one seed collapses from 150 to 54 between 80k and 112k before recovering
to 418, and at 192k the three have mean 561 with standard deviation ~137.
Separating two such distributions by 100 return units needs roughly **30 seeds
per side**.

So when quoting the curve comparison:

- An overlapping band means **"not contradicted"**. It never means "matches".
- State the resolution limit alongside the numbers, every time.
- If a return-level answer is genuinely wanted, spend the budget on many seeds
  at a shorter horizon, not three seeds at 200k.

---

## 6. Recommended order

1. Per-loss-key gradient norms (`report_gradnorms: true` on the JAX side).
   Cheapest, and would catch a gross backward-pass error immediately.
2. Per-parameter gradients on a single shared step. Decisive for §1 item 1.
3. Parameter delta after one optimizer step, then a ~50-step trajectory.
4. Data-pipeline record-stream diff (§4).
5. The curve, as a smoke test only.

Items 1-3 are the only ones that can turn "not contradicted" into "matches".
