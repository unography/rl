# [BugFix] DreamerV3 world-model loss deviates from Hafner et al. 2023 in two places (KL free-nats granularity + double-symlog reconstruction)

## Summary

`torchrl/objectives/dreamer_v3.py` (added in #3621, refactored in #3780) implements the
DreamerV3 world-model loss. Two parts of the loss deviate from the reference
implementation (Hafner et al., *Mastering Diverse Domains through World Models*,
[arXiv:2301.04104](https://arxiv.org/abs/2301.04104); official code
[danijar/dreamerv3](https://github.com/danijar/dreamerv3)):

1. **KL free-nats are clamped per categorical instead of over the KL summed across
   categoricals.** This raises the effective free-nats floor by a factor of
   `num_categoricals` (default 32) and, more importantly, zeroes the KL gradient for the
   dynamics/representation learning in the common regime.
2. **The reconstruction loss applies `symlog` to *both* the target and the decoder
   prediction (double-symlog).** The reference applies `symlog` to the target only; the
   decoder already predicts in symlog space.

Both are present on `main` today (verified against commit `ae421b98d`). Reference lines
below are pinned to danijar/dreamerv3 commit `e3f0224`.

---

## Bug 1 — KL free-nats clamped per-categorical

### Current TorchRL code

[`torchrl/objectives/dreamer_v3.py#L226-L233`](https://github.com/pytorch/rl/blob/ae421b98d0dba86e5ab0b24917d1e64f376ee6f9/torchrl/objectives/dreamer_v3.py#L226-L233):

```python
# posterior/prior: [..., num_categoricals, num_classes]
post_sg = posterior.detach()
kl_term1 = (post_sg * (post_sg.log() - prior.log())).sum(-1)   # [..., num_categoricals]

prior_sg = prior.detach()
kl_term2 = (posterior * (posterior.log() - prior_sg.log())).sum(-1)

# Free bits per categorical (clamp before reducing). Hafner et al. 2023, eq. 5.
kl_term1 = kl_term1.clamp_min(free_bits).mean()   # <-- clamp is per-categorical
kl_term2 = kl_term2.clamp_min(free_bits).mean()   # <-- and reduces by MEAN, not SUM
```

The docstring (L194-L197) explicitly claims this is per-categorical and matches the paper.

### Reference (danijar/dreamerv3)

The RSSM distribution sums KL over the categorical axis *before* the free-nats floor:

[`dreamerv3/rssm.py#L173-L176`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L173-L176):

```python
def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)   # sum over the categorical axis
    return out
```

[`embodied/jax/outs.py#L73-L76`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/jax/outs.py#L73-L76) — `Agg.kl` sums the per-categorical KL:

```python
def kl(self, other):
    assert isinstance(other, Agg), other
    kl = self.output.kl(other.output)   # per-categorical KL
    return self.agg(kl, self.axes)      # agg=jnp.sum over axis -1 (categoricals)
```

Only *then* is the floor applied — [`dreamerv3/rssm.py#L125-L129`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L125-L129):

```python
dyn = self._dist(sg(post)).kl(self._dist(prior))   # already summed over categoricals
rep = self._dist(post).kl(self._dist(sg(prior)))
if self.free_nats:                                  # free_nats default = 1.0 (rssm.py L32)
    dyn = jnp.maximum(dyn, self.free_nats)          # floor on the SUMMED KL
    rep = jnp.maximum(rep, self.free_nats)
```

So the reference is `L_KL = max(free_nats, Σ_c KL_c)` — one floor per latent. TorchRL computes
`mean_c max(free_nats, KL_c)` — one floor per categorical.

### Why it matters

- **Over-regularization by `num_categoricals`×.** With the default 32 categoricals and
  `free_bits=1.0`, TorchRL floors *each* categorical at 1 nat (32 nats of budget) where the
  reference floors the *sum* at 1 nat.
- **Dead KL gradient (the real problem).** `clamp_min(x, f)` has gradient 0 wherever `x < f`.
  Per categorical, `KL_c` is usually below `free_bits` (its maximum is `log(num_classes)`), so
  the clamp is active and the KL term contributes **no gradient** to the dynamics/representation
  model across most of training. The prior never learns to predict the posterior, so imagined
  rollouts are uninformative and the actor-critic cannot improve. Summing first lets
  `Σ_c KL_c` exceed the floor, so gradients flow.

### Fix

```python
kl_term1 = kl_term1.sum(-1).clamp_min(free_bits).mean()   # sum over categoricals, then clamp
kl_term2 = kl_term2.sum(-1).clamp_min(free_bits).mean()
```

---

## Bug 2 — reconstruction loss double-symlogs the prediction

### Current TorchRL code

[`torchrl/objectives/dreamer_v3.py#L437-L448`](https://github.com/pytorch/rl/blob/ae421b98d0dba86e5ab0b24917d1e64f376ee6f9/torchrl/objectives/dreamer_v3.py#L437-L448):

```python
pixels = tensordict.get(("next", self.tensor_keys.pixels)).contiguous()
reco_pixels = tensordict.get(("next", self.tensor_keys.reco_pixels)).contiguous()  # decoder output
# Apply symlog before computing distance
if self.reco_loss == "l2":
    reco_loss = (symlog(pixels) - symlog(reco_pixels)).pow(2)   # <-- symlog on prediction too
else:
    reco_loss = (symlog(pixels) - symlog(reco_pixels)).abs()
```

`symlog` is compressive (`sign(x)·log1p(|x|)`), so `symlog(reco_pixels)` compresses the
prediction a *second* time (the decoder already emits values in symlog space), distorting the
target the network is trained to match.

### Reference (danijar/dreamerv3)

For vector/proprio observations the decoder uses a `symlog_mse` head —
[`dreamerv3/rssm.py#L299`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L299):

```python
o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
```

[`embodied/jax/heads.py#L127-L130`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/jax/heads.py#L127-L130):

```python
def symlog_mse(self, x):
    pred = self.sub('pred', nets.Linear, self.space.shape, **self.kw)(x)
    return outs.MSE(pred, nets.symlog)   # squash applied to TARGET only
```

[`embodied/jax/outs.py#L129-L141`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/jax/outs.py#L129-L141):

```python
class MSE(Output):
    def __init__(self, mean, squash=None):
        self.mean = f32(mean)
        self.squash = squash or (lambda x: x)
    def pred(self):
        return self.mean                                   # raw prediction, never squashed
    def loss(self, target):
        return jnp.square(self.mean - sg(self.squash(target)))   # (pred - symlog(target))^2
```

So the reference reconstruction loss is `(pred − symlog(target))²` — `symlog` on the target
only, prediction raw.

(Note: for *image* observations the reference uses a `sigmoid` head with plain MSE and **no
symlog at all** — [`rssm.py#L349-L356`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L349-L356), target `pixels/255` — so TorchRL's `symlog(symlog(·))` is wrong under either interpretation of the `pixels` key.)

### Fix

```python
if self.reco_loss == "l2":
    reco_loss = (symlog(pixels) - reco_pixels).pow(2)
else:
    reco_loss = (symlog(pixels) - reco_pixels).abs()
```

(The same double-symlog appears in the scalar reward fallback,
[`dreamer_v3.py#L464`](https://github.com/pytorch/rl/blob/ae421b98d0dba86e5ab0b24917d1e64f376ee6f9/torchrl/objectives/dreamer_v3.py#L464), which is only reached when `reward_two_hot=False`.)

---

## Expected behavior

The TorchRL world-model loss should match Hafner et al. 2023 / danijar/dreamerv3: free-nats
floored on the KL summed over categoricals, and reconstruction computed as raw prediction vs.
`symlog`-compressed target.

## Impact (measured)

Single-variable A/B on `Pendulum-v1` (`num_categoricals=32`, `free_bits=1.0`, seed 0, same
config; only the KL clamp granularity differs), with a correct acting policy in both arms:

| KL variant | world-model KL (nats) | eval return >=14k (mean / best) |
|---|---|---|
| fixed (summed clamp) | rises to 2-4.5 | **-244 / -86** (solves Pendulum) |
| buggy (per-categorical) | pinned at **1.10** (dead at floor) | **-1387 / -1061** (never learns) |

The buggy KL freezes the world-model KL at the free-bits floor, so the dynamics/representation
never train, imagination diverges from reality, and the actor-critic cannot learn -- eval return
stays at random level. The fixed KL learns the task. So **Bug 1 alone is the difference between
learning and not learning** on this task. (Plot: attached.)

**Bug 2 (reco)** deviation from JAX is unambiguous in code, but its performance impact is
*task-dependent*: on Pendulum (observations `|o| < ~5`, where `symlog` is near-identity) the
double-symlog is negligible and both variants solve the task equally. It requires a
wide-observation-range environment to show a performance delta; the code/reference argument
stands on its own.

The existing unit tests catch neither bug (they check keys/shapes/gradient existence, not the
loss formula).

## Verification (done)

- Bug 1: the A/B above (fixed learns Pendulum ~-90, buggy stays ~-1400), single variable.
- Both bugs: code matches the JAX reference lines cited above (three files for Bug 1, three for
  Bug 2). For a published-curve comparison, `dmc_cartpole_swingup` (reference `scores/` bundle)
  is the smallest proprio task; Bug 2's performance delta would need such a wider-obs task.

## Environment

- TorchRL `main` @ `ae421b98d`
- Reference: danijar/dreamerv3 @ `e3f0224`
- Affected file: `torchrl/objectives/dreamer_v3.py`
