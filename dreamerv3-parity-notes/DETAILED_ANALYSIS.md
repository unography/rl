# DreamerV3 world-model loss: TorchRL `main` vs. the JAX reference — detailed analysis

This is the "understand it deeply" companion to `DRAFT_GH_ISSUE.md`. It walks the full call
chains on both sides, does the math for why each deviation breaks learning, and lists every
file/line you need to look at.

**Pinned versions**

| Codebase | Repo | Commit | Key file |
|---|---|---|---|
| TorchRL (buggy) | `pytorch/rl` | `ae421b98d` (`origin/main`) | `torchrl/objectives/dreamer_v3.py` |
| Reference (correct) | `danijar/dreamerv3` | `e3f0224` (local `_ref/dreamerv3`) | `dreamerv3/rssm.py`, `embodied/jax/{outs,heads,nets}.py`, `dreamerv3/agent.py` |

The two bugs are the two commits on our branch `dreamerv3-jax-parity`:
`c423c092f` (KL) and `1d687a614` (reco). Everything below is why those commits are correct.

---

## 0. Background: what the world-model loss is supposed to be

DreamerV3's world model is trained with three losses (plus reward/continue heads):

```
L = β_pred · L_reco  +  β_dyn · L_dyn  +  β_rep · L_rep
```

- `L_reco` — reconstruct the observation from the latent (decoder).
- `L_dyn` — KL(sg(posterior) ‖ prior): train the **prior** to predict the posterior.
- `L_rep` — KL(posterior ‖ sg(prior)): train the **posterior** (representation) toward the prior.

The stochastic latent is a grid of `stoch` categorical variables, each over `classes` classes.
Reference defaults (`configs.yaml` base `rssm`, L91): `stoch=32, classes=64`. The tiny `size1m`
preset (L120-123) overrides `classes: 4` but **leaves `stoch=32`** — this matters below.

Two reference constants:
- `symlog(x) = sign(x)·log1p(|x|)`, `symexp(x) = sign(x)·expm1(|x|)` — a signed log compressor and
  its inverse (`embodied/jax/nets.py:59-64`). TorchRL has identical definitions
  (`dreamer_v3.py:43`, `:67`).
- `free_nats = 1.0` (reference `rssm.py:32`); TorchRL calls it `free_bits`, default `1.0`.

---

## 1. Bug 1 — KL free-nats granularity

### 1.1 The reference call chain (correct)

`dreamerv3/rssm.py:120-133` — `RSSM.loss`:

```python
def loss(self, carry, tokens, acts, reset, training):
    ...
    prior = self._prior(feat['deter'])
    post  = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))   # L125
    rep = self._dist(post).kl(self._dist(sg(prior)))   # L126
    if self.free_nats:                                 # L127
      dyn = jnp.maximum(dyn, self.free_nats)           # L128
      rep = jnp.maximum(rep, self.free_nats)           # L129
    losses = {'dyn': dyn, 'rep': rep}
```

What does `self._dist(...).kl(...)` return? `dreamerv3/rssm.py:173-176`:

```python
def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)   # per-categorical distribution
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)          # wrap: aggregate 1 event dim by SUM
    return out
```

`Agg.kl` — `embodied/jax/outs.py:73-76`:

```python
def kl(self, other):
    kl = self.output.kl(other.output)   # KL per categorical -> shape [..., stoch]
    return self.agg(kl, self.axes)      # agg=jnp.sum, axes=[-1]  ->  Σ over the stoch axis
```

`Agg.__init__` (`outs.py:42-45`) sets `axes = [-i for i in range(1, dims+1)] = [-1]` for `dims=1`.

**So the reference computes `KL = Σ_c KL_c` first, and only then `max(KL, free_nats)`.** The
floor is a single per-latent budget. Then `agent.py:166-168` collects `dyn`/`rep` into the loss
and later means over batch/time.

### 1.2 The TorchRL call chain (buggy)

`torchrl/objectives/dreamer_v3.py:182-235` — `categorical_kl_balanced`:

```python
post_sg = posterior.detach()
kl_term1 = (post_sg * (post_sg.log() - prior.log())).sum(-1)      # L226  [..., stoch]  (sum over CLASSES)
prior_sg = prior.detach()
kl_term2 = (posterior * (posterior.log() - prior_sg.log())).sum(-1)  # L229  [..., stoch]

# Free bits per categorical (clamp before reducing). Hafner et al. 2023, eq. 5.   # L231 (comment)
kl_term1 = kl_term1.clamp_min(free_bits).mean()                  # L232  clamp per-categorical, then MEAN over [batch, stoch]
kl_term2 = kl_term2.clamp_min(free_bits).mean()                  # L233
```

Note `.sum(-1)` on L226/L229 sums over **classes** (correct — that is the per-categorical KL).
The bug is the *next* reduction: TorchRL clamps each categorical, then takes `.mean()` over
everything including the `stoch` axis.

### 1.3 Side-by-side

```
                        per-categorical KL_c            floor                reduce over stoch
reference (correct):    KL_c (over classes)   ── Σ_c ──►  max(·, free_nats)  ── (mean over batch)
torchrl  (buggy):       KL_c (over classes)   ────────►  max(·, free_nats)  ── mean over stoch+batch
                                                          ▲ applied per categorical, before summing
```

Formulae:

```
reference:  L_KL = mean_batch[ max( free_nats , Σ_c KL_c ) ]
torchrl:    L_KL = mean_batch[ mean_c max( free_nats , KL_c ) ]
```

The one-line fix (`c423c092f`) inserts `.sum(-1)` before the clamp:

```python
kl_term1 = kl_term1.sum(-1).clamp_min(free_bits).mean()
kl_term2 = kl_term2.sum(-1).clamp_min(free_bits).mean()
```

### 1.4 Why the buggy version breaks learning

Two coupled effects:

**(a) The floor is `stoch`× too high.** Because each of `stoch` categoricals is independently
floored at `free_nats`, the effective total budget is `stoch · free_nats` (32 nats at defaults),
not `free_nats` (1 nat).

**(b) The KL gradient is dead in the common regime (the dominant effect).**
`clamp_min(x, f)` has derivative `1` if `x > f` and `0` if `x < f`. Consider a single categorical:
`KL_c ∈ [0, log(classes)]`. Early and mid training, per-categorical KL is small — often
`KL_c < 1.0 = free_nats`. Whenever that holds, `clamp_min(KL_c, 1.0)` is on its flat branch and
contributes **zero gradient** to prior/posterior. So the dynamics (`L_dyn`) and representation
(`L_rep`) terms stop pushing the prior toward the posterior.

With `size1m` (`classes=4`) this is nearly guaranteed: `max KL_c = log 4 = 1.386` nats, barely
above the 1.0 floor, so almost every categorical sits in the dead zone. The world model never
learns coherent dynamics → imagined trajectories are garbage → the actor-critic (trained purely
in imagination) cannot improve → flat eval return despite decreasing reconstruction loss.

In the reference, `Σ_c KL_c` (summed over 32 categoricals) comfortably exceeds `free_nats=1.0`,
so `max(Σ_c KL_c, 1.0)` is on its identity branch and gradients flow to all categoricals.

### 1.5 Numeric sanity example

Say `stoch=32`, and each `KL_c ≈ 0.1` nat (small, typical once the posterior partly matches the prior):

| Quantity | Reference | TorchRL (buggy) |
|---|---|---|
| Σ_c KL_c | 3.2 nats | — |
| after floor | `max(3.2, 1.0)=3.2` | per-cat `max(0.1,1.0)=1.0` each |
| reduce | `3.2` | `mean_c 1.0 = 1.0` |
| gradient to KL_c | flows (identity branch) | **zero** (flat branch) |

Same latent, opposite learning signal.

---

## 2. Bug 2 — reconstruction double-symlog

### 2.1 The reference decoder: two paths

`dreamerv3/rssm.py:288-359` — `Decoder.__call__`. It has **two** reconstruction paths:

**Vector / proprio observations** (`rssm.py:297-306`):

```python
o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')   # L299
outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
...
outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)   # symlog_mse head
```

**Image observations** (`rssm.py:308-356`):

```python
x = jax.nn.sigmoid(x)                          # L349  output in [0,1]
...
out = embodied.jax.outs.MSE(out)               # L354  plain MSE, NO squash
out = embodied.jax.outs.Agg(out, 3, jnp.sum)   # L355
```

The `symlog_mse` head — `embodied/jax/heads.py:127-130`:

```python
def symlog_mse(self, x):
    pred = self.sub('pred', nets.Linear, self.space.shape, **self.kw)(x)
    return outs.MSE(pred, nets.symlog)   # squash = symlog, passed as the TARGET transform
```

`MSE` — `embodied/jax/outs.py:129-141`:

```python
class MSE(Output):
    def __init__(self, mean, squash=None):
        self.mean = f32(mean)
        self.squash = squash or (lambda x: x)
    def pred(self):
        return self.mean                                    # prediction returned RAW (symexp is done by callers when needed)
    def loss(self, target):
        return jnp.square(self.mean - sg(self.squash(target)))   # (pred - symlog(target))^2
```

And where the loss is actually taken — `dreamerv3/agent.py:178-182`:

```python
for key, recon in recons.items():
    space, value = self.obs_space[key], obs[key]
    target = f32(value) / 255 if isimage(space) else value   # L181 images normalized, vectors raw
    losses[key] = recon.loss(sg(target))                     # L182  symlog applied INSIDE MSE.loss (target only)
```

### 2.2 The three reconstruction formulae

| Case | Reference loss | symlog on target? | symlog on prediction? |
|---|---|---|---|
| Vector/proprio | `(pred − symlog(target))²` | yes | **no** |
| Image | `(sigmoid(net) − target/255)²` | no | no |
| **TorchRL (both cases)** | `(symlog(target) − symlog(pred))²` | yes | **yes (bug)** |

`torchrl/objectives/dreamer_v3.py:437-448`:

```python
pixels      = tensordict.get(("next", self.tensor_keys.pixels)).contiguous()       # target
reco_pixels = tensordict.get(("next", self.tensor_keys.reco_pixels)).contiguous()  # decoder prediction
# Apply symlog before computing distance                                            # L441
if self.reco_loss == "l2":
    reco_loss = (symlog(pixels) - symlog(reco_pixels)).pow(2)                        # L443  <-- symlog(pred) is the bug
else:
    reco_loss = (symlog(pixels) - symlog(reco_pixels)).abs()                         # L445
```

TorchRL's `DreamerV3ModelLoss` symlogs the target (so it is modeling the *vector* `symlog_mse`
path — appropriate for its Pendulum example, whose observation is a 3-vector, not an image).
But it *also* symlogs the prediction, which the reference never does in **either** path.

### 2.3 Why it's wrong (math)

`symlog` is a compressive nonlinearity. A `symlog_mse` output head is *defined* so the network's
raw output lives in symlog space: at inference the scalar is recovered with
`symexp(pred)`. The training target is therefore `symlog(true_value)`, and the residual is
`pred − symlog(true_value)`.

Applying `symlog` to `pred` as well computes `symlog(pred) − symlog(target)`, i.e.
`symlog(symlog_space_output)` — a second compression. Consequences:

- **Wrong optimum.** The network is now driven so that `symlog(pred) ≈ symlog(target)`, i.e.
  `pred ≈ target` in *raw* space — but the head/`symexp` readout assumes `pred ≈ symlog(target)`.
  Prediction and readout disagree, so reconstructions are biased.
- **Shrunken, saturating gradient.** `d symlog(p)/dp = 1/(1+|p|)`. For large `|pred|` the extra
  factor drives the reconstruction gradient toward 0, so large-magnitude observation dimensions
  are under-fit. The compression also squashes the dynamic range the world model must model.

The fix (`1d687a614`) drops the inner `symlog` on the prediction:

```python
reco_loss = (symlog(pixels) - reco_pixels).pow(2)   # raw pred vs symlog(target)
```

### 2.4 Related instance (reward fallback)

`torchrl/objectives/dreamer_v3.py:464` has the same pattern for the scalar reward head:

```python
reward_loss = (symlog(true_reward) - symlog(pred_reward)).pow(2).squeeze(-1)
```

This branch is only hit when `reward_two_hot=False` (the default DreamerV3 reward head is
`symexp_twohot`, `heads.py:132-144`, which our branch uses). Worth fixing for consistency but
not on the default path.

---

## 3. File/line inventory (quick lookup)

### TorchRL `main` @ `ae421b98d` — `torchrl/objectives/dreamer_v3.py`
| Lines | What |
|---|---|
| 43, 67 | `symlog` / `symexp` defs |
| 182-235 | `categorical_kl_balanced` (Bug 1) |
| 226, 229 | per-categorical KL (sum over classes) — correct |
| **232, 233** | **per-categorical clamp + mean (Bug 1)** |
| 416-448 | `DreamerV3ModelLoss.forward` reco section |
| **443, 445** | **double-symlog reconstruction (Bug 2)** |
| 464 | double-symlog reward fallback (related) |

### Reference @ `e3f0224`
| File:lines | What |
|---|---|
| `dreamerv3/rssm.py:32` | `free_nats = 1.0` |
| `dreamerv3/rssm.py:120-133` | `RSSM.loss` — KL + `jnp.maximum(·, free_nats)` after the sum |
| `dreamerv3/rssm.py:173-176` | `_dist` = `OneHot` then `Agg(·, 1, jnp.sum)` |
| `embodied/jax/outs.py:40-76` | `Agg` (sums per-categorical KL over axis -1) |
| `embodied/jax/outs.py:129-141` | `MSE` (squash applied to target only) |
| `embodied/jax/heads.py:127-130` | `symlog_mse` head = `MSE(pred, symlog)` |
| `dreamerv3/rssm.py:297-306` | vector decoder → `symlog_mse` |
| `dreamerv3/rssm.py:349-356` | image decoder → `sigmoid` + plain `MSE`, no symlog |
| `dreamerv3/agent.py:178-182` | reconstruction loss taken as `recon.loss(sg(target))` |
| `dreamerv3/configs.yaml:91` | base rssm `stoch:32, classes:64` |
| `dreamerv3/configs.yaml:120-123` | `size1m`: `classes:4` (stoch stays 32) |

---

## 4. How to prove it (no full training required to understand it)

1. **Static proof (already done above):** the code + reference lines establish that TorchRL
   deviates. Bug 2 is airtight from three reference files; Bug 1's *existence* is airtight, its
   *magnitude* is what the curve demonstrates.
2. **Cheapest empirical proof — ablation on our branch:** run the example on a task with a
   published DreamerV3 curve (`dmc_cartpole_swingup`, `scores/dmc_proprio-dreamerv3.json.gz`),
   with (a) both fixes, (b) Bug 1 reverted, (c) Bug 2 reverted. Each revert is one line, so
   attribution is clean. Expectation: reverting Bug 1 flatlines eval return; reverting Bug 2
   distorts reconstruction and slows/limits learning.
3. **Gold-standard — inject into the reference:** apply the *TorchRL* behavior to danijar/dreamerv3
   (`rssm.py:127-129` per-categorical max; `outs.py:MSE.loss` double-symlog) and run
   `dmc_cartpole_swingup` at `size1m`. Same regression in the ground-truth code = zero-confound
   evidence. (Compute caveat: the reference pins `jax[cuda12]`; on this Apple-silicon box it needs
   a CPU jax build, so use short partial runs — the KL bug shows up early.)

---

## 5. One-paragraph summary

TorchRL's merged DreamerV3 world-model loss (`#3621`) has two transcription errors vs. Hafner's
reference. (1) It applies the KL free-nats floor **per categorical** (`clamp_min` then `mean`
over the `stoch` axis) instead of on the KL **summed over categoricals** (`sum` then `max`),
which raises the floor ~`num_categoricals`× and — because `clamp_min` has zero gradient below
the floor — kills the dynamics/representation KL gradient in the usual small-per-categorical-KL
regime, so the world model never learns. (2) It applies `symlog` to the **decoder prediction**
as well as the target, double-compressing a head that already predicts in symlog space, biasing
the optimum and shrinking the reconstruction gradient. Both are one-line fixes
(`c423c092f`, `1d687a614`) and both are demonstrable via a `dmc_cartpole_swingup` ablation.
