# DreamerV3 `walker_walk` — torchrl vs JAX parity results

Branch `dreamerv3-jax-parity-dmc`. Reference: danijar/dreamerv3 `dmc_proprio`
(size1m) published curve, `reference/dmc_walker_walk_dreamerv3_mean.csv`
(5 seeds). JAX source read directly from a local checkout at `/root/dreamerv3`.

## Hardware / environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (driver 580.126.09) |
| CPU / RAM | 22 cores / 117 GB |
| OS | Ubuntu 22.04.5 |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| torchrl | 0.13.0+g66e08fb6 (editable, C++ ext built for py3.12) |
| tensordict | 0.13.0+g8f37f8e |
| mujoco / dm_control | 3.10.0 |
| headless GL | `MUJOCO_GL=egl` |

## Step 1 — VERIFY items, resolved against the JAX source

The plan flagged six best-effort mappings. All six were checked line-by-line
against `/root/dreamerv3`. **Two of the plan's own premises turned out to be
misreadings of the JAX source** (items 1 and 3); the corrected findings are
below.

### 1. `use_reinforce` — no change (plan premise was wrong)

The plan asserted "JAX continuous control backprops the actor through the
dynamics (`reward_grad: True`), not REINFORCE". Both halves are wrong:

- `dreamerv3/agent.py:411-414` (`imag_loss`) computes
  `policy_loss = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + actent * ent)`.
  That is **REINFORCE with a stop-gradiented advantage** — for continuous
  control too. There is no dynamics-backprop branch in this codebase.
- `reward_grad` is unrelated to the actor. It appears once, at
  `agent.py:172`: `inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)`
  — it controls whether the **reward head's** gradient flows back into the RSSM
  representation features, i.e. a world-model concern.

torchrl's `DreamerV3ActorLoss._forward_imag_loss`
(`torchrl/objectives/dreamer_v3.py:1010-1016`) computes
`(w * -(logp * adv.detach() + entropy_bonus * ent)).mean()` — the same thing.
`use_reinforce` is **dead config** on this path: it only gates the non-`imag_loss`
`forward`. Left at `false`; comment updated to say so.

### 2. Reward bins — `41` -> `255` (changed)

JAX `rewhead: {output: symexp_twohot, bins: 255}`. `embodied/jax/heads.py:136-144`
builds the grid as `symexp(linspace(-20, 0, 128))` mirrored, i.e. exactly
`symexp(linspace(-20, 20, 255))`.

torchrl builds `reward_bins = torch.linspace(-20, 20, N)` and decodes
`symexp(two_hot_decode(logits, bins))` — the bins live in **symlog** space, so
the effective reward-space grid is `symexp(linspace(-20, 20, N))`. With `N=255`
the two grids are **identical**; the only residual difference is that JAX
interpolates the two-hot weights in reward space and torchrl in symlog space
(second-order).

At `N=41` the symlog-space spacing is 1.0, so walker's per-step reward in `[0,1]`
(symlog `[0, 0.693]`) only ever touches **two** bins, versus ~10 for JAX. The
two-hot mean is still exact by interpolation, but the classification target is
far coarser than the reference. Set to `255` to match. (Value bins were already
255 and use the same `_default_bins` grid.)

### 3. MLP `depth` — `2` -> `3` (changed; plan premise was wrong)

The plan read `size1m`'s `.*\.depth: 4` as an MLP hidden-layer count. It is not.
`size1m` is a regex overlay:

```yaml
size1m: &size1m
  .*\.rssm: {deter: 512, hidden: 64, classes: 4}
  .*\.depth: 4
  .*\.units: 64
```

`.*\.depth` matches `enc.simple.depth` / `dec.simple.depth`, which are **CNN
channel depths**. `dmc_proprio` sets `env.dmc.image: False`, so the CNN path is
never built and `depth` is irrelevant. The MLP hidden-layer count in JAX is
`layers`, which `size1m` does **not** touch: `enc 3, dec 3, policy 3, value 3,
rewhead 1, conhead 1`. Only `units` is overridden, to 64.

torchrl's `_norm_mlp` applies one global `cfg.networks.depth` to every head, so
an exact match is impossible without per-head config. Set to `3` to match the
four networks that dominate capacity (encoder, decoder, policy, value); the
reward and continue heads get 3 hidden layers instead of 1, which is cheap at
`units=64`.

### 4. `contdisc` — no change (`false` is equivalent here)

JAX `contdisc: True` does two things: `agent.py:175-176` scales the continue
**target** by the horizon (`con *= 1 - 1/horizon`), and `imag_loss` then sets
`disc = 1`. torchrl's `contdisc: false` instead sets `disc = 1 - 1/horizon` and
uses the raw continue head. For walker, which never terminates early, `con -> 1`
under both, so JAX's `cumprod(1 * 0.997*con)` and torchrl's
`cumprod(0.997*con)/0.997` agree up to a constant factor of `1/0.997` on the
imagination weights. Kept `false`.

### 5. Actor entropy `actent` — no change (already correct)

JAX `imag_loss.actent: 3e-4`. `DreamerV3ActorLoss.entropy_bonus` defaults to
`3e-4` and the example does not override it. Not config-exposed, but correct.

### 6. `obs_embed_dim` — no change

JAX `enc.simple.units` under `size1m` is 64. Config already 64.

### Additional cross-checks (all matching, not in the plan's list)

| Item | JAX | torchrl | |
|---|---|---|---|
| actor dist | `bounded_normal`: `tanh(mean)`, `(hi-lo)*sigmoid(raw+2)+lo` | `BoundedNormalActor` identical | ok |
| retnorm | `perc`, rate .01, limit 1.0, perc 5/95, no debias | `_ReturnNormalizer` identical | ok |
| valnorm / advnorm | `impl: none` | not implemented | ok |
| slow critic | rate .02, `slowreg 1.0`, `slowtar: False` | same; bootstrap on online critic | ok |
| free nats | `maximum(sum_c KL_c, 1.0)` | `kl.sum(-1).clamp_min(1.0)` | ok |
| KL scales | `dyn 1.0, rep 0.1` | same | ok |
| optimizer | lr 4e-5, agc 0.3, warmup 1000, RMS-before-momentum | `DreamerV3Optimizer` | ok |
| action repeat | `env.dmc.repeat: 1` | DMC default 1 | ok |
| train_ratio | 1024 | 16*16*64/16 = 1024 | ok |
| RSSM | deter 512, stoch 32, classes 4, hidden 64, blocks 8, unimix .01 | same | ok |

### Known remaining gaps vs JAX (not fixed)

These are real deviations, recorded for honesty rather than fixed in this pass:

1. **Output-layer init (`outscale`).** JAX zero-inits the reward and value head
   output layers (`outscale: 0.0`) and scales the policy output layer by `0.01`.
   The torchrl example uses default init throughout. Affects early-training
   bias, not the asymptote.
2. **Replay critic loss (`repl_loss` / `repval`).** JAX also trains the critic on
   *real replay sequences* (`agent.py:219-233`, `loss_scales.repval: 0.3`,
   `repval_loss: True`). torchrl trains the critic on imagined trajectories only.
   This is the largest structural gap.

## Step 2 — throughput

_pending_

## Step 3 — shakeout

_pending_

## Step 4 — full grid

_pending_

## Acceptance vs JAX reference

_pending_
