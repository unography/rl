# DreamerV3 `walker_walk` — torchrl vs JAX parity results

Branch `dreamerv3-jax-parity-dmc`. Reference: danijar/dreamerv3 `dmc_proprio`
(size1m) published curve, `reference/dmc_walker_walk_dreamerv3_mean.csv`
(5 seeds). JAX source read directly from a local checkout at `/root/dreamerv3`.

## Minimum A100 evidence (2026-07-26)

Fresh A100 80GB runs using separate dependency-identical Torch worktrees and a
third JAX environment. The control branch is `main` plus DMC and CUDA plumbing;
the CUDA fix is commit `bb58456d` and does not change its algorithm. Detailed
checkpoints, summary, and plot are committed as
`reference/a100_three_seed_comparison.csv`,
`reference/a100_three_seed_summary.csv`, and
`plots/a100_three_seed_returns.png`.

### Return at 10,240 environment steps

| implementation | seeds | mean | median | range | seeds in JAX band |
|---|---|---:|---:|---:|---:|
| JAX published reference (10k) | 5 | 43.02 | -- | 37.14--51.81 | -- |
| torchrl parity | 3 | 75.08 | 49.30 | 45.57--130.36 | 2 / 3 |
| torchrl main control | 3 | 26.49 | 34.32 | 10.55--34.61 | 0 / 3 |

This is the requested minimum evidence. The parity median is 6.28 return points
from the JAX mean versus 8.70 for main, two parity seeds are inside the JAX
range versus zero main seeds, and parity's mean return is 2.83x main's. Seed 7
is a high parity outlier, so the mean overshoots JAX; this is evidence of a real
improvement and closer robust behavior, not yet a full curve-parity claim.

The main control is also qualitatively wrong: it does not train until 8,192
frames, evaluates a blind actor that never consumes the observation, and seed
0's actor loss diverges to -72.9 million at 9,216 then +28.7 million at 10,240.

### Matched 51.2k seed-0 curves and V3-off ablation

The longer seed-0 comparison uses the same A100, dependency-matched worktrees,
51,200-frame budget, 10,000-frame evaluation interval, and one evaluation
episode per checkpoint. The committed source data and rendered plot are
`reference/a100_51k_four_way.csv`,
`reference/a100_51k_four_way_summary.csv`, and
`plots/a100_51k_four_way.png`. Raw logs are
`plots/a100_parity_contdisc_seed0_51200.log`,
`plots/a100_v3off_contdisc_seed0_51200.log`, and, on the control branch,
`dreamerv3-baseline-notes/a100_baseline_seed0_51200.log`.

| approximate step | JAX mean (5 seeds) | parity | main control | V3-off |
|---:|---:|---:|---:|---:|
| 10k | 43.02 | 65.65 | -- | 22.65 |
| 20k | 88.53 | 63.39 | 10.41 | 4.40 |
| 30k | 134.04 | 202.90 | 27.22 | 5.28 |
| 40k | 217.07 | 198.62 | 26.97 | 4.47 |
| 50k | 289.21 | 121.02 | 27.96 | 22.38 |

Checkpoints differ by at most 1,808 frames because main begins training at
8,192 frames while the parity configurations begin at 1,024. Across the four
comparable 20k--50k JAX checkpoints, parity is closer to the JAX mean at every
point. Mean absolute return error is 70.16 for parity, versus 159.07 for main
and 173.08 for V3-off. At 40k, parity is inside the published JAX five-seed
range (198.62 versus 142.68--347.83); main and V3-off remain near zero-return
behavior at 26.97 and 4.47.

V3-off retains the shared acting-policy, KL/reconstruction, and continuous-
discount fixes, but disables the DreamerV3 feature set (two-hot value,
bounded-normal actor, slow value/replay value, return normalization, unimix,
JAX-style recurrent core and optimizer, and imagination objective). It stays
between 4.40 and 22.65 throughout the run. This isolates the improvement from
the plumbing fixes and shows that the parity features are necessary for this
seed's learning curve.

This 51k comparison is deliberately presented as one-seed curve evidence, not
a variance estimate or a claim of complete curve parity. Parity falls below
the JAX range at 50k. The independent three-seed 10k result above is the more
robust minimum-evidence statement; the longer run demonstrates sustained
separation from main and the V3-off ablation.

### Same-seed loss parity at ~10k

Fresh JAX seed 7 at step 10,096 versus compiled torchrl seed 7 at step 10,240:

| term | torchrl | JAX | absolute error |
|---|---:|---:|---:|
| dyn / rep | 6.805 | 6.282 | 0.523 |
| reconstruction sum | 2.451 | 2.646 | 0.195 |
| reward | 0.514 | 0.502 | 0.012 |
| value | 1.092 | 1.193 | 0.101 |
| repval | 1.606 | 1.836 | 0.230 |
| policy | 1.011 | 1.482 | 0.471 |

These evidence runs preceded the final `contdisc` fix, so continue is offset
(`0.0001` vs `0.0205`). The config now uses the JAX path: live continue targets
are scaled by `1 - 1 / horizon` and the actor consumes the learned discount
without applying a second constant. Main cannot produce this loss vector: it
lacks separate dyn/rep losses, continuation, two-hot value, replay value, and
the JAX imagination objective.

Environment versions: Python 3.10.12, torch 2.13.0+cu130, TensorDict
0.13.0+g8f37f8e, dm-control 1.0.43; JAX/JAXlib 0.4.33. Parameter parity was
reproduced independently in both environments: 640,867 total with every module
count matching.

## Hardware / environment

| | |
|---|---|
| GPU | Steps 1-1b: A100-SXM4-80GB. Steps 1c on: **RTX A6000 48GB** (driver 580.126.09) |
| CPU / RAM | A100 box: 22 cores / 117 GB. A6000 box: **6 cores / 47 GB** |
| OS | Ubuntu 22.04.5 |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| torchrl | 0.13.0 editable (C++ ext built for py3.12) |
| tensordict | 0.13.0 |
| mujoco / dm_control | 3.10.0 |
| JAX reference | `danijar/dreamerv3 @ e3f0224`, jax 0.4.33 (cuda12), own venv at `/home/ubuntu/dreamerv3/.venv` |
| headless GL | `MUJOCO_GL=egl` |

The JAX reference is **runnable on the same box** now, so per-term losses can be
regenerated rather than quoted:

```bash
cd /home/ubuntu/dreamerv3 && MUJOCO_GL=egl .venv/bin/python dreamerv3/main.py \
  --configs dmc_proprio --task dmc_walker_walk --logdir <dir> --run.steps 20000 \
  --run.log_every 30 --jax.prealloc False
```

Its per-term losses on this box are committed as
`reference/jax_walker_walk_losses_a6000.csv` and reproduce the A100 numbers
below to within run-to-run noise (e.g. `dyn` 6.53 at 7.6k here vs 6.89 at 7.1k
there).

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

### 4. `contdisc` — initially equivalent for returns, now exact

JAX `contdisc: True` does two things: `agent.py:175-176` scales the continue
**target** by the horizon (`con *= 1 - 1/horizon`), and `imag_loss` then sets
`disc = 1`. torchrl's `contdisc: false` instead sets `disc = 1 - 1/horizon` and
uses the raw continue head. For walker, which never terminates early, `con -> 1`
under both, so JAX's `cumprod(1 * 0.997*con)` and torchrl's
`cumprod(0.997*con)/0.997` agree up to a constant factor of `1/0.997` on the
imagination weights. This was initially kept `false`, but it left the reported
continue BCE at zero rather than JAX's 0.0208 floor. The model loss now exposes
`continue_target_scale`; `config_dmc.yaml` sets `contdisc: true` and passes
`1 - 1/horizon`, matching both JAX target training and imagination discounting.
Post-fix A100 seed-7 smoke at step 2,048 reports `con=0.0243` (the pre-fix run
was `0.0094`) while dyn/reconstruction/reward remain within 1% of the pre-fix
compiled checkpoint. The new continue loss is in JAX's 0.02 regime.

The post-fix seed-7 run was then extended to step 10,240. Against fresh JAX
seed 7 at 10,096: `con 0.0205 / 0.02048`, `reward 0.538 / 0.502`,
`reconstruction 2.589 / 2.646`, `repval 1.873 / 1.836`, and
`policy 1.211 / 1.482`. Dyn (`7.340 / 6.282`) is now the largest loss offset.
Return is 104.93: the high seed-7 return persists after the fix, so it was not
caused by the continuous-discount approximation.

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

## Step 1b — gaps found by preparing a per-term loss comparison

The VERIFY sweep above compared *config values*. Diffing the **loss terms**
against the reference's `metrics.jsonl` surfaced three further differences that
no return-curve comparison would have localised. All three are fixed.

1. **Encoder input symlog (missing).** JAX squashes every vector observation
   with `symlog` before the encoder MLP (`rssm.py` `SimpleEncoder.__call__`:
   `squish = nn.symlog`). The example symlogged the reconstruction *target*
   correctly but fed the encoder **raw** observations. For walker, `velocity`
   spans far wider than `orientations`/`height`.
2. **Reconstruction under-weighted 24x.** JAX sums the per-key reconstruction
   loss over the observation dims, then averages over batch/time
   (`embodied.jax` `outs.Agg(..., agg=jnp.sum)`; `agent.py:186` asserts every
   loss is `(B, T)`). The example ran `global_average=True`, averaging over the
   observation dims too — so reconstruction carried 1/24 of its intended weight
   against `dyn 1.0` / `rep 0.1`. `global_average=False` was itself unusable:
   it hardcoded `sum((-3,-2,-1))`, correct for `(B,T,C,H,W)` pixels but
   collapsing batch and time for `(B,T,D)` vectors. Now sums event dims
   generically (pixel behavior unchanged).
3. **Replay warmup 8x too long.** JAX trains as soon as replay holds one batch
   (`embodied/run/train.py:71` — `batch_size*batch_length` = 1024 frames).
   `warmup_factor: 8` withheld the first update until 8192 env steps, displacing
   every loss curve by ~7k updates relative to the reference.

Effect of (2) alone: reconstruction loss at init went from `1.15` to `28.7`,
and `kl` now sits in the reference's regime (`dyn + 0.1*rep ~ 6.8`) instead of
well below it.

### JAX reference loss terms (originally measured on an A100, walker_walk)

| env step | dyn | rep | rew | con | policy | value | repval | recon (sum) |
|---|---|---|---|---|---|---|---|---|
| 4432 | 6.06 | 6.06 | 2.34 | 0.125 | 1.11 | 3.20 | 6.39 | 12.10 |
| 7120 | 6.89 | 6.89 | 0.58 | 0.021 | 1.71 | 1.65 | 3.16 | 3.32 |
| 9856 | 6.32 | 6.32 | 0.50 | 0.020 | 1.53 | 1.29 | 1.93 | 2.48 |
| 12576 | 6.21 | 6.21 | 0.50 | 0.020 | 0.89 | 1.26 | 1.51 | 2.31 |

`dyn` and `rep` are numerically identical by construction (`sg` changes
gradients, not values) — a free correctness check for the torchrl side.

### Known remaining gaps vs JAX — all three closed

The three gaps recorded earlier (output-layer `outscale` init, the missing
replay critic loss `repval`, and single-env collection) are addressed below,
along with six further architecture bugs that a parameter-count comparison
surfaced.

## Step 1c — architecture parity via the reference's parameter budget

The JAX process prints an exact per-module parameter budget at startup. That
one table pins down every hidden width, layer count and input wiring in the
`dmc_proprio`/`size1m` preset, and proved a far sharper instrument than
comparing config values: **the example built 2,209,059 parameters against the
reference's 640,867** -- 3.4x oversized overall, with the decoder 2.8x
*under*sized.

| module | torchrl (before) | torchrl (after) | JAX |
|---|---|---|---|
| dyn (RSSM) | 1,944,320 | 364,416 | 364,416 |
| val | 66,111 | 66,111 | 66,111 |
| rew | 66,111 | 57,663 | 57,663 |
| dec | 18,328 | 51,096 | 51,096 |
| pol | 50,316 | 50,316 | 50,316 |
| con | 49,601 | 41,153 | 41,153 |
| enc | 14,272 | 10,112 | 10,112 |
| **total** | **2,209,059** | **640,867** | **640,867** |

Verify with `scripts/check_param_parity.py` (exits non-zero on any mismatch).

The bugs behind those deltas:

1. **Decoder read only the stochastic latent.** JAX decodes from the full model
   state, `concat([stoch, deter])` (`rssm.py` `Decoder.__call__`). Reconstructing
   from `stoch` alone forces the posterior to re-encode everything the belief
   already carries, which inflates the representation KL.
2. **Prior/posterior hidden width was `deter` (512), not `hidden` (64).** JAX's
   `rssm.hidden` sets the width of the prior/posterior hidden layers *and* the
   three GRU input projections; `rssm.deter` is only the recurrent state width.
   This single bug accounted for nearly all of the excess.
3. **Prior head had 1 hidden layer**; JAX `rssm.imglayers: 2`.
4. **Reward/continue heads had 3 hidden layers**; JAX `rewhead.layers: 1`.
5. **Encoder had an extra output linear.** The JAX encoder *is* the MLP trunk --
   its embedding is the last hidden activation, normed and activated.
6. **`outscale`**: reward and value output layers are zero-initialized and the
   policy's is scaled by `0.01`. With the symmetric two-hot grid this makes both
   heads predict *exactly* 0 at init.
7. **`winit`**: JAX draws `trunc_normal_in` -- truncated normal on `[-2, 2]`
   scaled by `1.1368 * sqrt(1/fan_in)` (the constant undoes the truncation's
   variance shrinkage), with zero biases. torch's default is
   `U(-1/sqrt(fan_in), +1/sqrt(fan_in))` for weights *and* biases: 1.7x
   narrower, non-zero bias. For `BlockLinear`, JAX's `compute_fans` gives
   `fan_in = in_per * blocks`, i.e. the total input width.
8. **RMSNorm `eps`**: JAX uses `1e-4`; torch defaults to the dtype epsilon
   (~1e-7).

## Step 1d — further loss bugs

1. **Continue target used `done` instead of `terminated`.** JAX trains the
   continue head against `1 - is_terminal` (`agent.py:174`). A DMC episode ends
   by *truncation* every 1000 steps (`done=True, terminated=False`, confirmed on
   this box), so the example was teaching the model "the episode ends here" at
   every time limit, and the imagination discount inherited it.
2. **Two-hot space.** JAX's `symexp_twohot` head places the bins at
   `symexp(linspace(-20, 20, N))` and does *both* the two-hot interpolation and
   the decode in **reward** space, so its prediction is `E[reward]`. The paper's
   formulation -- what the example implemented -- works in symlog space and
   decodes `symexp(E[symlog reward])`. The bin *centers* coincide (as Step 1
   recorded), but a spread-out distribution decodes differently, and early in
   training the critic's distribution is very spread out. Now selectable via
   `bin_space`, set to `reward` in `config_dmc.yaml`.
3. **Two-hot decode was numerically unsound on that grid.** Over reward-space
   bins the extremes reach `symexp(20) ~ 4.85e8`, and a left-to-right sum of
   `probs * bins` does not cancel: a uniform distribution decoded to `0.32`
   instead of `0`. That is exactly the state of a zero-initialized head, and the
   reason the reference sums symmetrically (`outs.TwoHot.pred`, with a comment
   saying so). `two_hot_decode` now does the same.
4. **Replay critic loss (`repval`)**, now implemented as
   `DreamerV3ValueLoss.replay_value_loss` (JAX `repl_loss`,
   `loss_scales.repval: 0.3`): the critic is also trained along the real replay
   sequences, with the imagination lambda-returns as the per-step bootstrap.
   `repval_grad: True` means the gradient is *not* stopped at the world-model
   features, so `DreamerV3ModelLoss` gained `detach_output=False` to expose the
   live output.
5. **Dead deprecation check.** `DreamerV3ActorLoss`'s `gamma`/`lmbda` guards sat
   after a `return` in `_decode_value` and never fired.

### Verified as already matching (no change needed)

retnorm (`perc`, rate .01, limit 1.0, 5/95, no debias), the optimizer chain
(AGC -> RMS with bias correction -> momentum with bias correction -> warmup),
the imagination lambda-return, the slow-critic EMA (`rate .02, every 1`,
initialized as a copy of the online critic), free-nats on the summed KL, unimix
in the sampler *and* in the KL, and the two-scale dyn/rep weighting.

### Residual differences (documented, not fixed)

1. **Parallel envs.** `collector.num_envs` now builds a `SerialEnv`, but the
   parity arm keeps 1. torchrl's `ndim=2` storage appends a row per worker per
   batch and `SliceSampler` cannot cut a slice across rows, so multi-env
   collection needs `frames_per_batch >= num_envs * seq_len` -- trading the
   reference's train-after-every-step interleaving for the replay diversity.
   Validated that the resulting slices are single-trajectory and time-contiguous,
   so the option is usable; which side of that trade is better is a judgement
   call, not a bug.
2. **Replay carry.** JAX streams *consecutive* chunks and carries the RSSM state
   across them (`replay_context: 1`, `_apply_replay_context`); torchrl samples a
   random 64-step window and starts it from a zero belief. Worth quantifying --
   it should inflate the loss on the first steps of each sampled sequence.
3. **Compute dtype.** JAX runs the train step in bfloat16
   (`jax.compute_dtype`); torchrl runs fp32.
4. **`grad_clip: 100`** global-norm clip in the example has no JAX counterpart.
   It sits after AGC and should never bind.

## Throughput (measured on the A6000 box)

The run is **launch-bound, not compute-bound**: GPU utilization sat at 4% with
one CPU core pinned at 100%. On this box a trivial CUDA op costs ~15 us to
dispatch and an `nn.Linear` ~50 us, so a 64-step recurrence of ~35 tiny kernels
per step is essentially all dispatch overhead.

| world-model forward (B=16, T=64) | ms |
|---|---|
| baseline | 314 |
| tensor-only rollout (no per-step TensorDict) | 217 |
| + `compile_scan()` (unrolled `torch.compile`) | 65 |

In order of size:

- `RSSMRolloutV3` gained a tensor-only fast path. The generic loop paid a module
  dispatch, a `select` and two `set` calls per timestep, plus a stack of `T`
  tensordicts.
- The prior's sampled latent is no longer drawn during the observation pass (the
  posterior overwrites it, and the reference does not draw one either), and the
  straight-through sampler uses Gumbel-max rather than constructing a
  `torch.distributions.Categorical` per step.
- `BlockLinear` computes its block-diagonal product with `bmm` over the block
  axis. An `einsum` -> broadcast-`matmul` swap was tried here and reverted: it
  bought no measurable time and cost 27 GiB of peak memory (see Memory below).
- `RSSMRolloutV3.compile_scan()` (`optimization.compile_rssm=true`) compiles the
  unrolled recurrence for a further 3x on this microbenchmark, at ~8.5 min of
  one-off compile (forward *and* backward through the 64-step unrolled graph):
  worth it for a real run, not for a smoke test. See "compile_rssm validation"
  below for end-to-end numbers, which are smaller than 3x.
  `mode="reduce-overhead"` (CUDA graphs) was tried and came out *slower*
  (5.2 ms/step vs 2.3), so it is not used.

A separate O(buffer_size) rescan in `SliceSampler._get_stop_and_length` was
fixed earlier (`cache_values=True`).

For scale: the JAX process reaches ~39-57 env steps/s on this same box
(`fps/policy`), compiling the whole train step into one XLA executable at bf16.

## Step 2 — loss parity vs the JAX reference (measured)

Single seed, `config_dmc` on the A6000, against `reference/jax_walker_walk_losses_a6000.csv`
(the same reference run on the same box). torchrl's eval cadence is 1008 env
steps and JAX's is 1184, so each row pairs the nearest available JAX point.

| env step | dyn (t / j) | recon (t / j) | rew (t / j) | value (t / j) | repval (t / j) | torch eval |
|---|---|---|---|---|---|---|
| 1024 | 7.93 / 5.01 | 38.2 / 23.5 | 5.54 / 4.79 | 1.29 / 4.02 | 11.1 / 9.77 | +22.2 |
| 2032 | 5.46 / 5.01 | 9.49 / 23.5 | 1.98 / 4.79 | 5.13 / 4.02 | 5.70 / 9.77 | +22.8 |
| 3040 | 6.99 / 5.01 | 4.70 / 23.5 | 0.65 / 4.79 | 2.34 / 4.02 | 4.24 / 9.77 | +9.9 |
| 4048 | 6.88 / 6.40 | 3.60 / 7.06 | 0.65 / 1.28 | 1.98 / 3.00 | 4.11 / 4.91 | +9.9 |
| 5056 | 6.42 / 7.02 | 2.99 / 4.09 | 0.45 / 0.60 | 1.50 / 1.88 | 2.85 / 3.79 | +22.5 |
| 6064 | 8.11 / 6.76 | 3.87 / 3.29 | 0.50 / 0.54 | 1.49 / 1.58 | 2.68 / 3.03 | +31.6 |
| 7072 | 7.21 / 6.53 | 2.83 / 2.90 | 0.47 / 0.51 | 1.49 / 1.40 | 2.27 / 2.49 | +33.6 |
| 8080 | 6.71 / 6.53 | 2.97 / 2.90 | 0.68 / 0.51 | 1.27 / 1.40 | 2.57 / 2.49 | +45.9 |
| 9088 | 6.22 / 6.33 | 2.45 / 2.71 | 0.51 / 0.50 | 1.43 / 1.33 | 1.81 / 2.11 | +53.0 |
| 10096 | 6.70 / 6.21 | 2.65 / 2.57 | 0.52 / 0.50 | 1.03 / 1.19 | 1.61 / 1.73 | +50.3 |
| 11104 | 6.75 / 6.09 | 2.25 / 2.44 | 0.56 / 0.50 | 1.40 / 1.18 | 1.85 / 1.57 | +55.2 |

**From ~5k steps on, every term is within ~10-20% of the reference and tracking
its trajectory.** For contrast, the same table before Steps 1c/1d had `dyn` at
8-9 against 5-7, `recon` at 36-38 against 23, and `rew` pinned at 5.541 =
`ln(255)` -- a uniform reward head that never moved, because the zero-init and
the reward-space two-hot were both missing.

Two systematic offsets were observed in these pre-`contdisc`-fix runs:

- **`con` was lower than JAX's** (1e-4 vs 0.02). This is now fixed by scaling
  the continue target to `1 - 1/horizon = 0.997`, whose BCE floor is 0.0208
  nats, and enabling the matching `contdisc` actor path. A post-fix long curve
  is pending.
- **The first ~4k steps do not line up.** torchrl's reconstruction and reward
  losses fall faster than the reference's. Both start from a zero-init head, so
  the difference is in what happens between the first update and ~4k -- the
  replay-carry difference (residual difference 2) is the prime suspect, since a
  zero belief at the start of every sampled window is exactly a
  transient-dominated regime.

Eval return over the same window: +22 -> +55 by 11k. The reference's published
curve begins at 10k with a 5-seed mean of **43.0** (range 37.1-51.8), so this
single seed is at or slightly above the band's start. Not evidence of parity by
itself -- the acceptance table starts at 50k -- but the shape is right.

### Memory

Peak GPU memory was 28.2 GiB for this config, from `BlockLinear` computing its
block-diagonal product as a broadcast `matmul`, which materializes the expanded
weight (537 MB per call at the imagination batch). Using `bmm` over the block
axis gives bit-identical losses at **1.10 GiB** peak. Separately, the replay
buffer took its device from the collector and preallocated 1e6 steps of
`state`/`belief` on the GPU; it now lives in host memory.

## `compile_rssm` validation

`optimization.compile_rssm=true` is safe to use for the long runs. Two checks,
both on this box with the `config_dmc` shapes (B=16, T=64).

**Unit level** — `RSSMRolloutV3._scan` eager vs compiled, with the categorical
sample replaced by a deterministic argmax so RNG is not a confound:

| | worst relative deviation |
|---|---|
| forward outputs (6 tensors) | 4.2e-07 |
| parameter gradients (27 tensors) | 2.4e-07 |
| loss scalar | equal to 8 d.p. (1.22250509) |

`states_in`/`next_states` are bit-identical; deviations are confined to `belief`
and the logits and are ordinary fp32 reassociation from inductor's fusion.

Under *real* sampling, `torch.compile` reorders the RNG draws, so a compiled run
is **not** bit-reproducible against an eager run at the same seed -- only
statistically equivalent (prior-logit mean -0.0096 vs -0.0099, std within 0.3%
over 8 draws). RNG-free quantities at t=0 still match to 7e-07.

**Training level** — seed-matched 2048-step runs, `env.seed=7`:

| env step | kl | dyn | reco | reward | value | repval | policy |
|---|---|---|---|---|---|---|---|
| 1024 eager | 6.816 | 6.197 | 38.299 | 5.541 | 0.960 | 11.082 | -0.000 |
| 1024 compiled | 6.762 | 6.148 | 38.098 | 5.541 | 0.951 | 11.082 | -0.000 |
| 1536 eager | 5.000 | 4.545 | 21.950 | 4.515 | 7.452 | 9.262 | 0.036 |
| 1536 compiled | 5.006 | 4.551 | 22.209 | 4.519 | 7.419 | 9.252 | 0.032 |
| 2048 eager | 6.465 | 5.878 | 9.621 | 1.928 | 3.556 | 5.234 | 1.237 |
| 2048 compiled | 6.527 | 5.934 | 9.436 | 1.925 | 3.546 | 5.105 | 0.942 |

All world-model and critic terms agree to <=2.5%, consistent with the RNG
reordering. The exception is `policy` at 2048 (0.942 vs 1.237, 24%): a
small-scale loss computed from sampled imagination trajectories, so it is the
term most exposed to RNG divergence. The unit check shows its gradient path is
exact under matched sampling, but this is worth re-checking on a longer run
before reading anything into a compiled actor curve.

**Cost/benefit.** Compile is a net *loss* on short runs and a large win on long
ones:

| | eager | compiled |
|---|---|---|
| per 512 env steps (steady state) | 357 s | 214 s (1.67x) |
| one-off compile | -- | ~400 s |
| total, 2048-step run | 768 s | 883 s (slower) |

Break-even is ~1400 training steps, i.e. ~2500 total env steps given the
1024-step warmup. Projected per seed at 500k env steps: **~97 h eager vs ~58 h
compiled**, so ~39 h saved per seed.

Note the end-to-end speedup is 1.67x, not the 3x measured on the world-model
forward in isolation -- the compiled rollout is only one part of each update.

## Step 3 — shakeout

_pending_

## Step 4 — full grid

_pending_

## Acceptance vs JAX reference

_pending_
