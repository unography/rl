# DreamerV3 DMC parity — what every change does

Blunt reference for the `dreamerv3-jax-parity-dmc` branch. Goal: match danijar's
JAX DreamerV3 `walker_walk` curve. No verbiage; bullets only.

- **Branch = `main` + 18 parity commits (cherry-picked) + DMC commits (this session).**
- Reference code: `danijar/dreamerv3 @ e3f0224`. Paper: Hafner et al. 2023, arXiv:2301.04104.
- JAX config: `../dreamerv3/dreamerv3/configs.yaml`. Reference curves: `../dreamerv3/scores/*.json.gz`.
- Files touched (torchrl): `torchrl/objectives/dreamer_v3.py`, `torchrl/modules/models/model_based_v3.py`, `sota-implementations/dreamer_v3/{dreamer_v3.py,config.yaml,config_dmc.yaml}`.

---

## Part A — the 18 inherited parity changes (vs `main`)

Grouped by area. Each: **what** / why / JAX ref.

### RSSM / world model (`model_based_v3.py`)
- **unimix** — mix 1% uniform into the categorical latent: `probs = 0.99*softmax + 0.01/N`. Stops dead/degenerate classes. JAX `rssm.unimix: 0.01` (configs.yaml:91).
- **Block GRU (`jax_core`)** — GRU weight split into `blocks` groups (block-diagonal), gate `sigmoid(update - 1)` (starts near "keep"), action soft-clip, RMSNorm on inputs. JAX `rssm.blocks: 8`, `.act: silu`, `.norm: rms` (configs.yaml:91).
- **RMSNorm + SiLU** across RSSM heads and world-model MLPs. JAX `norm: rms, act: silu` everywhere.

### Losses (`dreamer_v3.py`)
- **[BugFix] KL free-nats summed over categoricals** — clamp KL to `free_bits` *after* summing over the 32 categoricals, not per-categorical. Per-categorical clamp pins KL at the floor and kills the gradient (`clamp_min` is flat below the floor). JAX sums latent dims then applies `free_nats` (rssm.py loss). JAX `free_nats: 1.0`.
- **[BugFix] reco loss: don't symlog the prediction** — target is `symlog(obs)`, prediction is raw. Compute `(symlog(obs) - pred)^2`, not `(symlog(obs) - symlog(pred))^2`. Double-symlog double-compresses and shrinks the error. JAX `symlog_mse` head: MSE(pred, symlog(target)) (embodied/jax/heads.py).
- **Two-scale KL (dyn/rep)** — separate weights: dynamics KL 1.0, representation KL 0.1 (replaces single `kl_alpha`). JAX `loss_scales: {dyn: 1.0, rep: 0.1}`.
- **Two-hot value + reward (`symexp_twohot`)** — regression as classification over `symexp`-spaced bins; decode = `sum(softmax * bins)`. JAX `output: symexp_twohot, bins: 255`.
- **Slow (EMA) critic + slowreg** — `slowreg` pulls the live value toward an EMA copy. With `slowtar: False`, returns bootstrap from the live critic, not the EMA critic. JAX `slowvalue.rate: 0.02`, `imag_loss.slowreg: 1.0`.
- **Percentile return norm (retnorm)** — divide advantages by a running 5th–95th percentile spread. Scale-free actor updates. JAX `retnorm: {impl: perc, perclo: 5, perchi: 95}`.
- **bounded_normal actor + analytic entropy** — plain Normal with `tanh(mean)` and std in `[minstd, maxstd]`; closed-form entropy for the actent term. Samples are not tanh-squashed and can leave `[-1, 1]`. JAX `policy_dist_cont: bounded_normal, minstd 0.1, maxstd 1.0`.
- **Faithful `imag_loss`** — lambda-return + per-step weight + stop-grads transcribed from JAX (`imag_loss.lam: 0.95`, `actent: 3e-4`).

### Example (`dreamer_v3.py`)
- **[BugFix] recurrent real-env acting policy** — the acting policy encodes obs -> posterior -> action each step and carries the RSSM belief. Without this it acts blind (constant action); nothing learns. See `[[dreamerv3-example-doesnt-learn-pendulum]]`.
- **[Feature] match JAX initial belief at reset** — on `is_init` steps, belief = `prior(0,0,0)`; needs `InitTracker` on the env.
- **[BugFix] share prior/reward with imagination env** — imagination rolls out the *same* trained prior + reward modules, not frozen random copies.
- **DreamerV3 optimizer** — AGC (adaptive grad clip) -> RMS scale -> momentum -> LR warmup. JAX `opt: {lr: 4e-5, agc: 0.3, warmup: 1000, momentum: True}`.
- **Continue (termination) head**, single joint optimizer, and an action coordinate system normalized to `[-1, 1]`. The Torch example does not reproduce JAX's separate action-clipping wrapper.

---

## Part B — the DMC generalization (this session)

Makes the one script run Gym **and** DMC. All in `sota-implementations/dreamer_v3/`.

- **`make_env(cfg, seed)` dispatches on `cfg.env.backend`** (`gym` | `dmc`).
  - `gym`: `GymEnv(cfg.env.name)` — obs already one `observation` vector.
  - `dmc`: `DMControlEnv(cfg.env.domain, cfg.env.task, from_pixels=False)`.
- **DMC obs is multi-key** — walker exposes `orientations`(14) + `height`(1) + `velocity`(9), all float64.
  - **`CatTensors(in_keys=sorted(obs_keys), out_key="observation", del_keys=True)`** concatenates them into one 24-D `observation`. `sorted` = stable order across versions.
  - **`DoubleToFloat()`** — dm_control emits float64; the nets want float32. Skip this and you get a dtype crash.
  - Everything downstream keys on `observation`, so nothing else changed.
- **`ActionScaling()`** — exposes a normalized `[-1,1]` action coordinate system and maps it to the environment range. DMC is already `[-1,1]`, so the map is an identity. It does not clip out-of-range Normal samples; JAX separately uses `ClipAction`, which is missing here.
- **`eval_max_steps` is config-driven** — was hardcoded `max_steps=200` (Pendulum). DMC episodes are 1000.
- **`config.yaml`** gains `env.backend/domain/task`, `logger.eval_max_steps`; defaults unchanged (Pendulum still works).
- **`config_dmc.yaml`** — walker parity config (below).

### `config_dmc.yaml` maps the JAX `dmc_proprio` preset (size1m)
Traced to configs.yaml `dmc_proprio` (line 178) + `size1m` (line 120):

| torchrl field | value | JAX source |
|---|---|---|
| `rnn_hidden_dim` | 512 | `size1m rssm.deter` |
| `num_categoricals` | 32 | `rssm.stoch` |
| `num_classes` | 4 | `size1m rssm.classes` |
| `hidden_dim` | 64 | `size1m rssm.hidden` / `units` |
| `batch_size` / `seq_len` | 16 / 64 | `batch_size` / `batch_length` |
| `num_envs` | 16 | `run.envs` |
| train ratio | 1024 | `dmc_proprio run.train_ratio` |
| `lr` / `agc` / `opt_warmup` | 4e-5 / 0.3 / 1000 | `opt` |
| `imagination_horizon` | 15 | `imag_length` |
| `kl_dyn_scale` / `kl_rep_scale` | 1.0 / 0.1 | `loss_scales.dyn/rep` |
| `free_bits` | 1.0 | `rssm.free_nats` |
| `gamma` / `horizon` | 0.997 / 333 | `horizon: 333` -> `1 - 1/333` |
| `lmbda` | 0.95 | `imag_loss.lam` |
| `unimix` | 0.01 | `rssm.unimix` |
| `actor_minstd/maxstd` | 0.1 / 1.0 | `policy.minstd/maxstd` |

- **train_ratio math**: `train_ratio = updates_per_batch * batch_size * seq_len / frames_per_batch`. With 16 / 16 / 64 / 16 = 1024. So `updates_per_batch = frames_per_batch` gives ratio 1024 (≈1 grad step per env step).
- **Replay layout**: each collector batch is `[environment=16, time=1]`.
  `dim_extend=1` stores it as `[time, environment]`, so one-step writes append
  to 16 independent streams and a sampled 64-step window stays in one stream.
- **Warm-up**: `2 * 16 * 64 = 2048` collected transitions, which approximates
  1024 valid sequence starts across 16 streams.
- **`use_reinforce: false`** — this flag selects the old actor-loss path only. With `imag_loss: true`, the port follows JAX `imag_loss`, which uses stopped imagined features and a log-probability policy loss. It does not backpropagate the policy loss through the dynamics.
- **Verified config mappings**: reward bins/count, per-head MLP depths,
  `use_reinforce`, and `contdisc`. The latter now scales the continue target by
  `1 - 1/horizon`, not just the imagination discount.

### V3-off ablation (`config_dmc_v3off.yaml`)
- Isolates the **V3 feature set**. Keeps the acting-policy fix + the KL/reco loss
  bugfixes (working, correct agent); turns off the V3 architecture/algorithm.
- Toggle flags (default = V3-on): `networks.value_head` (twohot|scalar),
  `networks.actor_dist` (bounded|tanh), `optimization.slow_value` (EMA target),
  `optimization.retnorm`. Plus existing: `unimix`, `jax_core`, `optimizer`,
  `imag_loss`, `use_reinforce`, `kl_rep_scale`.
- Library support: scalar critic = actor loss `num_value_bins=None` + value loss
  `symlog_mse`; no-EMA = `slow_value_model=None`, `slowreg=0`; retnorm off =
  `normalize_returns=False`; tanh actor uses sampled (not analytic) entropy.
- Expectation: **learns but underperforms** the parity arm -> the gap is the V3
  features' contribution. Run: `run_dmc_parity.py --arm v3off config_dmc_v3off`.

---

## Part C — architecture + loss parity (A6000 session)

The JAX reference now runs on the same box, which made two sharper checks
possible than reading config tables.

### The parameter budget is the strongest architecture check
JAX prints per-module parameter counts at startup. Matching each module count is
a strong check for the reviewed widths, layer counts, and input wiring. Counts
alone cannot prove that two architectures are identical. torchrl was at
2,209,059 vs the reference's 640,867. Run `scripts/check_param_parity.py` after
any model change.

- **Decoder input** — decode from `[stoch, deter]`, not `stoch` alone. The
  belief carries most of the information; decoding without it makes the
  posterior do double duty and inflates the rep KL.
- **`rssm.hidden` vs `rssm.deter`** — `hidden` (64) is the width of the
  prior/posterior hidden layers and the three GRU input projections; `deter`
  (512) is only the recurrent state. Confusing them inflated the RSSM 5.3x.
- **`imglayers: 2`** — the dynamics predictor has two hidden layers, not one.
- **`rewhead.layers: 1` / `conhead.layers: 1`** — `size1m` only overrides
  `units`, not `layers`; only enc/dec/policy/value have 3.
- **The encoder is the MLP trunk** — its output is the last hidden activation
  (normed + activated), not a further linear projection.
- **`outscale`** — reward and value output layers zero-initialized, policy
  scaled by 0.01. Combined with the symmetric two-hot grid, the reward and value
  predictions are *exactly* 0 at init, which is a checkable invariant.
- **`winit: trunc_normal_in`** — trunc-normal on [-2,2] times
  `1.1368*sqrt(1/fan_in)`, zero bias. torch's default uniform is 1.7x narrower
  with a non-zero bias. `BlockLinear`'s fan_in is `in_per * blocks`. The example's
  `_apply_jax_init` also reinitializes the block layers in the RSSM prior.

### Loss bugs the parameter check could not see
- **continue target is `1 - is_terminal`, not `1 - done`** — DMC truncates at
  1000 steps (`done=True, terminated=False`); training the continue head on
  `done` teaches it that time limits are terminations.
- **`symexp_twohot` works in reward space** — bins at
  `symexp(linspace(-20,20,N))`, two-hot interpolation *and* expectation taken
  there, so the prediction is `E[reward]`. The paper's symlog-space version
  decodes `symexp(E[symlog r])`. Same bin centers, different decode for a
  spread-out distribution. `bin_space` selects which.
- **decode the two-hot symmetrically** — reward-space bins reach ~5e8, so a
  left-to-right `sum(p*b)` leaves ~0.3 of rounding error where the answer should
  be exactly 0. The reference pairs bin `-k` with `+k` before summing.
- **`repval`** — the critic is *also* trained on the real replay sequences
  (scale 0.3), bootstrapping off the imagination returns, with the gradient
  flowing back into the world model (`repval_grad: True`).

### Throughput: it is dispatch, not FLOPs
GPU at 4%, one core pinned. ~35 tiny kernels per RSSM step at ~15-50 us of
dispatch each. Removing the per-timestep TensorDict work and skipping the
discarded prior sample took the world-model forward 314 -> 217 ms;
`compile_scan()` (unrolled `torch.compile`) takes it to 65 ms. CUDA graphs
(`mode="reduce-overhead"`) were slower, not faster.

## Key concepts (one line each)
- **symlog / symexp** — `symlog(x)=sign(x)ln(1+|x|)`; compresses wide-range targets. `symexp` inverts it. (nets.py:59)
- **two-hot** — encode a scalar as weights on the two nearest bins; regression-as-classification. Stable for rewards/values.
- **free nats / free-bits** — don't penalize KL below a floor; below it, `clamp_min` zeroes the gradient (the KL-bug trap).
- **unimix** — uniform mixture on categoricals; floors class probabilities.
- **retnorm** — percentile advantage normalization; scale-free policy gradient.
- **AGC** — clip grad by ratio to the parameter norm; DreamerV3's optimizer front-end.
- **block GRU** — block-diagonal recurrent weight + `sigmoid(update-1)` gate; DreamerV3's `jax_core` recurrence.

## References
- Paper: https://arxiv.org/abs/2301.04104
- JAX config: `../dreamerv3/dreamerv3/configs.yaml` (defaults + `size1m` + `dmc_proprio`)
- JAX curves: `../dreamerv3/scores/dmc_proprio-dreamerv3.json.gz`
- Deep dives (Pendulum branch): `dreamerv3-parity-notes/DETAILED_ANALYSIS.md`, `ACTING_POLICY_FIX.md`
