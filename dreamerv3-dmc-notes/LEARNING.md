# DreamerV3 DMC parity — what every change does

Blunt reference for the `dreamerv3-jax-parity-dmc` branch. Goal: match danijar's
JAX DreamerV3 `walker_walk` curve. No verbiage; bullets only.

- **Branch = `main` + 18 parity commits (cherry-picked) + DMC commits (this session).**
- Reference code: `danijar/dreamerv3 @ e3f0224`. Paper: Hafner et al. 2023, arXiv:2301.04104.
- JAX config: `_ref/dreamerv3/dreamerv3/configs.yaml`. Reference curves: `_ref/dreamerv3/scores/*.json.gz`.
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
- **Slow (EMA) target critic + slowreg** — value bootstraps off an EMA copy; `slowreg` pulls live value toward it. JAX `slowvalue.rate: 0.02`, `imag_loss.slowreg: 1.0`.
- **Percentile return norm (retnorm)** — divide advantages by a running 5th–95th percentile spread. Scale-free actor updates. JAX `retnorm: {impl: perc, perclo: 5, perchi: 95}`.
- **bounded_normal actor + analytic entropy** — `tanh`-squashed Normal, std in `[minstd, maxstd]`; closed-form entropy for the actent term. JAX `policy_dist_cont: bounded_normal, minstd 0.1, maxstd 1.0`.
- **Faithful `imag_loss`** — lambda-return + per-step weight + stop-grads transcribed from JAX (`imag_loss.lam: 0.95`, `actent: 3e-4`).

### Example (`dreamer_v3.py`)
- **[BugFix] recurrent real-env acting policy** — the acting policy encodes obs -> posterior -> action each step and carries the RSSM belief. Without this it acts blind (constant action); nothing learns. See `[[dreamerv3-example-doesnt-learn-pendulum]]`.
- **[Feature] match JAX initial belief at reset** — on `is_init` steps, belief = `prior(0,0,0)`; needs `InitTracker` on the env.
- **[BugFix] share prior/reward with imagination env** — imagination rolls out the *same* trained prior + reward modules, not frozen random copies.
- **DreamerV3 optimizer** — AGC (adaptive grad clip) -> RMS scale -> momentum -> LR warmup. JAX `opt: {lr: 4e-5, agc: 0.3, warmup: 1000, momentum: True}`.
- **Continue (termination) head**, single joint optimizer, action space normalized to [-1, 1].

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
- **`ActionScaling()`** — normalizes policy [-1,1] to env action range. DMC is already [-1,1] -> identity. Pendulum [-2,2] -> real rescale.
- **`eval_max_steps` is config-driven** — was hardcoded `max_steps=200` (Pendulum). DMC episodes are 1000.
- **`config.yaml`** gains `env.backend/domain/task`, `logger.eval_max_steps`; defaults unchanged (Pendulum still works).
- **`config_dmc.yaml`** — walker parity config (below).

### `config_dmc.yaml` = JAX `dmc_proprio` preset (size1m)
Traced to configs.yaml `dmc_proprio` (line 178) + `size1m` (line 120):

| torchrl field | value | JAX source |
|---|---|---|
| `rnn_hidden_dim` | 512 | `size1m rssm.deter` |
| `num_categoricals` | 32 | `rssm.stoch` |
| `num_classes` | 4 | `size1m rssm.classes` |
| `hidden_dim` | 64 | `size1m rssm.hidden` / `units` |
| `batch_size` / `seq_len` | 16 / 64 | `batch_size` / `batch_length` |
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
- **`use_reinforce: false`** — walker is continuous -> DreamerV3 backprops the actor through the dynamics (JAX `reward_grad: True`), not REINFORCE. **VERIFY** on GPU.
- **VERIFY tags** (see plan doc): reward bins count/spacing (JAX 255 vs example 41), MLP `depth` mapping, `use_reinforce`, `contdisc`.

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
- JAX config: `_ref/dreamerv3/dreamerv3/configs.yaml` (defaults + `size1m` + `dmc_proprio`)
- JAX curves: `_ref/dreamerv3/scores/dmc_proprio-dreamerv3.json.gz`
- Deep dives (Pendulum branch): `dreamerv3-parity-notes/DETAILED_ANALYSIS.md`, `ACTING_POLICY_FIX.md`
