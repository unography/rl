# DreamerV3 walker_walk: TorchRL vs JAX parity — status and resume guide

Last updated: 2026-08-17 08:00 UTC. Branch `fix/dreamerv3-gpu-usage`, all work
uncommitted. This file is the handoff: read it end-to-end before resuming.

**Goal.** Make `sota-implementations/dreamer_v3` reproduce the JAX reference at
`/home/ubuntu/dreamerv3` on DMC walker_walk, both in internals and in the
learning curve.

---

## 1. Where things stand

Component-level fidelity is **verified**. The end-to-end learning curve is
**still open** — the comparison has only ever been run out to 48k env steps,
which is too early to distinguish the two.

### Verified: forward numerics match the reference

`scripts/numeric_check.py` loads the JAX checkpoint's weights directly into the
TorchRL modules and compares forward passes in float64. All agree to float32
precision:

| component | max abs diff | value scale |
|---|---|---|
| block-GRU core | 1.3e-07 | 0.46 |
| prior logits | 8.7e-07 | 2.13 |
| encoder | 3.4e-07 | 0.45 |
| posterior logits | 4.0e-06 | 3.37 |
| decoder | 4.1e-07 | 0.78 |
| reward logits | 5.2e-06 | 48.2 |
| policy mean | 4.4e-07 | 0.80 |
| policy std | 1.3e-08 | 0.13 |

`scripts/cmp.py` separately checks the return math against a transcription of
the JAX `lambda_return`: lambda return 9.5e-07, discount weight 8.9e-08,
advantage 9.5e-07, replay-value target exactly 0.

Parameter count matches the JAX `size1m` config at 640,867 trainable params
(asserted by `test_dreamer_v3_dmc_parameter_parity`).

### Verified: update cadence matches

Both sides run exactly 1.0 learner update per env step after warmup
(`train_ratio: 1024` / `batch 16 x seq 64`). Measured slopes: JAX 0.9976,
Torch 1.0010.

### Open: the learning curve, and whether three seeds can settle it

The three-seed JAX reference band to 200k is complete and committed under
`results/jax-seed{0,1,2}/`. Mean episode return per 16k bucket:

| step | s0 | s1 | s2 | median | spread |
|---|---|---|---|---|---|
| 16000 | 48.5 | 37.4 | 48.4 | 48.4 | 11 |
| 32000 | 84.2 | 87.2 | 98.6 | 87.2 | 14 |
| 48000 | 156.6 | 184.4 | 155.6 | 156.6 | 29 |
| 64000 | 229.1 | 184.1 | 146.6 | 184.1 | 83 |
| 80000 | 216.7 | 213.7 | 149.8 | 213.7 | 67 |
| 96000 | 272.7 | 301.5 | 98.0 | 272.7 | 204 |
| 112000 | 282.1 | 374.7 | 54.3 | 282.1 | 320 |
| 128000 | 396.4 | 436.8 | 110.2 | 396.4 | 327 |
| 144000 | 465.9 | 478.0 | 274.3 | 465.9 | 204 |
| 160000 | 463.0 | 514.6 | 232.5 | 463.0 | 282 |
| 176000 | 511.1 | 637.7 | 241.0 | 511.1 | 397 |
| 192000 | 573.6 | 690.8 | 417.8 | 573.6 | 273 |

**The reference disagrees with itself far more than any Torch/JAX gap observed
so far.** Seed 2 plateaus near 150 from 48k to 80k, collapses to 54 by 112k --
below where it stood at 32k -- and only recovers to 418. Seeds 0 and 1 rise
monotonically to 574 and 691. The spread reaches 397 at 176k.

This sets a hard resolution limit. At 192k the three seeds have mean 561 and
standard deviation ~137. Detecting a 100-return difference between two such
distributions at n=3 is hopeless; it needs roughly 30 seeds per side. **A
three-seed Torch run can only falsify gross divergence, not establish parity.**

Consequences for how to judge this work:

- Treat a Torch band that overlaps the JAX band as "not contradicted", never as
  "matches". State the resolution limit whenever quoting the comparison.
- The per-term `train/*` diagnostics (section 5c) are far more sensitive than
  the return curve and already agree within noise. They, plus the exact forward
  numerics, are the real parity evidence. The curve is a smoke test.
- If a return-level answer is genuinely needed, the budget is better spent on
  many seeds at a shorter horizon than on three seeds at 200k.

The stopped Torch runs reached 64k. Their medians against the band: 55.5 vs
[37.4, 48.5] at 16k and 116.0 vs [84.2, 98.6] at 32k, both above; then 77.8 vs
[155.6, 184.4] at 48k and 142.2 vs [146.6, 229.1] at 64k, both below. The 48k
point is the only one that is clearly outside, and Torch's own six runs there
span 76.5 to 139.6 against JAX's six spanning 115.7 to 233.5 -- overlapping
tails. Nothing is established either way.

For reference, the original 1.1M-budget seed-0 run continues past this range,
reaching ~700 by 224k and plateauing near 960 after ~320k.

---

## 2. Runs: JAX reference done, Torch not yet relaunched

**Nothing is running. The GPU is free.**

`scripts/run_jax_serial.sh 200000 0 1 2` ran the JAX reference for seeds 0, 1
and 2 to 200k env steps, one seed at a time, between 09:38 and 14:34 UTC. Each
took 99 minutes at ~33.7 env-steps/s and exited 0.

| seed | final step | scores | logs | commit |
|---|---|---|---|---|
| 0 | 196,208 | 192 | `results/jax-seed0/` | `fab3cba7` |
| 1 | 196,224 | 192 | `results/jax-seed1/` | `627d4bc0` |
| 2 | 196,640 | 192 | `results/jax-seed2/` | `3d50e446` |

Each directory holds `metrics.jsonl`, `scores.jsonl` and `config.yaml` (~136 KB
per seed). `ckpt/`, `replay/`, `plugins/` and `scope/` were deliberately not
committed: the first three are large binaries and `scope/` only re-encodes
scalars `metrics.jsonl` already carries. The full run directories remain in
`/tmp/jaxrun200k-seed{0,1,2}` until the box is rebooted.

Runs stop a little short of the budget (196.2k of 200k) because the JAX driver
advances in blocks of 10 steps across 16 envs and exits on the first check past
the target. The original reference run behaves the same way. Bucket boundaries
still line up with Torch's; only the endpoints differ.

Running serially reproduced the reference's own conditions (its curve came from
a solo run) and let preallocation stay at the reference default of `true`
instead of the `noprealloc` workaround the earlier concurrent runs needed.

**Next step:** relaunch the Torch runs with `scripts/launch.sh torch`, reading
section 1 first — a three-seed Torch band cannot establish parity, only rule out
gross divergence.

### Earlier runs, all stopped and preserved

| runs | reached | kept at |
|---|---|---|
| torch seeds 0/1/2 (200k budget) | step 67,584, 64 episodes each | `/tmp/dv3-partial-seed{0,1,2}` |
| jax seeds 1/2 (200k budget) | step 44,944 / 39,136, 32 episodes each | `/tmp/jaxrun-partial-seed{1,2}` |
| torch seeds 0/1/2 (50k budget) | step 49,936, complete | `/tmp/dv3-seed{0,1,2}` |
| jax seeds 1/2 (50k budget) | step 48,048, complete | `/tmp/jaxrun-seed{1,2}` |

`compare.py` reads all of them, so no comparison data was lost when the runs
were stopped. The Torch 200k runs will need re-launching from scratch
(`scripts/launch.sh torch`) once the JAX seeds finish.

Throughput measured on this A100, same walker config throughout. Steady-state
rates are taken between two consecutive `train` rows; cumulative rates include
startup and `torch.compile` warmup and so understate the sustained rate:

| configuration | per run | aggregate | basis |
|---|---|---|---|
| 3 torch parallel | **14.4 steps/s** | **43.2** | steady state |
| 3 torch + 2 jax | 5.2 steps/s | 15.6 torch + 21.2 jax = 36.8 | steady state |
| 1 torch alone | 22.7 steps/s | 22.7 | cumulative, 704s run |
| 3 torch parallel | 10.3 steps/s | 30.9 | cumulative, 50k runs |
| 1 jax alone (prealloc on) | 33.9 steps/s | 33.9 | run mean |

Two conclusions, both measured rather than predicted:

- **Do not serialise the Torch seeds.** The workload is latency-bound on small
  kernels (memory-bandwidth utilisation is 1-4%), so concurrent runs fill each
  other's gaps: three seeds together sustain 43.2 aggregate steps/s against
  22.7 for one at a time.
- **Do not mix the two sides.** Five concurrent runs oversubscribe the GPU and
  cost aggregate throughput, not just ordering: 36.8 steps/s mixed against 43.2
  for three Torch runs alone. Run the Torch seeds, then the JAX seeds.

Memory: a JAX run with the reference default `prealloc: true` takes ~61 GB of
the 80 GB device; the concurrent runs passed `noprealloc` and held ~1.8 GB
instead. Torch's ~14 GB is mostly the 5M-record replay buffer, which
`config_dmc_walker.yaml` places on the learner device.

Torch logs a `train` row every 4096 steps, so ~8 minutes can pass between rows;
that is not a stall. Episode-return rows only start at 16k (one 1000-step
episode across 16 envs). JAX writes `scores.jsonl` only from 16k as well.

**Caveat: outputs are in `/tmp` and do not survive a reboot.** If the runs
finish, copy the metrics out before doing anything else:

```bash
mkdir -p dreamerv3-parity/results
for d in /tmp/dv3-200k-seed* /tmp/jaxrun200k-seed* /tmp/jaxrun-partial-seed*; do
  cp -r "$d" dreamerv3-parity/results/
done
```

### Checking progress

```bash
# which seed the serial JAX driver is on, and what has been copied in
cat dreamerv3-parity/logs/jax-serial-driver.log
# current seed's step count
for s in 0 1 2; do tail -1 /tmp/jaxrun200k-seed$s/metrics.jsonl 2>/dev/null; done
# torch, when its runs are back
for s in 0 1 2; do
  grep '"type": "train"' /tmp/dv3-200k-seed$s/metrics.jsonl | tail -1
done
# alive?
pgrep -af "run_jax_seed.py|dv3-200k-seed"
```

### If a run died

JAX runs **race on startup**: launching two at once kills one silently with an
empty log. `scripts/launch.sh` now staggers them by 60s. To restart one:

```bash
rm -rf /tmp/jaxrun200k-seed2
MUJOCO_GL=egl nohup /home/ubuntu/dreamerv3/.venv/bin/python -u \
  dreamerv3-parity/scripts/run_jax_seed.py 2 200000 /tmp/jaxrun200k-seed2 noprealloc \
  > dreamerv3-parity/logs/jax-seed2.log 2>&1 &
```

Torch:

```bash
MUJOCO_GL=egl nohup .venv/bin/python sota-implementations/dreamer_v3/dreamer_v3.py \
  --config-name=config_dmc_walker env.seed=0 collector.total_frames=200000 \
  logger.train_every=4096 logger.diagnostics=true \
  logger.metrics_json=/tmp/dv3-200k-seed0/metrics.json \
  logger.metrics_jsonl=/tmp/dv3-200k-seed0/metrics.jsonl \
  hydra.run.dir=/tmp/dv3-200k-seed0/hydra \
  > dreamerv3-parity/logs/torch-seed0.log 2>&1 &
```

---

## 3. Analysing the results

```bash
.venv/bin/python dreamerv3-parity/scripts/compare.py --bucket 16000
```

Prints the episode-return curve for every run directory it finds, then a
term-by-term table of the reference `train/*` scalars against the Torch
`logger.diagnostics=true` output. Ratios near 1.0 mean parity. Edit `JAX_RUNS`
/ `TORCH_RUNS` at the top to point at other run dirs.

Alignment detail that is easy to get wrong: JAX's `step` is a 1-based driver
record counter. The matching Torch field is `environment_steps`, **not**
`action_steps` — the latter excludes reset-only records and lands one bucket
early at episode boundaries.

### What to conclude

- If the Torch band overlaps the JAX band through 200k, parity holds and the
  investigation can close.
- If Torch tracks consistently below JAX past ~100k with non-overlapping
  bands, there is a real gap and the entropy/warmup findings in section 5 are
  the first place to look.
- Bear in mind n=3 per side and a very wide JAX seed spread. Do not read a
  single-bucket difference as signal.

---

## 4. Fixed this session

**Diagnostics reported weighted world-model losses.** `_reference_diagnostics`
logged the representation KL after its 0.1 coefficient while JAX logs it
before, showing a constant 0.10 ratio that looked like a 10x training
discrepancy. It was logging only — the coefficient is applied exactly once in
`DreamerV3ModelLoss` and the training loop does not re-apply it. The
`_unweighted` helper at `sota-implementations/dreamer_v3/dreamer_v3.py:204` now
divides the world-model terms by their coefficients, so
`loss_dynamic == loss_representation` as in the reference (the two KL terms
share a forward value and differ only in where the gradient is stopped).

Covered by `test_dreamer_v3_reference_diagnostics` at
`test/objectives/test_dreamer_v3.py:2131`, which also pins the read-only
contract (the pass must not move the training flag or the return-normalization
EMA).

**Note when reading old data:** the 50k runs in `/tmp/dv3-seed{0,1,2}` predate
this fix, so their `loss_representation` is still 10x low. The 200k runs have
the fix.

Local test status: `test/objectives/test_dreamer_v3.py` 78 passed;
`test/modules/test_dreamer_components.py` + `benchmarks/test_dreamer_v3_benchmark.py`
232 passed.

---

## 5. Open leads (both startup transients, neither yet actioned)

### 5a. Torch begins learner updates ~1000-2000 env steps earlier

Offset = first-logged `step` minus `updates`:

| run | offset |
|---|---|
| jax s0 | 3968 |
| jax s1 | 3968 |
| jax s2 | 3016 |
| torch s0/s1/s2 | 2031 (identical every seed) |

Torch is deterministic: `warmup_factor: 2` -> 2048 records, first update at env
step 2032. JAX gates on `len(replay) >= batch_size * batch_length` (1024 items,
each a 65-record window) inside `trainfn` in
`/home/ubuntu/dreamerv3/embodied/run/train.py`, and its `elements.when.Ratio`
initialises `_prev` on the first call *after* that gate opens. In principle the
gate should open near step 2048, but the measured runs open at ~3000-4000 and
vary between JAX's own seeds — the async driver keeps stepping envs during
startup and XLA compilation.

Assessment: Torch matches the configured intent and JAX's value is not a
protocol constant, so this was deliberately left alone. It does mean Torch
carries ~2000 extra updates at any given early step — i.e. Torch should be
slightly *ahead*, which makes the slightly-lower 48k returns marginally more
notable, not less.

### 5b. Policy entropy collapses ~2-4k steps earlier in Torch

Raw `train/ent/action` (Normal policy, 6 action dims, entropy summed; the
analytic range is [-5.30, 8.51]):

```
jax s0: 5k:8.22 10k:7.48 14k:5.53 18k:-1.63 22k:-2.58
jax s1: 5k:8.22  9k:8.00 14k:4.75 18k: 1.12 21k:-1.06
jax s2: 5k:8.28 11k:6.47 13k:3.57 15k: 2.19 18k:-1.38 22k:-3.03
tor s0: 6k:8.07 10k:7.55 14k:1.86 18k:-2.04 22k:-3.37
tor s1: 6k:8.12 10k:7.19 14k:-0.52 18k:-2.31 22k:-2.98
tor s2: 6k:8.12 10k:5.68 14k:-0.19 18k:-2.88 22k:-3.41
```

Torch is clearly lower in the 14k-22k window and converges by 30k. This is
consistent with the update head start in 5a rather than an independent defect.
Do not use a wide step-alignment window here — a +/-3000 tolerance across the
steep decline exaggerates the gap substantially.

Confirmed *not* the cause: the entropy coefficient matches
(`DreamerV3ActorLoss(entropy_bonus=3e-4)` vs JAX `imag_loss.actent: 0.0003`),
and JAX's `advnorm.impl` is `none` so no advantage normalisation is missing.

### 5c. Everything else in the diagnostics agrees

After the 4b fix, every other `train/*` term matched within seed noise across
0-48k: `loss/{dyn,con,rew,policy,value,repval}`, `val`, `slowval`, `ret`,
`ret_max`, `ret_rate`, `adv`, `adv_mag`, `adv_std`, `weight`, `con`, `rew`.
`weight` and `con` agree to three decimals from 8k onward.

---

## 6. Reference material

- JAX reference implementation: `/home/ubuntu/dreamerv3`, venv at
  `/home/ubuntu/dreamerv3/.venv`. Actor/critic metrics are defined in
  `dreamerv3/agent.py` around lines 395-445; the train loop and replay gate in
  `embodied/run/train.py`.
- JAX reference run (seed 0, to ~438k steps):
  `/home/ubuntu/logdir/jax-dmc/walker_walk-seed0` — `metrics.jsonl`,
  `scores.jsonl`, `config.yaml`, and a checkpoint under `ckpt/` that
  `numeric_check.py` reads.
- Torch env: `/home/ubuntu/rl/.venv` (built by `runner.sh`). Torch 2.15 nightly
  cu130, A100 80GB.
- Key config values on the JAX side: `imag_loss: {actent: 0.0003, lam: 0.95,
  slowreg: 1.0, slowtar: false}`, `retnorm: {impl: perc, perclo: 5, perchi: 95,
  limit: 1.0, rate: 0.01}`, `advnorm.impl: none`, `valnorm.impl: none`,
  `horizon: 333`, `contdisc: true`, `envs: 16`, `batch_size: 16`,
  `batch_length: 64`, `train_ratio: 1024`, `replay_context: 1`.

### Scripts in `dreamerv3-parity/scripts/`

| script | purpose |
|---|---|
| `compare.py` | **Start here.** Curve + `train/*` term-by-term comparison |
| `numeric_check.py` | Loads JAX weights into Torch modules, compares forwards |
| `cmp.py` | Lambda-return / weight / advantage / replay-target math |
| `scale_invariance.py` | Shows the optimizer chain is gradient-scale invariant |
| `launch.sh` | Launches the 5-run 200k comparison |
| `run_jax_seed.py` | Runs the JAX reference at a given seed/step budget |
| `diag_compare.py` | Earlier, hardcoded version of `compare.py`'s second table |
| `dump_{jax,torch}_params.py`, `compare_leaves.py` | Parameter-tree comparison |
| `env_{jax,torch}.py` | Environment-wrapper behaviour comparison |

Two related issue drafts sit at the repo root: `dreamerv3-gpu-issue.md` (fixed
by commit 7d00b796) and `dreamerv3-latent-state-issue.md`.
