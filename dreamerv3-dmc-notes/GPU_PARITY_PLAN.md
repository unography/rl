# GPU parity plan — DreamerV3 `walker_walk` vs JAX

**For the agent resuming this on a GPU box.** Goal: show torchrl DreamerV3 on
this branch reproduces danijar's published `dmc_walker_walk` curve. Read
`LEARNING.md` first for what every change does. Constraint: **local only — never
push, never open a PR/issue.** Produce artifacts; the user submits them.

---

## State when you pick this up (done on CPU)
- Branch `dreamerv3-jax-parity-dmc` = `main` + 18 parity commits + DMC commits.
- Example runs **both** Gym and DMC (`env.backend`). Verified on CPU: Pendulum
  regression + DMC cartpole_balance (both actor paths).
- `config_dmc.yaml` = JAX `dmc_proprio` size1m preset (walker/walk).
- JAX reference curve extracted, committed: `reference/dmc_walker_walk_dreamerv3_mean.csv`
  (5 seeds, 10k–490k). Final mean ~881.
- Harness ready + unit-tested: `scripts/run_dmc_parity.py`, `scripts/extract_jax_curve.py`.
- **Not done** (needs GPU): any real training run, the VERIFY items, the overlay.

## What you must NOT assume
- Do **not** trust `config_dmc.yaml` blindly. The `VERIFY` items below are
  best-effort mappings. A mismatch looks like a "parity failure" that is really
  a hyperparameter/impl difference. Resolve them *before* a full run.

---

## Step 0 — environment
- uv venv, editable torchrl (`.venv/bin/python` already used in scripts). Confirm
  `dm_control` importable: `.venv/bin/python -c "import dm_control; from torchrl.envs.libs.dm_control import DMControlEnv"`.
- Confirm CUDA: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
- If deps missing: `uv pip install dm_control` (and `mujoco`).

## Step 1 — resolve the VERIFY items (cheap, do before burning GPU hours)
Read the torchrl impl and compare to JAX; fix `config_dmc.yaml` if wrong.

1. **`use_reinforce`** — JAX continuous control backprops the actor through the
   dynamics (`agent.py` imag loss, `reward_grad: True`), not REINFORCE. Config
   sets `use_reinforce: false`. Confirm `DreamerV3ActorLoss` with `imag_loss=true`
   actually does dynamics-gradient (grep the loss for the reinforce vs reparam
   branch in `torchrl/objectives/dreamer_v3.py`).
2. **reward bins** — JAX `rewhead.bins: 255` with `symexp_twohot` (symexp-spaced).
   Example uses `num_reward_bins: 41` over a **linear** `linspace(-20,20)`
   (`dreamer_v3.py`, search `reward_bins = torch.linspace`). Check whether the
   spacing matches JAX; walker per-step reward is [0,1] so 41 linear bins may be
   fine, but confirm. Value bins already 255.
3. **MLP `depth`** — JAX `size1m` sets `depth: 4`, `units: 64`. Confirm torchrl
   `depth` semantics (hidden-layer count) and set to match.
4. **`contdisc`** — JAX `contdisc: True` (continue head provides per-step
   discount). Config uses `contdisc: false` + `horizon: 333` (constant 0.997).
   Walker never terminates early, so both give ≈0.997; keep `false` unless the
   curve lags — then try the contdisc path.
5. **actor entropy (`actent`)** — JAX `imag_loss.actent: 3e-4`. Check the example
   exposes/hardcodes this; if configurable, set 3e-4.
6. **`obs_embed_dim` / encoder width** — JAX enc `units` for size1m is 64; config
   uses 64. Fine.

## Step 2 — throughput sanity (1 short GPU run)
```bash
.venv/bin/python sota-implementations/dreamer_v3/dreamer_v3.py --config-name config_dmc \
  env.device=cuda collector.total_frames=20000 logger.eval_every=5000
```
- Confirm it runs on CUDA, no NaNs, eval logs appear.
- Time it -> estimate wall-clock for 500k steps. Note frames/sec.
- If OOM: lower `replay_buffer.batch_size` or `seq_len` (keep train_ratio via
  `updates_per_batch`), or shrink model (but that breaks size1m parity — prefer
  batch/seq).

## Step 3 — single-seed shakeout (walker, to ~150k)
```bash
.venv/bin/python dreamerv3-dmc-notes/scripts/run_dmc_parity.py \
  --seeds 0 --total-frames 150000 --device cuda --eval-every 10000
```
- Overlay lands in `plots/dmc_walker_walk_parity.png`.
- **Checkpoint the curve shape**, not the final value. By 50k JAX mean is ~289,
  by 100k ~475 (see table). If torchrl is flat near 0 at 100k, stop and debug
  (Step 5) — do not launch the full grid.

## Step 4 — full parity grid (+ V3-off ablation)
```bash
# parity + V3-off ablation, both overlaid on the JAX band:
.venv/bin/python dreamerv3-dmc-notes/scripts/run_dmc_parity.py \
  --seeds 0 1 2 --total-frames 500000 --device cuda --eval-every 10000 \
  --arm v3off config_dmc_v3off
```
- 3 seeds min (JAX uses 5; add `3 4` if time). Seeds run sequentially here; if
  you have >1 GPU or memory headroom, launch seeds as parallel processes and
  point the harness at the logs.
- **`v3off` arm** = `config_dmc_v3off.yaml`: acting-policy fix + loss bugfixes
  ON, but the V3 feature set OFF (scalar critic, tanh actor, no EMA/retnorm/
  unimix/block-GRU/AGC, plain REINFORCE). It should **learn but underperform**
  the parity arm -- that gap is the V3 features' contribution. Contrast with the
  `dreamerv3-baseline-dmc` branch (main's algorithm, expected flat).
- Other ablation arms via `--extra-arm NAME "overrides"` (e.g. a buggy-KL A/B).

## Acceptance criteria (JAX `walker_walk` reference)
| env steps | JAX mean | JAX seed range |
|---|---|---|
| 50k  | 289 | [206, 448] |
| 100k | 475 | [298, 714] |
| 200k | 800 | [709, 922] |
| 300k | 844 | [673, 967] |
| 400k | 932 | [798, 986] |
| 490k | 881 | [736, 955] |

**Pass** = torchrl mean curve lands inside the JAX seed range at 200k and 490k,
and reaches ≥800 final. **Soft pass** = same shape, slower (breakthrough later)
— still strong evidence; note the offset. **Fail** = flat / collapses / plateaus
far below band -> Step 5.

## Step 5 — if it doesn't match (debug order)
1. **Flat at ~0 through 100k** — learning is broken, not slow. Check: acting
   policy actually perceives obs (rollout probe: vary obs, action must change);
   KL not pinned at the free-bits floor (log shows `kl≈1.10` stuck = the summed
   free-nats bug or wiring); reward head predicting non-constant.
2. **Learns but plateaus low (~300–500)** — hyperparameter gap. Re-check Step 1
   items in this order: `use_reinforce`/dynamics-grad, actent, train_ratio,
   retnorm, EMA value rate.
3. **Unstable / NaNs** — AGC off or lr too high; confirm `optimizer: dreamerv3`
   active and `agc: 0.3`; try `opt_warmup: 1000`.
4. **Right shape, horizontally shifted** — env-step accounting (action_repeat).
   DMC default repeat = 1; confirm x-axis counts post-repeat env steps like JAX.
5. Cross-check one ingredient at a time against JAX; the Pendulum ablation
   harness pattern (`dreamerv3-parity-notes/`) is the template.

## Deliverables to produce (commit locally, do not push)
- `plots/dmc_walker_walk_parity.png` (+ `.csv`) — the overlay figure.
- Update `RESULTS.md` (create) with: hardware, wall-clock, seeds, final numbers,
  pass/fail vs table, and any VERIFY items you changed + why.
- If you changed `config_dmc.yaml`, commit it with a message explaining the fix.
- A one-paragraph summary the user can paste into a PR description.

## Reproduce the reference (if you need other tasks)
```bash
.venv/bin/python dreamerv3-dmc-notes/scripts/extract_jax_curve.py \
  --scores <path>/scores/dmc_proprio-dreamerv3.json.gz --task dmc_cartpole_balance
```
