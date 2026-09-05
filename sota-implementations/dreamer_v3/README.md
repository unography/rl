# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration,
three DeepMind Control presets and a Crafter preset that reproduce the
author-maintained JAX implementation at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`:

| Preset | JAX protocol | Task | Observation | Action | Model |
| --- | --- | --- | --- | --- | --- |
| `config_dmc_walker` | `dmc_proprio` | Walker Walk | proprioceptive vector | continuous | `size1m` |
| `config_dmc_cheetah` | `dmc_proprio` | Cheetah Run | proprioceptive vector | continuous | `size1m` |
| `config_dmc_walker_vision` | `dmc_vision` | Walker Walk | 64x64 RGB pixels | continuous | `size12m` |
| `config_crafter` | `crafter` | Crafter, reward task | 64x64 RGB pixels | 17 one-hot classes | `size200m` |

Run the small example with:

```bash
python sota-implementations/dreamer_v3/train.py
```

Run the Walker Walk and Cheetah Run proprioceptive configurations with:

```bash
python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_walker
python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_cheetah
```

Run the Walker Walk vision configuration with:

```bash
MUJOCO_GL=egl python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_walker_vision
```

Run the Crafter configuration, with the optional `crafter` dependency
installed, with:

```bash
python sota-implementations/dreamer_v3/train.py --config-name=config_crafter
```

## Configuration groups

A preset names its task and composes two config groups, which hold every
value once:

- `protocol/` holds the JAX-derived schedule, replay and optimizer settings:
  `dmc_proprio` (16 environments, batches of 16 sequences of 64, replay ratio
  1024, 1.1 million driver records), `dmc_vision` (the same schedule with
  replay ratio 256, pixel-only 64x64 observations on camera 0, action repeat
  1, and a CPU replay of raw `uint8` images) and `crafter` (one environment,
  replay ratio 512, 64x64 images, one-hot actions and a CPU replay).
- `model_size/` holds the JAX dimension bundles. `size1m` is RSSM
  deterministic size 512, RSSM hidden and MLP units 64, 4 classes and image
  depth 4; `size12m` is 2048, 256, 16 classes and image depth 16; `size200m`
  is the JAX default, 8192, 1024, 64 classes and image depth 64. All use 32
  categoricals. Override it on the command line, as in `model_size=size12m`.
  Overriding `protocol=` instead is not meant for the shipped presets: their
  own task blocks, such as the Walker threshold and decoder event dims, would
  still apply.

The action space comes from the environment: a one-dimensional continuous
spec gives the bounded Normal policy, a one-hot spec gives a categorical
policy whose one-hot sample feeds the RSSM and replay, and which the
environment adapter turns into an integer. `networks.policy_unimix` mixes a
uniform distribution into the categorical policy; see the Crafter section for
why the presets keep it at zero.

The learner starts after `replay_buffer.warmup_records` driver records; the
default `null` applies the reference rule, `batch_size * seq_len` sampleable
items after each stream has spent its first `seq_len` records: 2048 records
for the 16-environment protocols, 1088 for Crafter's single environment. The
run manifest records the threshold and the record of the first update.

The run manifest, the `summary` record of the metrics file, names the
protocol and the model size, the observation mode, key and shape, the action
kind and size, the parameter count, the replay bytes, and the whole effective
configuration. `dmc_vision` with the JAX default model is not the same
experiment as `dmc_vision` with `size12m`; the manifest tells them apart.
Each `train_episode` record carries the driver step and the action count at
which the episode ended, its `episode_return` and `episode_length`, whether
it terminated rather than hit the time limit, and, for Crafter, its
achievement counts.

## Reference protocols

The proprioceptive presets match the JAX implementation's 640,867-parameter
`size1m` configuration. BF16 training is enabled on CUDA. They log stochastic
training-episode returns against environment steps, matching the current JAX
curve protocol without relying on wall-clock-dependent training iterations.

The DMC tasks are seeded from `env.seed`, as every other TorchRL example is;
pass `env.use_seed=false` for the JAX implementation's unseeded DMC resets. The
step axis counts initial and reset-only driver records as that implementation
does. The JAX and TorchRL random-number streams are not identical, so paired
seeds do not give paired trajectories.

This is deliberately a reproduction of the pinned JAX presets, not of the
paper's protocols. The two differ:

| Setting | Pinned JAX `dmc_proprio` preset | DreamerV3 paper proprioceptive protocol |
| --- | --- | --- |
| Model size | `size1m` (640,867 parameters here) | 12M parameters |
| Environment steps | 1.1M | 500K |
| Action repeat | 1 | 2 |
| Replay ratio | 1024 | 512 |
| Optimizer | AGC, LaProp-style RMS scaling then momentum, 1,000-step warmup | Paper recipe |
| Reported aggregation | Three-seed median and interquartile range in this benchmark | Five-seed mean and standard deviation |

TorchRL's public DreamerV3 API documentation remains centered on the paper's
algorithmic semantics. These named SOTA presets document later choices in the
evolving JAX codebase instead of silently treating them as paper requirements.

## Vision

The vision preset builds a pixel-only `DMControlEnv` that renders camera 0 at
64x64. Each replay record stores one HWC `uint8` image; the learner converts
each sampled batch on its device, so replay holds no FP32 copy. The image
encoder and decoder follow the JAX `simple` networks: same-padded 5x5
convolutions with depths `image_depth * [2, 3, 4, 4]`, 2x2 max pooling,
channel RMS normalization and SiLU in the encoder; block-linear and two-layer
projections into the 4x4 map, pixel-repeat upsampling and a sigmoid output in
the decoder. The reconstruction loss compares the sigmoid output with the
image scaled to `[0, 1]`, summed over height, width and channels.

MuJoCo picks its renderer from `MUJOCO_GL` when dm_control is first imported,
which happens as soon as `torchrl.envs` loads, so export `MUJOCO_GL=egl` in
the run environment for a headless Linux run; the script cannot set it for
you. At startup a dm_control run renders one frame and fails clearly if the
renderer has no context or draws a constant image. Before collection it estimates the bytes of
one replay record (image, latent state and belief, action, reward, flags and
the writer's generation counter) and refuses a CPU replay above 90% of the
memory it may use, or of `replay_buffer.host_memory_limit_gb` when set. That
limit is the smaller of the container's cgroup limit and the physical memory.
The bytes of one record come from the environment's fake tensordict passed
through the record builder, so they are what the storage allocates. Both
protocols use action repeat 1, which is what the environment does; a repeat
above 1 would need a terminal-aware frame skip and driver-step accounting in
inner steps, so the example does not expose it. The vision preset keeps
1.2 million records, above the 1.1 million-record run, so nothing is evicted;
at `size12m` that is about 25 GiB. The nominal JAX capacity of five million
records would need about five times that.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

## Crafter

The Crafter preset reproduces the pinned JAX `crafter` preset: the `reward`
task of Crafter 1.8.3, one environment, 1.1 million driver records, batches of
16 sequences of 64, train ratio 512 (one update for every two records, the
first after driver record 1088), imagination horizon 15, BF16 on CUDA, 64x64
images and the JAX default model, which the JAX configuration names
`size200m` and which has 165,965,075 parameters here; `model_size=size12m`
gives a 10,498,259-parameter ablation that the manifest names as such. The
JAX README's `--run.train_ratio 32` example is not that preset. The preset
needs the optional dependency:

```bash
pip install crafter==1.8.3
```

The environment is Crafter's own `crafter.Env`, constructed with the image
size, the reward flag, the step limit and the root seed, behind a private
gym-like TorchRL adapter, so nothing depends on Crafter's optional Gym
registration or on the installed Gym version. The observation is the HWC
`uint8` image under `pixels`; each step also reports the episode's 22
achievement counts under `achievements`, in the order of
`crafter.constants.achievements`, which the manifest records. The policy,
the RSSM and replay use a 17-entry FP32 one-hot action, which the adapter
turns into the integer Crafter expects; `collector.frames_per_batch=1`
matches the reference driver, which trains after every environment step,
and `collector.num_envs` stays at 1, since deaths end episodes at different
times and the replay stream of several environments needs synchronized
resets. Death ends the episode with Crafter's discount 0 and is recorded as
`terminated`; the 10,000-step limit is `truncated`. `env.seed` is passed to
the constructor, since Crafter reads its seed there and not from a later
`set_seed`; the pinned JAX preset constructs Crafter unseeded, which
`env.use_seed=false` reproduces, so the seeded default is a disclosed
reproducibility correction. Two environments with the same seed produce the
same reset and the same first nine steps of a fixed action trace. Beyond
that, Crafter 1.8.3 itself may diverge at a chunk-balancing step, every ten
steps: a despawn draws from a list built from a Python set of objects, so
identical seeds and actions are not guaranteed to give identical episodes
in either implementation.

Crafter's episode return is not the official metric. The official Crafter
score is the geometric mean of `1 + s_i` minus one over the 22 achievements,
with `s_i` the percentage of training episodes that unlock achievement `i`,
computed from the episodes that end within the first million environment
actions. `benchmark.py` computes it from the `train_episode` records of each
seed, under the budget `benchmark.crafter_action_budget`, and writes the
success rates and the score of every seed with the median under `crafter` in
`summary.json`; the metrics file keeps the raw records so the score can be
recomputed. The metrics use the names `episode_return` and `crafter_score`;
the JAX `scores.jsonl` field `episode/score` is the episode return.

The pinned JAX configuration sets `policy.unimix: 0.01`, but its categorical
policy head builds the distribution without the mixture, so the discrete
policy samples the raw softmax; the preset keeps `networks.policy_unimix`
at 0 to match. `networks.policy_unimix=0.01` restores the paper's mixture.

Each replay record holds the image, the 32x64 latent state and the 8192-wide
belief in FP32, the one-hot action, the reward, four flags and the writer's
generation counter, 53,332 bytes at `size200m`, and the no-eviction capacity
of 1.2 million records needs about 60 GiB of host memory; the startup check
refuses a smaller host, as for the vision preset. The nominal JAX capacity of
five million records would need about four times that. The deterministic
evaluation runs one episode every 50,000 records, since an episode can last
10,000 steps; the official metric comes from the training episodes.

## Multi-seed benchmark

For a three-seed median and interquartile reproduction of a preset:

```bash
python sota-implementations/dreamer_v3/benchmark.py --config-name=config_dmc_walker
python sota-implementations/dreamer_v3/benchmark.py --config-name=config_dmc_cheetah
MUJOCO_GL=egl python sota-implementations/dreamer_v3/benchmark.py \
  --config-name=config_dmc_walker_vision
python sota-implementations/dreamer_v3/benchmark.py --config-name=config_crafter
```

The benchmark writes one metrics file per seed plus `summary.json` under
`<preset>_runs`, or `--output-dir`. It aggregates the stochastic training
returns into median and interquartile curves over fixed windows and records
the config name and task; the Crafter preset also gets the achievement
success rates and the Crafter score of each seed, from the episodes within
the one-million-action budget of the official protocol. The seeds and the window come from the protocol's
`benchmark` block: three seeds and 50,000-step windows. The Walker preset
requires a final median return of 900; the Cheetah and vision presets have no
threshold until a baseline cohort defines one, and a null
`benchmark.minimum_final_median_return` disables the check instead of reusing
the Walker value. `benchmark.*` Hydra overrides change these settings, as in
`benchmark.seeds=[0,1,2,3,4]`. `env.seed` and `logger.metrics_jsonl` are set
per run and are rejected as overrides, since either would collapse the seeds
onto one trajectory. Full learning-curve runs are intended for scheduled or
manual validation; pull-request CI uses short smoke overrides.

The Walker preset also has a wrapper that runs the benchmark with named
modes:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh
```

For the fastest supported accelerator path, enable the compiled RSSM scan
(unrolled eight steps at a time) and CUDA-graph capture of the fixed-shape
learner forward/backward:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh --fast
```

To measure the same fixed-shape learner update after compile and capture
warmup—including every loss, backward, optimizer, and slow-target update—run:

```bash
python benchmarks/ad_hoc/bench_dreamer_v3_learner.py
```

The timing excludes replay sampling and environment collection. Use the
benchmark arguments to change the batch size, sequence length, scan unroll,
warmup, or number of measured updates. Compilation and graph capture happen
during warmup and are excluded from the reported samples.

On one NVIDIA GB200 with PyTorch 2.12.0, CUDA 13.0, BF16, batch size 16,
sequence length 64, scan unroll 8, 10 warmup updates and 50 measured updates:

| Learner backend | Median update | Transitions/s | Speedup |
| --- | ---: | ---: | ---: |
| Compiled scan | 358.51 ms | 2,856 | 1.00x |
| Compiled scan + CUDA graph | 17.83 ms | 57,415 | 20.10x |

Compilation has an up-front cost, so the short validation remains eager:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh --smoke
```

The benchmark writes one metrics file per seed plus `summary.json`, aggregates
the stochastic training returns into median and interquartile curves over fixed
windows, and fails when the final window median falls short. The seeds, the
window and the threshold come from the `benchmark` block of
`config_dmc_walker.yaml`, which ships three seeds, 50,000-step windows and a
minimum final median return of 900; `benchmark.*` Hydra overrides change them,
as in `benchmark.seeds=[0,1,2,3,4]`. `env.seed` and `logger.metrics_jsonl` are
set per run and are rejected as overrides, since either would collapse the
seeds onto one trajectory. Full learning-curve runs are intended for scheduled
or manual validation; pull-request CI uses short smoke overrides. Set
`OUTPUT_DIR` to change the output directory (the defaults are
`dmc_walker_runs` and `dmc_walker_smoke`), and append any other Hydra overrides
to the wrapper, for example `benchmark.seeds=[0]`. Each run logs the resolved
training device, replay device, RSSM backend, scan unroll, mixed-precision state
and learner CUDA-graph setting.

For a smaller ablation, shorten the run rather than the window:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir smoke \
  collector.total_frames=100000 \
  benchmark.minimum_final_median_return=null
```

Every worker runs to the same time limit, so episodes finish in bursts one
episode apart: `(env.max_episode_steps + 1) * collector.num_envs`, or 16,016
records for the presets. A window narrower than that holds no completed
episode over most of the run, so the script refuses one before launching
anything. The command above keeps the 50,000-step window and still fills two
of them with about 48 episodes each.

## Local smoke tests

Each path has a small CPU run of five updates. The proprioceptive one:

```bash
python sota-implementations/dreamer_v3/train.py --config-name=config_dmc_cheetah \
  optimization.device=cpu env.max_episode_steps=10 \
  collector.num_envs=2 collector.frames_per_batch=8 collector.total_frames=44 \
  replay_buffer.buffer_size=1000 replay_buffer.batch_size=2 \
  replay_buffer.seq_len=4 replay_buffer.warmup_records=10 \
  optimization.train_ratio=null optimization.updates_per_batch=1 \
  logger.eval_every=20 logger.eval_episodes=1 logger.train_every=10 \
  networks.rnn_hidden_dim=16 networks.hidden_dim=8 \
  networks.num_categoricals=4 networks.num_classes=4 \
  networks.encoder_layers=1 networks.decoder_layers=1 \
  networks.actor_layers=1 networks.value_layers=1
```

The vision one adds `networks.image_depth=2` and selects
`--config-name=config_dmc_walker_vision`. The Crafter one, eight updates over
44 records of one environment:

```bash
python sota-implementations/dreamer_v3/train.py --config-name=config_crafter \
  optimization.device=cpu env.max_episode_steps=10 \
  collector.frames_per_batch=4 collector.total_frames=44 \
  replay_buffer.buffer_size=1000 replay_buffer.batch_size=2 \
  replay_buffer.seq_len=4 replay_buffer.warmup_records=10 \
  optimization.train_ratio=null optimization.updates_per_batch=1 \
  logger.eval_every=20 logger.eval_episodes=1 logger.train_every=10 \
  networks.rnn_hidden_dim=16 networks.hidden_dim=8 \
  networks.num_categoricals=4 networks.num_classes=4 networks.image_depth=2 \
  networks.encoder_layers=1 networks.decoder_layers=1 \
  networks.actor_layers=1 networks.value_layers=1
```

All three are also the end-to-end tests of `test/objectives/test_dreamer_v3.py`
when `dm_control`, respectively `crafter`, is installed.

`optimization.compile_rssm` compiles the RSSM recurrence and is off by default,
since a short run never repays the build. `step` compiles the deterministic work
and draws the same categories as an eager run; `scan` compiles the unrolled
recurrence and the imagination prior, and is faster, but its draws fall inside
the compiled region, so a seeded run diverges from an eager one. The scan uses
`optimization.rssm_scan_unroll=8` by default; lower values reduce compilation
time and graph size, while `1` disables manual unrolling.
`optimization.cudagraph_train_step=true` captures the learner forward and
backward after five warmup calls. It requires CUDA and fixed input shapes;
optimizer and target-network steps remain outside capture so their schedules
continue to advance normally.
