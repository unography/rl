# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration
and a proprioceptive DeepMind Control Walker Walk reproduction configuration.

Run the small example with:

```bash
python sota-implementations/dreamer_v3/train.py
```

Run the full Walker Walk configuration with:

```bash
python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_walker
```

The Walker preset tracks the author-maintained JAX implementation at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. It matches that implementation's
640,867-parameter `size1m` configuration, uses 16 environments, batches of 16
sequences of length 64, a replay ratio of 1024, and 1.1 million environment
steps. BF16 training is enabled on CUDA. It logs stochastic training-episode
returns against environment steps, matching the current JAX curve protocol
without relying on wall-clock-dependent training iterations.

The Walker task is seeded from `env.seed`, as every other TorchRL example is;
pass `env.use_seed=false` for the JAX implementation's unseeded DMC resets. The
step axis counts initial and reset-only driver records as that
implementation does.

This is deliberately a reproduction of the pinned JAX `dmc_proprio` preset,
not of the paper's proprioceptive protocol. The two protocols differ:

| Setting | Pinned JAX `dmc_proprio` preset | DreamerV3 paper proprioceptive protocol |
| --- | --- | --- |
| Model size | `size1m` (640,867 parameters here) | 12M parameters |
| Environment steps | 1.1M | 500K |
| Action repeat | 1 | 2 |
| Replay ratio | 1024 | 512 |
| Optimizer | AGC, LaProp-style RMS scaling then momentum, 1,000-step warmup | Paper recipe |
| Reported aggregation | Three-seed median and interquartile range in this benchmark | Five-seed mean and standard deviation |

TorchRL's public DreamerV3 API documentation remains centered on the paper's
algorithmic semantics. This named SOTA preset documents later choices in the
evolving JAX codebase instead of silently treating them as paper requirements.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

For a three-seed median and interquartile reproduction run:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir dmc_walker_runs
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
or manual validation; pull-request CI uses short smoke overrides.

For a smaller ablation, shorten the run rather than the window:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir smoke \
  collector.total_frames=100000 \
  benchmark.minimum_final_median_return=0
```

Every worker runs to the same time limit, so episodes finish in bursts one
episode apart: `(env.max_episode_steps + 1) * collector.num_envs`, or 16,016
records for the preset. A window narrower than that holds no completed episode
over most of the run, so the script refuses one before launching anything. The
command above keeps the 50,000-step window and still fills two of them with
about 48 episodes each.

`optimization.compile_rssm` compiles the RSSM recurrence and is off in the base
Pendulum configuration, since a short run never repays the build. `step`
compiles one deterministic transition. `loop` compiles fixed-size chunks while
keeping category draws outside the graph; the chunk size is controlled by
`optimization.rssm_loop_chunk_size`. `scan` compiles the higher-order recurrence
while likewise drawing its categorical samples outside the compiled region.
The scan uses
`optimization.rssm_scan_unroll=8` by default; lower values reduce compilation
time and graph size, while `1` disables manual unrolling.

The DMC Walker preset enables `optimization.compile_value_losses`, which
compiles the deterministic value and replay-value losses separately while
retaining the original modules for
parameter ownership and checkpoints. Compilation happens on the first learner
update and is intended to be amortized over the full benchmark run. PyTorch 2.6
or newer is required; older supported versions log a warning and keep both
losses eager. The stochastic actor remains eager: on the tested PyTorch 2.14
nightly, CUDA BF16 actor compilation preserves final RNG position but introduces
larger numerical drift than the deterministic loss graphs.

RSSM compilation is available for FP32 experiments but is not enabled by the
mixed-precision DMC preset. CUDA BF16 compilation preserved sampled states and
the final RNG position in local checks, but compiler reassociation produced a
material optimizer-momentum difference in a complete two-update learner. On
the tested PyTorch 2.14 nightly, use the default compile mode with `scan`:
`scan` plus `reduce-overhead` reaches a cudagraph backward assertion. The DMC
Walker preset is unaffected because RSSM compilation remains disabled.

The complete learner hot path, including backward, the optimizer, and the
slow-critic update, can be measured with:

```bash
TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1 \
COMPOSITE_LP_AGGREGATE=0 \
TD_GET_DEFAULTS_TO_NONE=1 \
python benchmarks/dreamer_v3_update.py \
  --compile-rssm none \
  --compile-components value-replay
```
