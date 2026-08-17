# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration
and a proprioceptive DeepMind Control Walker Walk reproduction configuration.

Run the small example with:

```bash
python sota-implementations/dreamer_v3/dreamer_v3.py
```

Run the full Walker Walk configuration with:

```bash
python sota-implementations/dreamer_v3/dreamer_v3.py \
  --config-name=config_dmc_walker
```

The Walker preset matches the 640,867 trainable parameters of the JAX ``size1m``
configuration, uses 16 environments, batches of 16 sequences of length 64, a
replay ratio of 1024, and 1.1 million environment steps. BF16 training and the
compiled RSSM and imagination scans are enabled on CUDA. The Walker CUDA preset
also captures the fixed-shape learner forward/backward in one outer CUDA graph;
optimizer scheduling and slow-target updates remain eager. It logs stochastic
training-episode returns against environment steps, matching the reference curve
protocol without relying on wall-clock-dependent training iterations.

As in the reference DMC wrapper, Walker task randomness is not seeded: the
benchmark seeds control learner and policy random streams, while environment
resets use DM Control's default randomness. The step axis counts initial and
reset-only driver records as JAX does. Consequently, the 1.1-million-record
run executes 1,098,896 control actions for the 16-environment, 1,000-action
Walker horizon.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution; the CUDA-graph learner flag automatically falls back to the eager
learner on a CPU device.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

For a three-seed median and interquartile reproduction run:

```bash
python sota-implementations/dreamer_v3/benchmark.py \
  --seeds 0 1 2 \
  --output-dir dmc_walker_runs
```

The benchmark writes one metrics file per seed plus `summary.json`, aggregates
50,000-step windows into median and interquartile curves, and checks a minimum
final median return of 900. Use `--minimum-final-return` or `--window-size` to
override these settings for a deliberately smaller ablation. Full
learning-curve runs are intended for scheduled or manual validation;
pull-request CI uses short smoke overrides.
