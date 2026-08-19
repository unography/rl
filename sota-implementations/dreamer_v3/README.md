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
replay ratio of 1024, and 1.1 million environment steps. BF16 training is
enabled on CUDA. It logs stochastic training-episode returns against environment
steps, matching the reference curve protocol without relying on
wall-clock-dependent training iterations.

As in the reference DMC wrapper, Walker task randomness is not seeded: the
benchmark seeds control learner and policy random streams, while environment
resets use DM Control's default randomness. The step axis counts initial and
reset-only driver records as JAX does. Consequently, the 1.1-million-record
run executes 1,098,896 control actions for the 16-environment, 1,000-action
Walker horizon.

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
minimum final median return of 900. Override them for a deliberately smaller
ablation either with `--seeds`, `--window-size` and `--minimum-final-return`, or
as Hydra overrides (`benchmark.window_size=1000`), which the aggregation reads
as well. Full learning-curve runs are intended for scheduled or manual
validation; pull-request CI uses short smoke overrides.

`optimization.compile_rssm` compiles the RSSM recurrence. It is off so that a
run reproduces the numbers above. Solo on one GPU, `step` is about 2x on the
learner and draws the same categories; `scan` is about 3.7x, also compiling the
prior the imagination calls, and draws differently, so a seeded run diverges
from an eager one. Several seeds sharing
a GPU are bound by replay sampling instead, and gain little from either.
