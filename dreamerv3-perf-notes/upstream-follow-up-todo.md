# TODO: Prepare DreamerV3 performance changes for upstream

## Goal

Build a new performance branch from the latest `origin/main`. Keep only changes
that improve the complete DreamerV3 learner update and preserve the behavior of
the current implementation.

Do not rebase or merge this old performance branch into the new branch. Use it
only as a source of candidate changes and benchmark evidence.

## Current evidence

The same-node A100 benchmark used BF16, batch size 16, sequence length 64, and
imagination horizon 15. Current main, with CUDA graphs, took a median of
26.74 ms per learner update. This branch took 15.68 ms per learner update. The
steady-state speedup was 1.71 times.

The faster branch needed 775.62 seconds before steady-state timing. Current main
needed 218.04 seconds. The estimated break-even point was about 50,400 learner
updates. This benchmark did not test learning correctness.

## Implementation plan

### 1. Create a clean investigation branch

- Fetch `origin/main`.
- Create a separate worktree from the latest `origin/main`.
- Create an investigation branch such as
  `perf/dreamerv3-cudagraph-followups`.
- Record the exact main commit before each benchmark.
- Do not copy the custom learner as one large change.

### 2. Measure the current main learner update

Extend the existing DreamerV3 learner benchmark to report:

- CUDA-graph forward and backward time;
- optimizer time;
- slow-target update time;
- input-copy and graph-launch time;
- complete learner-update time;
- compile and capture time;
- peak CUDA memory;
- graph breaks and recompilations.

Use these measurements to select each optimization. Do not infer the cause of
the speed difference from total update time alone.

### 3. Test the observation-prior projection change

The old branch applies the RSSM observation-prior projection after the recurrent
loop. Port this change in isolation and keep the current main scan behavior.

Required checks:

- eager, loop, and scan paths;
- scan unroll values 1, 3, and 8;
- identical output structure and shapes;
- model outputs and gradients within the agreed tolerance;
- unchanged random-number use;
- supported hooks and fallback behavior;
- no new graph breaks or recompilations.

Keep the change only if it improves the complete learner update by at least 5%
on the same A100 test.

### 4. Test compiled optimizer updates

If optimizer time is material, move only the optimizer tensor math into a small
compile-safe helper. Keep eager execution as the default and keep the public
optimizer interface unchanged.

Required checks:

- one, two, ten, and one hundred update steps;
- FP32 and BF16;
- one and multiple parameter groups;
- missing gradients;
- RMS and momentum state;
- warm-up schedule and step counters;
- state-dict save and restore;
- parameter-group order;
- agreement with current main after each comparison run.

Do not add a general `compile_learner` flag. Do not accept optimizer-state drift
as a speed trade-off.

### 5. Test a faster slow-target update

Only work on this part if profiling shows that it takes material time. Prefer a
general bucketed `foreach` implementation or a small CUDA graph. Do not depend
on private fields of `SoftUpdate`.

Keep the change only if it improves the complete learner update by at least 3%
or saves about 1 ms per update on the same A100 test.

### 6. Use targeted compilation only if needed

If a large gap remains, inspect existing public loss and imagination modules.
Fix graph breaks in those modules before adding new code. Do not duplicate the
world-model, actor, value, or replay-value formulas in a custom learner.

## Correctness gate

Use current `origin/main` as the reference. Compare both variants with the same
inputs, seeds, device, precision, and update count. Check:

- reported loss metrics;
- world-model parameters and buffers;
- actor parameters, buffers, and return-normalization state;
- value parameters and buffers;
- slow-critic parameters;
- optimizer RMS and momentum state;
- posterior state and belief after the update;
- CPU and CUDA random-number state;
- graph breaks and recompilations.

Near-zero scalar metrics are not enough to prove parity. Use parameter,
optimizer-state, and recurrent-state comparisons as well.

## Performance gate

- Run main and each candidate on the same A100 GPU and software image.
- Use BF16, batch size 16, sequence length 64, and imagination horizon 15.
- Report compile and capture time separately from steady-state time.
- Report the median and range for repeated timing windows.
- Run one repetition during development and three independent process
  repetitions for the final pull-request evidence.
- Report peak CUDA memory and the exact source commits.
- Target less than 18 ms per complete learner update without the 776-second
  start-up cost seen on this branch.

## End-to-end gate

After a candidate passes the correctness and performance gates:

1. Run a 50,000-record Walker Walk test.
2. Compare collection, replay, learner, evaluation, and total wall time.
3. Check the learning curve against current main.
4. Run the one-million-record, multi-seed experiment only for the winning
   implementation.

## Upstream branch and commit rules

The final upstream branch must contain only the changes that pass all gates.
Remove failed flags, unused fallbacks, investigation scripts, and copied
learner code. Extend existing tests and benchmarks where possible.

Use small commits with one-line subjects and no commit body or co-author line.
Possible subjects are:

- `[Performance] Batch DreamerV3 observation prior projections`
- `[Performance] Compile DreamerV3 optimizer updates`
- `[Performance] Batch DreamerV3 slow target updates`
- `[Performance] Extend the DreamerV3 learner benchmark`

Split unrelated improvements into separate pull requests when they can be
tested and reviewed independently.
