# DreamerV3 Crafter Implementation Specification

Status: design only. This file does not implement Crafter support.

## 1. Goal

Add a faithful Crafter path to the TorchRL DreamerV3 example that is already
being extended for DMC vision. The completed path must support:

1. the pinned JAX repository configuration;
2. the official Crafter learning metric;
3. matched-work systems measurements; and
4. eager and compiled Torch learner modes with tested behavior.

Crafter is useful because it adds image observations, categorical actions,
sparse rewards, exploration, and long episodes. It is not a controlled test of
image input alone. Walker vision remains the controlled observation-modality
test. Crafter is the transfer test to a different world-model workload.

The implementation must follow `AGENTS.md` and `CONTRIBUTING.md`. In
particular, it must use TensorDict and TorchRL components, keep policies as
`TensorDictModule` objects, use a collector or `env.rollout`, and avoid a
custom environment stepping loop.

Apply these repository rules throughout the change:

- keep normal imports at module scope and lazy-load only optional Crafter code;
- use `torchrl.implement_for` for Gym-version dispatch;
- use `NestedKey` for configurable TensorDict keys and test a nested key;
- derive device placement from inputs, parameters, or buffers;
- do not add `self.device` to an `nn.Module`;
- use the TorchRL logger instead of `print` in library code;
- use `torchrl.timeit` for library timing;
- add accurate type hints to new public signatures; and
- keep code comments short and necessary.

## 2. Current branch and reference revisions

This specification is stored on the DMC expansion branch:

```text
branch: plan/dreamerv3-cheetah-vision
current tip: 0902fb7aaaf358a56ae617f53b0b6b6736eb09b6
current base: 443297894c51ac196d67fe76f92fd2011b409588
```

The branch already contains three implementation commits:

1. `37c95a5a0` forwards DMC render arguments to the pixel wrapper.
2. `9c60ba08f` adds image reconstruction semantics.
3. `0902fb7aa` adds DMC Cheetah and Walker vision presets.

The untracked `dreamerv3-dmc-expansion-plan.md` says that the branch is plan
only. That statement is stale. Do not use the file as evidence of the branch
state, and do not modify or commit it as part of Crafter work.

Use these experiment references unless the dissertation manifest records a
new pin:

```text
JAX DreamerV3: b65cf81a6fb13625af8722127459283f899a35d9
Crafter: 1.8.3
Torch performance code: c1170408cf507d1b2dcb7d9c13ad6280804714d7
Torch performance evidence: 31baba2d8b1d467888a506fc8e790fb22f73999d
```

The performance evidence commit is not implementation code. Do not merge or
cherry-pick it into a Crafter implementation pull request.

## 3. Protocols that must remain separate

### 3.1 JAX repository reproduction

The pinned JAX `crafter` preset selects:

- task `crafter_reward`;
- 1.1 million driver records;
- one environment;
- train ratio 512;
- batch size 16;
- sequence length 64;
- replay context 1;
- imagination horizon 15;
- bfloat16 compute;
- 64 by 64 RGB images;
- 17 categorical actions; and
- the large default model.

The large default model uses 32 categorical latent variables, 64 classes,
an RSSM deterministic size of 8192, hidden and MLP size 1024, and image depth
64. Record the measured parameter count. Do not treat the `size200m` name as
an exact count.

The JAX README example overrides the train ratio to 32. That command is not
the named repository preset. The main reproduction must use the effective
configuration above.

### 3.2 Official Crafter learning protocol

The official learning budget is exactly one million environment actions.
Report:

- the success rate of each of the 22 achievements;
- the Crafter score;
- raw episode return as a secondary metric; and
- wall time, accelerator energy, and memory.

If `s_i` is the percentage of training episodes that unlock achievement `i`,
calculate the score as:

```text
exp(mean(log(1 + s_i))) - 1
```

Use percentages from 0 to 100 in this formula. Do not label raw episode return
as Crafter score. When a 1.1-million-record repository run is also used for
the official result, calculate the official metric from episodes within the
first one million actions.

### 3.3 DreamerV3 paper scaling study

The paper studies model sizes from 12M to 400M and replay ratios from 1 to 64
over much larger budgets. This is background evidence. It is not the primary
dissertation protocol and is not part of this implementation gate.

## 4. Source-of-truth comparison manifest

Before implementation, add a dissertation-side machine-readable manifest for
the JAX and Torch arms. Do not add experiment artifacts to the upstream
TorchRL pull request. The manifest must record:

- every source revision and container digest;
- the Crafter package version;
- root environment seed and agent seed;
- image key, layout, shape, data type, and range;
- action representation at the policy, RSSM, replay, and environment edges;
- episode limit and terminal versus truncation rules;
- driver records, environment actions, replay records, and raw frames;
- warm-up threshold and exact learner-update count;
- replay capacity, storage device, stored keys, and bytes per record;
- batch size, sequence length, and replay context;
- all model dimensions and measured parameter count;
- optimizer values, precision, and imagination horizon;
- compile backend, mode, warm-up, and cache state; and
- hardware, host-memory limit, GPU telemetry interval, and profiler status.

Generate effective Hydra and JAX configurations for every run. Do not infer a
configuration from the preset name alone.

## 5. Configuration design

### 5.1 Add a Crafter protocol and task preset

Add these files beside the existing DMC presets:

```text
sota-implementations/dreamer_v3/protocol/crafter.yaml
sota-implementations/dreamer_v3/config_crafter.yaml
sota-implementations/dreamer_v3/model_size/size200m.yaml
```

The protocol must set the JAX repository values from section 3.1. The task
preset must select Crafter reward mode and the large model explicitly. A
smaller `size12m` task preset may be added for development, but its name and
run manifest must state that it is an ablation.

Extend the existing configuration instead of adding an unrelated configuration
system. Add only fields used by more than one function. At minimum, represent:

```yaml
env:
  backend: crafter
  observation_mode: image
  image_size: [64, 64]
  action_type: categorical
  action_classes: 17
  max_episode_steps: 10000
  use_seed: true

networks:
  policy_unimix: 0.01
```

Prefer an action descriptor derived from the environment specification over
copying `action_type` and `action_classes` through many functions. If these
configuration fields are retained, validate them against the environment at
startup.

The current DMC vision validation rejects image mode outside `dm_control`.
Replace that check with capability checks. Image mode must require a valid
image observation. It must not require a specific environment backend.

### 5.2 Match the first-update rule

Do not reuse the current `warmup_factor: 2` without evidence. The JAX learner
starts after its replay-size condition is satisfied. Record the exact first
update record from a short reference run and express the Torch threshold in
records. The two arms must start learning at the same point.

At train ratio 512 with a 16 by 64 training batch, the intended steady-state
rate is one learner update for every two environment records. Replay context
adds data movement but does not add a trained time step. Record both counts.

## 6. Crafter environment integration

### 6.1 Keep Crafter optional

Crafter must remain an optional dependency. Detect it at module scope with
`importlib.util.find_spec`. Import it lazily only when the Crafter backend is
selected. Do not add an unconditional library import.

Request the `ci/optdeps` pull-request label. Add Crafter to the correct optional
test environment only if repository maintainers accept the dependency.

### 6.2 Use the native environment through TorchRL

Construct the pinned environment directly:

```text
crafter.Env(
    size=(64, 64),
    reward=True,
    length=10000,
    seed=<explicit root seed>,
)
```

Wrap it with the existing Gym/TorchRL adapter. Explicitly select one-hot action
encoding for the Torch policy and RSSM. Do not rely on Crafter's optional Gym
registration, because registration depends on the installed Gym version.

Do not create a public TorchRL environment class for the first implementation.
Keep the adapter private to the DreamerV3 example. If a public class is later
needed, it must have type hints, a Sphinx docstring, tests, configuration
parity, and a reference entry as required by `AGENTS.md`.

### 6.3 Observation contract

The transformed environment must expose:

```text
key: pixels
shape: [64, 64, 3]
dtype: uint8
range: 0 to 255
layout: HWC
```

Reuse the image descriptor, encoder, decoder, replay path, and unit-interval
reconstruction loss from the DMC vision branch. Remove DMC-specific checks
from generic image code. Crafter does not use MuJoCo, EGL, or OSMesa.

### 6.4 Action contract

Crafter receives an integer from 0 to 16. DreamerV3 must use a 17-value one-hot
action in the policy, imagination, RSSM, and replay path.

Add one internal action descriptor with:

- kind: continuous or categorical;
- TensorDict action key;
- model shape;
- environment representation;
- number of classes for categorical actions; and
- distribution settings.

Build the descriptor once from the transformed action specification. Pass it
to the policy, world model, learner, collector, and benchmark. Do not scatter
rank checks or backend checks through these components.

Use `ProbabilisticActor` and TorchRL's `OneHotCategorical` with the pass-through
gradient strategy. The pinned JAX categorical head does not pass its configured
uniform mixture to the distribution. Its effective mixture is therefore zero:

```text
probs = softmax(logits)
```

Pass probabilities to the distribution. Test log probability, entropy,
sampling, deterministic mode, and straight-through gradients against an
independent formula. Keep `0.01` as an explicit paper-semantics ablation only.

The environment adapter must convert the one-hot action to the integer expected
by Crafter. Keep that conversion at the environment boundary. Replay must keep
the one-hot action used by the RSSM.

### 6.5 Episode ending contract

Crafter returns one old-style `done` flag and `info["discount"]`:

- death: terminal, with discount 0;
- 10,000-step limit: truncation, with discount 1.

Expose `done`, `terminated`, and `truncated` correctly. Use an info-dict reader
and a TensorDict transform if needed. Do not infer terminal state from `done`
alone. Add a deterministic test for death and another for the time limit.

### 6.6 Seed contract

Pass the root seed to the Crafter constructor. Calling a later generic
`env.set_seed()` is not sufficient because Crafter's old API does not expose a
normal seed method.

The pinned JAX preset does not enable `env.crafter.use_seed`. Without this
field, the top-level agent seed does not clearly control the Crafter root seed.
For dissertation runs, add and record an explicit environment seed in both
stacks. Treat this as a disclosed reproducibility correction, not an algorithm
ablation.

Run a fixed-action environment check for each seed. Compare pixel hashes,
rewards, achievement counts, terminal flags, and truncation flags. Cross-stack
agent random streams do not need to match.

## 7. Achievement metrics and logging

Use Crafter's existing episode information or `crafter.Recorder` to retain all
22 achievement counts. Do not implement a custom collection loop.

Episode results must be TensorDict data. Aggregate them with TensorDict
operations after collection. The output must include:

- environment action at episode end;
- episode length and raw return;
- all 22 achievement counts;
- a Boolean success value for every achievement;
- per-seed success rates;
- per-seed Crafter score; and
- the exact interaction budget used for the calculation.

Extend the generalized DreamerV3 benchmark command. Do not add a second
benchmark framework. Store raw episode records so the score can be recomputed.
Use the official formula in a small pure function with independent numeric
tests. If it is public, add complete type hints and a Sphinx docstring. A
private helper is preferred unless another TorchRL example needs it.

The JAX `scores.jsonl` field named `episode/score` is raw episode return. The
Torch output must not repeat this misleading name. Use `episode_return` and
`crafter_score`.

## 8. Replay and memory

Reuse `DreamerV3ReplayRecordBuilder`. It must accept the image observation key
and the action descriptor without backend-specific branches.

For Crafter, verify that replay:

- stores one HWC `uint8` image per record;
- does not store a duplicate floating-point image;
- stores the 17-value one-hot action with a documented data type;
- preserves reset records and replay context;
- transfers image conversion to the encoder or loss boundary; and
- refreshes latent context without modifying sampled environment data.

Calculate bytes per stored record at startup. Include images, actions, rewards,
flags, latent context, generation metadata, and storage overhead. Check the
requested capacity against the process memory limit with a safety margin.

The nominal JAX replay capacity is five million records. A no-eviction
capacity just above the 1.1-million-record run budget is acceptable only when
the effective difference is disclosed and both arms retain all records used by
the run. Do not silently reduce capacity.

## 9. Learner changes

### 9.1 Eager learner first

Make the normal TorchRL losses support categorical actions before changing the
compiled learner. One eager update must cover:

- image encoding and reconstruction;
- RSSM observation and imagination steps;
- reward and continuation losses;
- categorical policy sampling;
- categorical log probability and entropy;
- actor, value, and replay-value losses;
- optimizer update; and
- slow target update.

Use existing loss modules and TensorDict keys. Do not create a parallel set of
plain tensor losses only for Crafter.

### 9.2 Fixed JAX fixtures

Create small dissertation-side fixtures from the pinned JAX implementation.
The upstream TorchRL tests must not import JAX. Fixtures must use fixed inputs
and weights and cover:

- image encoder output;
- image decoder output;
- unit-interval reconstruction loss;
- categorical probabilities after uniform mixing;
- categorical log probability and entropy;
- one RSSM observation step;
- one RSSM imagination step; and
- terminal and continuation targets.

Transpose convolution weights from JAX HWIO to Torch OIHW. Compare formulas and
outputs with declared tolerances. Do not require equal JAX and Torch random
samples.

## 10. Integrating the compiled performance branch

### 10.1 Rebase both lines on one fork base

Before integration:

1. fetch `origin/main` from the unography fork;
2. fetch `upstream/main`;
3. confirm whether `origin/main` contains the required upstream revision;
4. update the fork's `main` first if it is stale;
5. rebase the DMC vision branch on the refreshed `origin/main`; and
6. run all DMC vector and vision tests before adding Crafter.

Do not rebase a dirty worktree. Preserve unrelated untracked files. Use
`--force-with-lease`, not an unrestricted force push, after the rewritten
branch passes its tests.

Create the Crafter implementation branch from the tested, rebased DMC vision
tip. Do not implement against the current stale base.

### 10.2 Replay code, not the performance branch tip

The performance branch and DMC vision branch overlap in these files:

```text
sota-implementations/dreamer_v3/README.md
sota-implementations/dreamer_v3/config.yaml
sota-implementations/dreamer_v3/dreamer_v3_agent.py
sota-implementations/dreamer_v3/train.py
test/modules/test_dreamer_components.py
test/objectives/test_dreamer_v3.py
torchrl/modules/models/model_based.py
```

Do not merge the tip `31baba2d8`, because it includes dissertation evidence.
Rebase or recreate the code-only commit `c1170408c` on the same refreshed
base. Integrate that code into a separate dissertation branch after the eager
Crafter path passes.

Resolve conflicts by keeping the generic observation and action descriptors.
Do not restore vector-only assumptions from the performance branch. In
particular, remove fixed uses of:

- `("next", "observation")`;
- symlog vector reconstruction;
- Gaussian action noise;
- Gaussian log probability and entropy; and
- `action_spec.shape[0]` as the only action description.

Keep upstream pull requests separate from the dissertation integration branch.
The compiled learner can be reviewed as a performance pull request after its
correctness evidence is complete.

### 10.3 Generalize compiled randomness

The current compiled learner prepares Gaussian noise for continuous actions.
Add a categorical random tape for Crafter. Generate the random values outside
the captured update, then consume them inside the compiled imagination loop.
A uniform-CDF or Gumbel-max implementation is acceptable if it gives the same
categorical probabilities and a pass-through one-hot gradient.

Select the continuous or categorical implementation once during learner
construction. Do not branch on tensor values inside the compiled update. Keep
shapes and data types stable.

Test that successive compiled updates do not reuse the same categorical sample
tape. Test eager and compiled updates with the same fixed replay batch and
random tape.

### 10.4 Correctness gate for the performance code

The current performance evidence shows a large steady-state speedup, but its
same-branch eager-versus-compiled comparison still has optimizer-state and
actor-normalizer differences. Do not treat the branch as correctness-complete.

Before a Crafter learning run, compare eager and compiled modes for at least
two warm-up updates and two measured updates with identical inputs. Record:

- scalar losses;
- model parameters by module;
- gradients;
- optimizer parameter groups and state;
- actor return-normalization state;
- slow target state;
- replay context; and
- next policy state.

Set tolerances before examining the result. Relative error alone is invalid for
values close to zero. Use absolute error and a scaled RMSE or NRMSE. A learning
curve is additional evidence, not a replacement for this gate.

## 11. Tests required by the repository

Extend existing test files. Do not create a new test file when an existing
DreamerV3, environment-wrapper, distribution, replay, or objective test file
covers the behavior.

### 11.1 Normal CPU tests

Add focused tests for:

- Crafter configuration composition;
- action descriptor construction from a one-hot specification;
- 17-class probability mixing;
- categorical log probability and entropy;
- pass-through one-hot gradients;
- integer and one-hot action conversion;
- official score calculation from fixed achievement data;
- raw `uint8` image replay;
- reset and replay-context records;
- image reconstruction with a small model; and
- unchanged continuous-action Walker behavior.

Tests must assert values, not only shape, finiteness, key presence, or the lack
of an exception. Exercise public behavior rather than private attributes.

### 11.2 Optional-dependency tests

When Crafter is installed, test:

- observation and action specifications;
- deterministic reset from a constructor seed;
- a fixed-action pixel and reward trace;
- achievement information;
- death as termination;
- the 10,000-step limit as truncation;
- one collector batch;
- one replay sample; and
- one complete eager learner update with a small model.

Request the `ci/optdeps` label before pushing the CI-triggering commit.

### 11.3 GPU tests

Any CUDA-only test must have both `@pytest.mark.gpu` and a CUDA `skipif`.
Add small-model tests for:

- bfloat16 image learning;
- compiled categorical imagination;
- eager-versus-compiled update behavior; and
- fresh randomness across compiled updates.

Do not require Crafter rendering in a GPU test. The environment is a CPU
optional dependency; synthetic TensorDict inputs are sufficient for the GPU
learner test.

## 12. Benchmark requirements

Because the learner, replay, and image path are performance-sensitive, extend
the existing benchmark under `benchmarks/`. Do not place timing logic in
library code.

The benchmark must support:

- vector-continuous and image-categorical workloads;
- eager and compiled modes;
- FP32 and bfloat16 where supported;
- separate compile and steady-state times;
- exact batch, sequence, model, and imagination sizes;
- three or more steady-state timing windows;
- median and full range;
- updates per second and seconds per update;
- peak CUDA allocation and reservation; and
- source and dependency revisions.

Use the same code revision for eager and compiled comparisons. A comparison
against older upstream code answers a different question.

Do not launch a long Crafter job until a short A100 gate shows:

- the eager path completes with valid losses;
- the compiled path passes the declared correctness checks;
- the compiled steady-state median is faster than eager by a predeclared
  useful margin; and
- projected compile-time amortization fits within the full run.

After this gate, run a matched 20,000-record systems cohort before the one
million-action learning cohort. Use repeated unprofiled timing runs. Keep
profiler runs separate.

## 13. Documentation requirements

Update the DreamerV3 README with copyable commands for:

- a small Crafter smoke run;
- the large JAX repository reproduction;
- the exact one-million-action learning evaluation;
- eager mode;
- compiled mode; and
- achievement-score calculation.

State the Crafter version, source pin, model size, train ratio, and action
budget beside the commands. Explain that `episode_return` and `crafter_score`
are different metrics.

Do not add EIDF manifests, Docker files, run logs, plots, or checkpoints to the
upstream TorchRL change. Those belong in the dissertation repository. Do not
add notebooks, datasets, or videos.

## 14. Implementation order

Use this order so that semantic failures are isolated from compiler failures:

1. rebase and verify the current DMC vision branch;
2. add the generic action descriptor;
3. add the categorical policy and action boundary;
4. add the optional Crafter environment and protocol;
5. add achievement records and official score calculation;
6. verify replay and memory accounting;
7. pass one complete eager learner update;
8. validate against fixed JAX fixtures;
9. integrate the code-only compiled learner on a separate branch;
10. generalize compiled randomness and image reconstruction;
11. pass eager-versus-compiled correctness checks;
12. run short A100 performance gates; and
13. add dissertation-side EIDF jobs only after those gates pass.

Suggested one-line commits, with no co-author trailer or commit body:

```text
[Refactor] Describe DreamerV3 action spaces explicitly
[Algorithm] Add categorical actions to DreamerV3
[Algorithm] Add the DreamerV3 Crafter preset
[Test] Validate DreamerV3 Crafter behavior
[Doc] Document the DreamerV3 Crafter protocol
[Performance] Compile categorical DreamerV3 updates
[Performance] Measure complete DreamerV3 learner updates
```

Use only tags accepted by the repository at implementation time.

## 15. Pull-request boundaries

Prefer these upstream changes:

1. DMC vision and image reconstruction support from the current branch.
2. Generic categorical-action DreamerV3 support and the Crafter example.
3. Compiled complete-learner updates and their benchmark.

The dissertation integration branch may contain all three after each source
branch passes. Do not use the combined dissertation branch as the initial
upstream pull request.

The Crafter pull request must claim environment and algorithm support only. It
must not claim equal JAX learning performance until the multi-seed EIDF cohort
exists. The performance pull request must claim only the measured workload and
hardware.

## 16. Final acceptance checklist

Implementation is ready for short EIDF runs only when all items pass:

- [ ] The branch is based on the refreshed unography `main`.
- [ ] DMC Walker vector, DMC Cheetah, and DMC Walker vision tests still pass.
- [ ] The effective Crafter configuration matches the pinned JAX preset.
- [ ] Crafter 1.8.3 is an optional dependency.
- [ ] The environment root seed is explicit and recorded.
- [ ] Pixels are HWC `uint8` values under `pixels`.
- [ ] The policy, RSSM, and replay use 17-value one-hot actions.
- [ ] The environment receives integer actions from 0 to 16.
- [ ] The zero effective repository mixture and the optional 0.01 paper mixture
      are tested separately.
- [ ] Death is termination and the time limit is truncation.
- [ ] All 22 achievements are retained per episode.
- [ ] Official Crafter score matches independent test values.
- [ ] Replay context and first-update timing match JAX.
- [ ] Bytes per record and requested host memory are recorded.
- [ ] One complete eager learner update passes with a small model.
- [ ] Fixed JAX image, action, RSSM, and loss fixtures pass.
- [ ] CUDA-only tests have both required test markers.
- [ ] Optional-dependency tests are selected by the correct CI label.
- [ ] Vector-continuous DreamerV3 behavior does not regress.
- [ ] Compiled randomness changes between updates.
- [ ] Eager and compiled correctness tolerances pass.
- [ ] The complete-update benchmark reports compile and steady-state time.
- [ ] A short repeated A100 cohort proves useful throughput before long jobs.
- [ ] Documentation distinguishes repository return from official Crafter score.
- [ ] No run logs, checkpoints, plots, or dissertation infrastructure enter the
      upstream pull request.

## 17. Deferred review TODOs

The first implementation review found the following work. Complete it before an
upstream pull request or a long EIDF run.

- [ ] Rebase the branch on the current `unography:rl/main`.
- [ ] Keep two named Crafter protocols. The repository protocol leaves the
      environment seed unset. The dissertation protocol sets the same explicit
      environment seed in JAX and TorchRL.
- [ ] Match pixel conversion order across frameworks. The pinned JAX encoder
      casts pixels to BF16 before scaling. The current Torch encoder scales in
      FP32.
- [ ] Add fixed JAX fixtures for image encoding, image decoding, categorical
      outputs, RSSM steps, and loss targets.
- [ ] Retain `total_action_steps` in benchmark input and reject Crafter scores
      from runs that did not reach the requested action budget.
- [ ] Reject score aggregation with no completed episode, duplicate seeds, or
      different achievement names across seeds.
- [ ] Store and aggregate new episode and seed results with TensorDict.
- [ ] Move Crafter installation from the normal Linux suite to the optional
      dependency suite. Run the pull request with the `ci/optdeps` label.
- [ ] Add an A2C regression test for entropy lookup through a distribution
      subclass.
- [ ] Build the real `size200m` model and record its parameter count, first
      eager update, GPU memory, and replay memory on A100.
- [ ] Request enough EIDF host memory for the 1.2-million-record replay. Its
      current estimate is about 60 GiB before other process memory.
- [ ] Reduce repeated test code and keep the upstream change focused.
- [ ] Rewrite the new commits without commit bodies, co-author trailers, or
      session links.
- [ ] Reconcile the feature branch with the CUDA-graph implementation now in
      `unography:rl/main`. Compare current main with the older performance
      branch before keeping any overlapping optimization.
- [ ] Run only short repeated A100 gates until a useful throughput improvement
      and acceptable learning-state agreement are both shown.
