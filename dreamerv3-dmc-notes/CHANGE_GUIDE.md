# DreamerV3 JAX parity change guide

This guide explains the commits by `k4r4n.dhruv@gmail.com` on the
`dreamerv3-jax-parity-dmc` branch. The comparison base is `origin/main` at
`71f103e8`. The JAX reference is `../dreamerv3` at commit `e3f0224`.

Line numbers in this guide refer to the current files. They do not refer to the
old file in each commit.

## 1. Purpose and limits

The branch has two goals:

1. Make the TorchRL DreamerV3 algorithm agree with the JAX DreamerV3 algorithm.
2. Run the same `dmc_walker_walk` proprioception task and compare measured data.

The work has three types:

| Type | Meaning |
|---|---|
| Algorithm parity | The Torch code implements the same calculation as JAX. |
| Run parity | The example uses the same task and most of the same settings. |
| Torch support | The change improves speed, memory use, tests, or comparison tools. It does not change the algorithm. |

The branch does not yet prove full learning-curve parity. The short runs give
useful evidence, but the full-length multi-seed runs are not complete. See
`dreamerv3-dmc-notes/RESULTS.md`.

The public TorchRL classes keep compatibility defaults, so `jax_core`, RMS
normalization, and `unimix` are opt-in. The DMC example enables them.

## 2. Data flow after the changes

The following flow is the main design:

```text
DMC observation fields
  -> join fields and convert to float32
  -> symlog encoder
  -> RSSM posterior and recurrent belief
  -> real action
  -> replay buffer
  -> world-model loss
  -> imagined RSSM trajectories
  -> actor loss and value loss
```

The code reuses each trained object where it is needed. The world model and real
policy share the encoder, posterior, and prior. The world model and imagination
environment share the prior and reward head. The real policy and imagined actor
loss share the actor. Separate prior or reward objects would make the imagined
trajectories use unrelated parameters.

## 3. Model and RSSM changes

### 3.1 Block recurrent layer

- Torch: `torchrl/modules/models/model_based_v3.py:20-62`
- JAX definition: `../dreamerv3/embodied/jax/nets.py:254-278`
- JAX use: `../dreamerv3/dreamerv3/rssm.py:149-152`
- Main commits: `9b3cea8c`, `0082a0ce`, `30b7c77f`

`BlockLinear` splits the feature axis into independent blocks. Each block has
its own weight. JAX uses `einsum`. Torch uses `bmm`.

The block structure reduces the parameter count of the large recurrent state.
The later `bmm` fix also prevents a large broadcasted matrix operation. The
calculation stays the same.

`BlockLinear` first uses its compatibility-oriented uniform initializer at
`model_based_v3.py:51-52`. The DMC example then includes `BlockLinear` in
`_apply_jax_init()` at `sota-implementations/dreamer_v3/dreamer_v3.py:319-331`
and applies it to the full prior at line 362. Thus the parity run does use the
JAX `trunc_normal_in` rule. A direct library user who constructs
`RSSMPriorV3(jax_core=True)` does not get this second initialization pass.

### 3.2 JAX-style RSSM recurrent core

- Torch options: `torchrl/modules/models/model_based_v3.py:140-210`
- Torch calculation: `torchrl/modules/models/model_based_v3.py:315-351`
- JAX calculation: `../dreamerv3/dreamerv3/rssm.py:135-159`
- JAX settings: `../dreamerv3/dreamerv3/configs.yaml:89-91`
- Main commit: `9b3cea8c`

When `jax_core=True`, Torch uses the DreamerV3 block GRU instead of a standard
PyTorch `GRUCell`. It does these operations:

1. It soft-clips the action.
2. It makes separate projections for the old deterministic state, stochastic
   state, and action.
3. It applies RMS normalization and SiLU.
4. It calculates reset, candidate, and update gates in blocks.
5. It uses `sigmoid(update - 1)`.

The `-1` bias makes the initial gate keep more old state. This gives the model a
useful memory bias at the start of training.

### 3.3 Prior and posterior heads

- Torch prior: `torchrl/modules/models/model_based_v3.py:221-234`
- Torch posterior: `torchrl/modules/models/model_based_v3.py:403-462`
- JAX prior: `../dreamerv3/dreamerv3/rssm.py:161-171`
- JAX posterior: `../dreamerv3/dreamerv3/rssm.py:75-92`
- JAX normalization: `../dreamerv3/embodied/jax/nets.py:361-386`
- Main commits: `bb34d886`, `0082a0ce`

The heads now support the JAX order: linear layer, RMS normalization, then SiLU.
The prior also supports two hidden layers. JAX uses `imglayers: 2` for the prior
and `obslayers: 1` for the posterior.

The DMC size-1M model uses a hidden width of 64 and a recurrent deterministic
state of 512. These values have different functions. Earlier code used the 512
value in places where JAX uses 64. This made the Torch model too large.

### 3.4 Uniform mixture for categorical states

- Torch sample sites: `torchrl/modules/models/model_based_v3.py:278,460`
- Torch helper: `torchrl/modules/models/model_based_v3.py:656-686`
- JAX distribution and straight-through sample:
  `../dreamerv3/embodied/jax/outs.py:208-224,243-270`
- Torch KL mixture: `torchrl/objectives/dreamer_v3.py:281-305`
- JAX setting: `../dreamerv3/dreamerv3/configs.yaml:91`
- Main commit: `10dcbd90`

The code mixes 1 percent uniform probability into every categorical state:

```text
mixed probability = 0.99 * softmax probability + 0.01 / class count
```

This operation prevents a class probability from becoming zero. Torch then
draws a hard one-hot state and uses a straight-through gradient.

### 3.5 Faster observation rollout and optional compilation

- Torch no-sample path: `torchrl/modules/models/model_based_v3.py:283-313`
- Torch fast rollout: `torchrl/modules/models/model_based_v3.py:522-639`
- Torch compile entry: `torchrl/modules/models/model_based_v3.py:641-653`
- JAX scan: `../dreamerv3/dreamerv3/rssm.py:61-92`
- Main commits: `0082a0ce`, `6a2c1663`

During observation, the posterior state replaces the prior state. Therefore,
the prior sample is unused. The Torch fast path calculates prior logits at each
step but does not draw the sample. JAX also skips this sample, but it calculates
the prior logits later in one vectorized operation for the KL loss at
`../dreamerv3/dreamerv3/rssm.py:120-126`.

The fast path also removes TensorDict work from each time step. It reads tensors
once, runs the recurrent loop, and writes the results once. `compile_scan()` can
compile this tensor-only loop for a fixed sequence length.

The JAX counterpart is a scanned and JIT-compiled recurrence. The Torch
`compile_scan()` method is a performance counterpart. It is not a direct JAX
source translation.

Tests:

- Fast-path first-step, shape, and no-prior-sample checks:
  `test/objectives/test_dreamer_v3.py:971-1055`. Later stochastic states are not
  compared because the two paths consume different random draws.
- Compiled path versus eager path, including gradients:
  `test/objectives/test_dreamer_v3.py:1057-1145`
- Prior layer count: `test/objectives/test_dreamer_v3.py:1147-1171`

There is no focused numerical test for `BlockLinear`. There is also no focused
stochastic distribution test for `unimix`.

## 4. World-model loss changes

### 4.1 Separate dynamics and representation KL losses

- Torch helper: `torchrl/objectives/dreamer_v3.py:227-317`
- Torch use: `torchrl/objectives/dreamer_v3.py:503-569`
- JAX use: `../dreamerv3/dreamerv3/rssm.py:120-133`
- JAX weights: `../dreamerv3/dreamerv3/configs.yaml:86,91`
- Main commits: `808a63b9`, `b4a848f0`

The code now makes two KL terms:

- The dynamics term stops the current posterior probabilities, so its direct KL
  gradient goes to the prior logits. Its JAX weight is 1.0.
- The representation term stops the current prior probabilities, so its direct
  KL gradient goes to the posterior logits. Its JAX weight is 0.1.

The code sums KL over the categorical state dimensions before it applies the
one-nat minimum. The old order applied the minimum to every category. That order
made the effective minimum too large and removed useful gradients.

Tests at `test/objectives/test_dreamer_v3.py:354-397` check that the combined KL
reaches both sets of logits and that a zero KL is floored. They do not prove the
per-term gradient routing, the two beta weights, or the sum-before-clamp order.

### 4.2 Reconstruction loss

- Torch: `torchrl/objectives/dreamer_v3.py:577-597`
- JAX head: `../dreamerv3/embodied/jax/heads.py:127-130`
- JAX output loss: `../dreamerv3/embodied/jax/outs.py:129-141`
- JAX call: `../dreamerv3/dreamerv3/agent.py:178-182`
- Main commits: `b6be4de3`, `f3386717`

The target is `symlog(observation)`. The decoder output is already in symlog
space. The loss must not apply `symlog` to the decoder output again.

The loss also sums the observation feature dimensions. It then averages the
batch and time dimensions. This keeps the reconstruction term at the same scale
as JAX. For walker, an average over its 24 observation values would make this
term 24 times too small.

### 4.3 Two-hot reward and value space

- Torch bins: `torchrl/objectives/dreamer_v3.py:100-134`
- Torch stable decode: `torchrl/objectives/dreamer_v3.py:184-219`
- Torch reward loss: `torchrl/objectives/dreamer_v3.py:599-617`
- JAX bins: `../dreamerv3/embodied/jax/heads.py:132-144`
- JAX decode and loss: `../dreamerv3/embodied/jax/outs.py:273-330`
- JAX settings: `../dreamerv3/dreamerv3/configs.yaml:98-101`
- Main commits: `32b83528`, `0082a0ce`

The reward and value heads predict 255 logits. The centers are mathematically
`symexp(linspace(-20, 20, 255))`. Both implementations build one half and mirror
it so the stored grid is exactly symmetric. The current JAX code performs
interpolation and expectation in reward space. The Torch `bin_space="reward"`
option now does the same.

The bins can be close to 500 million. A simple left-to-right sum can give a
nonzero result for a symmetric distribution because of floating-point error.
The stable decoder pairs negative and positive bins before it sums them. Thus a
zero-initialized symmetric head gives exactly zero.

Tests at `test/objectives/test_dreamer_v3.py:852-880` check the reward-space grid,
symmetry, uniform zero decode, interpolation, and finite model reward loss in
both bin spaces. They do not directly check actor/value decoding in both spaces.

### 4.4 Continue target and truncation

- Torch options: `torchrl/objectives/dreamer_v3.py:495-538`
- Torch target: `torchrl/objectives/dreamer_v3.py:630-657`
- JAX target: `../dreamerv3/dreamerv3/agent.py:172-177`
- JAX settings: `../dreamerv3/dreamerv3/configs.yaml:106-107`
- Main commits: `dd83f67e`, `0082a0ce`, `8fd20f06`

The continue target uses `not terminated`. It does not use `not done`. A DMC
episode can end because of its time limit. This is a truncation, not a terminal
state. The model must still permit value bootstrap at that boundary.

When `contdisc=True`, Torch multiplies the target by `1 - 1 / horizon`. For the
JAX horizon of 333, this value is approximately 0.997.

Tests:

- Termination versus truncation: `test/objectives/test_dreamer_v3.py:882-929`
- Continuous target scale: `test/objectives/test_dreamer_v3.py:931-969`

## 5. Actor and value changes

### 5.1 Percentile return normalization

- Torch: `torchrl/objectives/dreamer_v3.py:665-702`
- Torch use: `torchrl/objectives/dreamer_v3.py:1135-1142`
- JAX normalizer: `../dreamerv3/embodied/jax/utils.py:16-91`
- JAX use: `../dreamerv3/dreamerv3/agent.py:407-410`
- JAX setting: `../dreamerv3/dreamerv3/configs.yaml:111`
- Main commit: `b863d700`

The normalizer tracks moving estimates of the 5th and 95th return percentiles.
It divides the advantage by at least 1, or by the percentile range when that
range is larger. This keeps actor gradients usable for tasks with different
reward scales.

### 5.2 JAX-style imagined actor and value path

- Torch entry: `torchrl/objectives/dreamer_v3.py:977-980`
- Torch return calculation: `torchrl/objectives/dreamer_v3.py:1070-1087`
- Torch loss: `torchrl/objectives/dreamer_v3.py:1089-1159`
- JAX imagination: `../dreamerv3/dreamerv3/agent.py:188-215`
- JAX loss: `../dreamerv3/dreamerv3/agent.py:382-446`
- JAX return calculation: `../dreamerv3/dreamerv3/agent.py:482-490`
- Main commit: `014ad57a`

Torch splits the JAX `imag_loss` work between `DreamerV3ActorLoss` and
`DreamerV3ValueLoss`. The path uses this sequence:

1. Imagine `H + 1` latent features.
2. Predict imagined reward, continue probability, and online value.
3. Calculate survival weights and lambda returns.
4. Normalize the advantage.
5. Apply the stopped gradients that correspond to the JAX source.
6. Add the analytic entropy term.
7. In the separate value-loss call, evaluate the slow critic for regularization.

The gradient stops are part of the algorithm. They control whether an actor,
critic, or world-model parameter receives each gradient.

This correspondence comes from source tracing. There is no focused numerical
test for the complete imagined-loss parity path.

### 5.3 Two-hot critic, slow critic, and slow regularization

- Torch value options: `torchrl/objectives/dreamer_v3.py:1277-1314`
- Torch point loss: `torchrl/objectives/dreamer_v3.py:1339-1361`
- Torch EMA update: `torchrl/objectives/dreamer_v3.py:1448-1464`
- Torch imagined value loss: `torchrl/objectives/dreamer_v3.py:1466-1509`
- JAX use: `../dreamerv3/dreamerv3/agent.py:397-422`
- JAX EMA model: `../dreamerv3/embodied/jax/utils.py:94-119`
- JAX settings: `../dreamerv3/dreamerv3/configs.yaml:101,108-110`
- Main commits: `32b83528`, `ba5612fb`

The critic predicts a two-hot value distribution. The return calculation uses
the decoded scalar.

A frozen slow critic follows the live critic with an exponential moving average
at rate 0.02. The live critic learns the return target and also learns toward
the detached slow prediction. This slow regularization improves stability.

The DMC setting has `slowtar: False`. Thus the online critic supplies the
lambda-return targets. The slow critic supplies only regularization, not the
bootstrap target.

There is no focused test for the slow regularizer or the EMA update.

### 5.4 Replay value loss (`repval`)

- Torch: `torchrl/objectives/dreamer_v3.py:1363-1446`
- JAX call: `../dreamerv3/dreamerv3/agent.py:218-235`
- JAX loss: `../dreamerv3/dreamerv3/agent.py:449-479`
- JAX settings: `../dreamerv3/dreamerv3/configs.yaml:86,109,115-116`
- Main commit: `0082a0ce`

The normal value loss uses imagined sequences. `repval` also trains the critic
on real replay sequences. It uses the first lambda-return from an imagination
rollout started at each replay state as the bootstrap. The configured loss scale
is 0.3.

With `repval_grad=True`, this loss also sends a gradient into the world-model
features. Torch controls this with the model-loss `detach_output` option.

Tests:

- Hand-calculated return and critic gradient:
  `test/objectives/test_dreamer_v3.py:1173-1228`. This does not test the gradient
  into the world model.
- Termination compared with truncation and no terminal flag:
  `test/objectives/test_dreamer_v3.py:1230-1269`

## 6. Example and training changes

### 6.1 DMC environment and action range

- Torch: `sota-implementations/dreamer_v3/dreamer_v3.py:89-131`
- Torch setting: `sota-implementations/dreamer_v3/config_dmc.yaml:13-19`
- JAX environment setting: `../dreamerv3/dreamerv3/configs.yaml:36,178-182`
- JAX environment construction and wrappers:
  `../dreamerv3/dreamerv3/main.py:212-258`
- Main commits: `1bfbb0cb`, `01f5f433`, `6bf9ecab`

The example can now select Gym or DM Control. DMC walker gives separate
`orientations`, `height`, and `velocity` fields. Torch joins these fields in a
stable order and converts them from `float64` to `float32`.

`ActionScaling` exposes the normalized action coordinate system and maps it to
the native environment range. DMC already has the range `[-1, 1]`, so this map
is an identity operation for DMC.

This transform does not clip actions. The DreamerV3 `bounded_normal` policy is a
plain Normal with a `tanh`-bounded mean; its samples can still be less than -1 or
greater than 1. JAX adds `ClipAction` after action normalization at
`../dreamerv3/dreamerv3/main.py:249-258`. The Torch example has no equivalent
clip transform. This is a remaining run-parity gap.

### 6.2 Symlog encoder and corrected network construction

- Torch encoder: `sota-implementations/dreamer_v3/dreamer_v3.py:237-253`
- Torch MLPs and initialization:
  `sota-implementations/dreamer_v3/dreamer_v3.py:256-335`
- Torch shared heads: `sota-implementations/dreamer_v3/dreamer_v3.py:338-382`
- Torch decoder feature: `sota-implementations/dreamer_v3/dreamer_v3.py:441-451`
- JAX encoder and decoder: `../dreamerv3/dreamerv3/rssm.py:179-250,253-305`
- JAX network settings: `../dreamerv3/dreamerv3/configs.yaml:91-101,120-123`
- Main commits: `bb34d886`, `f3386717`, `0082a0ce`

The encoder applies `symlog` to vector observations. This is important for
walker velocity values, which can be much larger than the other values.

For these network properties, the example now uses the JAX activation,
normalization, layer count, width, weight initialization, and output scale. The
reward and value output layers start at zero. The policy output-layer weights
are multiplied by 0.01. The compute type still differs: Torch trains in
float32, while the JAX configuration selects bfloat16.

The decoder now receives both the stochastic state and the deterministic
belief. JAX calls this combined value the feature. Without the belief, the
stochastic state must store information that the belief already has.

### 6.3 DreamerV3 optimizer and one joint update

- Torch optimizer: `sota-implementations/dreamer_v3/dreamer_v3.py:134-205`
- Torch parameter set: `sota-implementations/dreamer_v3/dreamer_v3.py:900-918`
- Torch update: `sota-implementations/dreamer_v3/dreamer_v3.py:1150-1162`
- JAX optimizer construction: `../dreamerv3/dreamerv3/agent.py:342-379`
- JAX optimizer use: `../dreamerv3/dreamerv3/agent.py:74-78,137-143`
- Main commits: `8ee32490`, `f864b8ee`

The optimizer performs these steps in this order: adaptive gradient clipping,
RMS scaling, momentum, and learning-rate warmup. RMS scaling before momentum is
not the same as Adam.

One optimizer updates the world model, actor, and critic. This agrees with the
JAX agent and makes the shared module gradients unambiguous.

The example also applies a global gradient-norm limit of 100 after AGC at
`sota-implementations/dreamer_v3/dreamer_v3.py:1158-1160`. JAX has no such
operation. It is intended as a high safety limit, but it is a difference if it
activates.

### 6.4 Recurrent real-world policy and reset state

- Torch: `sota-implementations/dreamer_v3/dreamer_v3.py:571-660`
- Torch collector use: `sota-implementations/dreamer_v3/dreamer_v3.py:951-987`
- JAX policy: `../dreamerv3/dreamerv3/agent.py:101-135`
- JAX initial state: `../dreamerv3/dreamerv3/rssm.py:45-49,75-92`
- Main commits: `ea311e0f`, `62ded70a`

The actor head needs latent features. The old collector did not update those
features from each real observation. The new policy encodes the observation,
updates the posterior state, selects an action, and carries the recurrent belief
to the next step.

At an episode reset, it feeds zero stochastic state, zero belief, and zero
action into the recurrent core. The core output becomes the belief that the
first posterior uses. `InitTracker` tells the policy when to do this.

### 6.5 Shared world model in imagination

- Torch construction: `sota-implementations/dreamer_v3/dreamer_v3.py:338-472`
- Torch reuse: `sota-implementations/dreamer_v3/dreamer_v3.py:692-738,765-823`
- JAX construction and use: `../dreamerv3/dreamerv3/agent.py:55-75,188-214`
- Main commit: `b93c1d33`

The world model and imagination environment now hold the same prior and reward
module objects. Thus imagination uses the trained model. Before this fix, the
construction did use separate newly initialized modules.

### 6.6 Bounded-normal actor and continue head

- Torch actor: `sota-implementations/dreamer_v3/dreamer_v3.py:475-557`
- Torch continue head and wiring:
  `sota-implementations/dreamer_v3/dreamer_v3.py:376-381,459-468,794-797,856-871`
- Torch imagined use: `torchrl/objectives/dreamer_v3.py:1119-1135`
- JAX actor: `../dreamerv3/embodied/jax/heads.py:146-155`
- JAX continue target and imagined use:
  `../dreamerv3/dreamerv3/agent.py:172-177,203-214,382-405`
- Main commits: `dc4330db`, `dd83f67e`

The bounded-normal actor has a `tanh`-bounded mean and a standard deviation
between 0.1 and 1.0. It is a plain Normal, not a tanh-squashed distribution.
Therefore, it has analytic entropy, but its samples are not bounded. The
continue head outputs one logit. Its sigmoid is the predicted survival
probability, which the actor and value losses use in trajectory discounts.

### 6.7 Replay memory, sampler cache, and compiled scan

- Torch replay: `sota-implementations/dreamer_v3/dreamer_v3.py:989-1017`
- Torch batch transfer: `sota-implementations/dreamer_v3/dreamer_v3.py:1061-1066`
- Torch compile selection: `sota-implementations/dreamer_v3/dreamer_v3.py:789-792`
- JAX replay settings: `../dreamerv3/dreamerv3/configs.yaml:39-46`
- JAX scan: `../dreamerv3/dreamerv3/rssm.py:61-73`
- Main commits: `4c438739`, `0082a0ce`, `57cd0507`, `6a2c1663`

Torch keeps the large replay buffer in host memory. It moves only each sampled
batch to the accelerator. The sampler caches trajectory boundaries until the
next buffer extension. These changes reduce accelerator memory use and remove
repeated full-buffer scans.

These are Torch support changes. They do not change the DreamerV3 algorithm.

## 7. DMC configuration map

The main Torch preset is
`sota-implementations/dreamer_v3/config_dmc.yaml:13-97`. Its main JAX sources
are `../dreamerv3/dreamerv3/configs.yaml:10-11,39-55,72-75,85-123,178-182`.

| Setting | Torch | JAX | Purpose |
|---|---:|---:|---|
| Batch size | 16 | 16 | Number of sequences per update. |
| Sequence length | 64 | 64 | Time steps in each sequence. |
| Concurrent environments | 16 | 16 | Independent trajectory streams. |
| Deterministic state | 512 | 512 | RSSM recurrent memory. |
| RSSM hidden width | 64 | 64 | Width of RSSM internal MLPs. |
| Categoricals and classes | 32 x 4 | 32 x 4 | Stochastic state shape. |
| Reward and value bins | 255 | 255 | Two-hot output size. |
| Imagination length | 15 | 15 | Number of imagined transitions. |
| Horizon | 333 | 333 | Long-term discount scale. |
| Learning rate | 4e-5 | 4e-5 | Peak optimizer rate. |
| AGC | 0.3 | 0.3 | Adaptive gradient clip ratio. |
| Warmup | 1000 | 1000 | Optimizer warmup updates. |
| Dynamics / representation KL | 1.0 / 0.1 | 1.0 / 0.1 | KL loss weights. |
| Replay value | 0.3 | 0.3 | Real-sequence critic loss weight. |
| Train ratio | 1024 | 1024 | Sampled training steps per environment step. |

The following settings do not agree exactly:

| Setting | Torch | JAX | Effect |
|---|---:|---:|---|
| Replay capacity | 1,000,000 | 5,000,000 | Torch saves host memory. One million is enough for this run length. |
| Requested run length | 1,000,000 | 1,100,000 | The reference requests 100,000 more steps. |
| Environment execution | One-process `SerialEnv` | 16 worker processes | Both maintain independent states and batch the policy call, but scheduling and process overhead differ. |
| Update ordering | Insert 16 transitions, then run 16 updates | Insert and update once per worker callback | Replay contents differ slightly between the 16 updates in a driver tick. |
| Replay sequence state | Zero belief at each sampled window | Carries context between consecutive chunks | The first Torch steps have less history. |
| Training compute type | float32 | bfloat16 | Numerical behavior and speed can differ. |
| Environment action clip | None | `ClipAction` | Torch Normal samples can leave `[-1, 1]`. |
| Extra global gradient clip | 100 | None | This changes an update only if the limit activates. |

Torch uses 16 updates after each 16 collected frames. Each update samples
`16 * 64` values. Thus the train ratio is:

```text
16 updates * 16 sequences * 64 steps / 16 collected frames = 1024
```

Each collector batch has shape `[environment=16, time=1]`. The replay buffer
extends along dimension 1 and stores the data as `[time, environment]`, so each
new batch appends one step to every environment column. `SliceSampler` therefore
draws contiguous 64-step windows without crossing environments or requiring a
64-step collector batch. Replay capacity remains 1,000,000 total transitions;
it is not divided by the environment count. Training starts after
`2 * 16 * 64 = 2048` transitions, approximately when the 16 streams provide
1024 valid 64-step starts.

## 8. V3-off ablation

- Torch: `sota-implementations/dreamer_v3/config_dmc_v3off.yaml:1-88`
- Main commits: `24c28a71`, `f3386717`, `60085910`, `0082a0ce`, `8fd20f06`

This configuration is not a JAX parity preset. It is an experiment. It keeps
the correctness fixes, such as recurrent acting and correct KL/reconstruction
losses. It disables the main DreamerV3 features:

- It uses a scalar critic and a tanh-normal actor at lines 51-52.
- It disables `unimix` and the block core at lines 55-56.
- It uses Adam at lines 63-66.
- It gives both KL terms weight 1.0 at lines 73-74.
- It disables the new imagined loss at lines 70 and 78.
- It disables the slow critic and return normalization at lines 81-82.
- It disables `repval` at line 62.

The experiment asks one question: does the DreamerV3 feature set close the
performance gap, or do only the correctness fixes close it?

## 9. Tests and evidence tools

The branch extends `test/objectives/test_dreamer_v3.py`. Important coverage is
listed in the sections above. Important direct coverage gaps include:

- No focused numerical test for `BlockLinear` against the JAX equation.
- No automated test of the JAX-style `BlockLinear` initialization distribution.
- No focused test of nonzero `unimix` stochastic behavior.
- No focused test of separate KL gradient routing, beta weights, or the
  sum-before-clamp reduction order.
- No focused numerical test of reconstruction target and event reduction.
- No focused test of `_ReturnNormalizer`.
- No focused numerical comparison of the complete imagined actor/value path.
- No focused actor/value decode test for both two-hot spaces.
- No focused test of slow regularization or the slow-value EMA update.
- No test that `repval` sends its feature gradient into the world model.
- No focused example-runtime test of action handling or shared-module identity.

The main runtime wiring is at these lines:

- Loss construction: `sota-implementations/dreamer_v3/dreamer_v3.py:825-898`
- Model, actor, and value calls:
  `sota-implementations/dreamer_v3/dreamer_v3.py:1087-1115`
- Replay-value call: `sota-implementations/dreamer_v3/dreamer_v3.py:1116-1148`
- Joint update and slow-value update:
  `sota-implementations/dreamer_v3/dreamer_v3.py:1150-1162`

The branch also adds local experiment support. These files do not have direct
JAX algorithm counterparts:

| File and current lines | Function |
|---|---|
| `dreamerv3-dmc-notes/scripts/check_param_parity.py:1-105` | Compares Torch module parameter counts with the JAX counts. |
| `dreamerv3-dmc-notes/scripts/run_dmc_parity.py:1-182` | Runs seeds and produces comparison curves. |
| `dreamerv3-dmc-notes/scripts/compare_a100_runs.py:1-312` | Compares measured Torch and JAX logs. |
| `dreamerv3-dmc-notes/scripts/extract_jax_curve.py:1-98` | Converts the published JAX score data to CSV and a plot. |

The CSV files and plots under `dreamerv3-dmc-notes/reference/` and
`dreamerv3-dmc-notes/plots/` are evidence artifacts. They do not change training.

## 10. Commit ledger

The following ledger lists the author commits in branch order. It excludes the
final merge commit from upstream main.

### Core algorithm and example parity

| Commit | Change |
|---|---|
| `b93c1d33` | Share trained prior and reward modules with imagination. |
| `808a63b9` | Apply free nats after the categorical KL sum. |
| `ba5612fb` | Add the slow critic and slow regularization. |
| `b863d700` | Add percentile return normalization. |
| `10dcbd90` | Add the uniform categorical mixture. |
| `b6be4de3` | Do not apply symlog to the reconstruction prediction. |
| `b4a848f0` | Add separate dynamics and representation KL weights. |
| `9b3cea8c` | Add the JAX-style block recurrent core. |
| `8ee32490` | Add the DreamerV3 optimizer. |
| `bb34d886` | Add RMSNorm and SiLU. |
| `dc4330db` | Add the bounded-normal actor and analytic entropy. |
| `dd83f67e` | Add the continue head. |
| `f864b8ee` | Use one joint optimizer. |
| `1bfbb0cb` | Normalize the environment action range. |
| `32b83528` | Add the two-hot critic and scalar decode. |
| `014ad57a` | Port the JAX imagined actor/value loss. |
| `ea311e0f` | Make the real-environment acting policy recurrent. |
| `62ded70a` | Match the JAX reset belief. |

### DMC path, parity corrections, and performance

| Commit | Change |
|---|---|
| `01f5f433` | Add a Gym/DMC environment factory. |
| `1f3950bb` | Add the DMC walker size-1M preset. |
| `24c28a71` | Add V3 feature switches for the ablation. |
| `6bf9ecab` | Move the environment, networks, loss buffers, sampled batches, and recurrent tensors to the configured device. |
| `d6f004f5` | Correct reward bins and MLP depth. |
| `4c438739` | Cache replay trajectory boundaries. |
| `60085910` | Align non-ablation V3-off settings with the main DMC preset. |
| `f3386717` | Add symlog input, correct reconstruction reduction, and warmup. |
| `0082a0ce` | Correct architecture, two-hot space, `repval`, and rollout speed. |
| `57cd0507` | Keep replay in host memory. |
| `30b7c77f` | Replace broadcast block matmul with `bmm`. |
| `6a2c1663` | Test compiled and eager RSSM equivalence. |
| `8fd20f06` | Match the JAX continuous continue target. |

### Documentation, harness, and measured evidence

| Commits | Change type |
|---|---|
| `d02d4d53`, `66e08fb6` | Plans, harness wiring, reference assets, and explanations. |
| `8423c33e`, `b79a0d8c`, `a5b3318a`, `1ccfe3d6` | Source-review findings and explanations. |
| `a91e744a` | Plot-only harness mode. |
| `006c885f`, `4434fc00`, `841a1690`, `beb688c1`, `ecd36854`, `4fc5e229`, `7aa0d45a` | Measured loss and curve evidence. |
| `720ebf58` | Add measured V3-off overlay support to the comparison script. |
| `b976314f` | Remove synthetic placeholder curve artifacts. |
| `6eae8731` | Add Torch-to-JAX log-comparison script support. |

## 11. Short reading order

For a fast review, read these parts in order:

1. Section 2 for the data flow.
2. Sections 3.2, 4.1, 4.3, 5.2, and 6.4 for the largest algorithm changes.
3. Section 7 for configuration agreement and known differences.
4. Section 9 for test coverage and gaps.
5. `dreamerv3-dmc-notes/RESULTS.md` for measured results.
