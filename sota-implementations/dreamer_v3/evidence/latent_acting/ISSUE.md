# DreamerV3 collector acts from permanently zero RSSM latents

## Summary

The DreamerV3 example gives the real-environment collector the imagination
actor directly. That actor consumes only `state` and `belief`. The environment
primer initializes both values to zero, but the collection policy has no
observation encoder, RSSM posterior, or RSSM transition that can update them.

Changing observations therefore produce a fixed action distribution. Randomly
sampled actions still vary, which can hide the problem; deterministic evaluation
repeatedly emits the same distribution mode.

## Reproduction

From revision `109783ec8ed55690ba1c4eda3aa429366bd2073d`:

```bash
CUDA_VISIBLE_DEVICES='' python \
  sota-implementations/dreamer_v3/evidence/latent_acting/repro.py
```

The script uses the checked-in Pendulum configuration, `make_env()`, and
`build_actor()`. It shortens episodes to eight steps and collects 32 frames so
that reset behavior is also covered.

## Observed result

```text
observation_temporal_range=5.90389347076416
state_abs_max=0.0
belief_abs_max=0.0
loc_temporal_range=0.0
scale_temporal_range=0.0
sampled_action_temporal_range=1.8649954795837402
episode_start_count=4
episode_start_belief_abs_max=0.0
noninitial_belief_abs_max=0.0

AssertionError: DreamerV3 collector never computed an
observation-conditioned state
```

The complete captured output is in `observed.txt`. The nonzero sampled-action
range is important: stochastic sampling varies while the underlying `loc` and
`scale` do not.

## Expected result

- The real-world policy shares the trained encoder, posterior, prior, and actor
  parameters.
- Each observation produces a posterior `state` before action selection.
- The prior carries recurrent `belief` into `("next", "belief")`.
- Episode reset restores the primed zero belief.
- Evaluation uses the same observation-conditioned real-world policy.

## Source-level cause

1. `build_actor()` creates a policy whose inputs are only `state` and `belief`.
2. `TensorDictPrimer` initializes those keys to zero.
3. `Collector` and `eval_episode_reward()` receive only that actor.
4. The encoder and RSSM posterior are used during replay training but are absent
   from real-environment acting.

This report reproduces entirely on CPU and is independent of device selection.
