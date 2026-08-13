# Observation-conditioned acting fix proof

Base revision: `109783ec8ed55690ba1c4eda3aa429366bd2073d`.

## Before

With changing observations, the unmodified collector retained zero latents and
a fixed action distribution:

```text
observation_temporal_range=5.90389347076416
state_abs_max=0.0
belief_abs_max=0.0
loc_temporal_range=0.0
scale_temporal_range=0.0
episode_start_count=4
noninitial_belief_abs_max=0.0
```

## Fix

`build_real_world_actor()` shares the world model's encoder, posterior, and
prior with the imagination actor. On each real-environment step it:

1. symlog-transforms and encodes the current observation;
2. computes an observation-conditioned posterior state;
3. selects an action from posterior state and recurrent belief; and
4. advances belief into `("next", "belief")` for the following step.

Both collection and deterministic evaluation use this policy. No trained
parameters are copied.

## After

Run the same 32-frame, four-episode acceptance probe:

```bash
CUDA_VISIBLE_DEVICES='' python \
  sota-implementations/dreamer_v3/evidence/latent_fix/repro.py
```

The expected acceptance signature is:

- changing observations;
- nonzero posterior state and recurrent belief;
- changing action-distribution parameters;
- zero belief at all episode starts; and
- nonzero belief away from resets.

The complete captured result is in `after.txt`.

## Regression checks

- The real-world actor regression verifies parameter identity and
  observation-dependent state, belief, and action distribution.
- Full `test/objectives/test_dreamer_v3.py`: 58 passed.
- A 64-frame Pendulum train/evaluate smoke test completed evaluations at steps
  32 and 64.
