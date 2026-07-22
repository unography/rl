# [BugFix] dreamer_v3 example: real-env acting policy is blind (never encodes observations into latents)

## Summary

In `sota-implementations/dreamer_v3/dreamer_v3.py` (added in #3621) the policy used for **data
collection and evaluation** is the bare actor head `actor(state, belief) -> action`. In the real
environment `state`/`belief` are supplied by a `TensorDictPrimer(default_value=0)` and **nothing
updates them from observations**, so the agent conditions on a constant zero latent at every
step. It cannot perceive the environment, so the world model trains but the policy never learns
and eval return stays at random level.

This is distinct from the two objective-level bugs (KL free-nats, reco double-symlog); it is a
wiring gap in the example script.

## Reproduction / evidence

Rollout the eval env with the current acting policy and print the latent + action each step:

```
t=0 |state|=0.000 |belief|=0.000 |obs|=1.355 action=[-0.0511]
t=1 |state|=0.000 |belief|=0.000 |obs|=1.936 action=[-0.0511]
...
t=5 |state|=0.000 |belief|=0.000 |obs|=4.772 action=[-0.0511]
```

The observation varies each step; `state`/`belief` stay at 0; the action is **identical** every
step. The agent is a fixed open-loop controller.

Symptom in a training run: world-model losses fall (reco 0.87 -> 0.03, KL grows) but eval return
stays flat at ~-1350 on Pendulum-v1 (random ~-1300, solved ~-150) and actor loss ~0.

## Root cause

`build_actor` returns only the actor head, and the collector (`Collector(explore_env,
actor_model, ...)`) and eval (`eval_episode_reward(eval_env, actor_model, ...)`) use it directly.
The example never composes the observation encoder + RSSM posterior (which turn an observation
into a latent) or the RSSM prior recurrence (which advances the belief) into the acting policy --
even though the world model already builds all of them.

## Fix

Add a recurrent acting policy that mirrors the DreamerV3 reference acting path (`agent.py:policy`
-> `rssm.py:_observe`) and TorchRL's own classic dreamer `_dreamer_make_actor_real`. Per step:

1. `encoder`   : observation -> encoded_latents
2. `posterior` : (belief, encoded_latents) -> state
3. `actor`     : (state, belief) -> action   (the shared actor head)
4. `prior`     : (state, belief, action) -> `("next", "belief")`   (advance belief for next step)

Plus, matching the reference's episode-reset handling, an `InitTracker` + an initial-belief step
that sets `belief = prior(0, 0, 0)` on the first step of each episode (JAX masks the carry to
zero and runs `_core(0,0,0)` before the first posterior). All modules are the shared, trained
instances -- no new parameters. The collector and eval use this policy; imagination keeps the
bare actor head (correct there, since the world model supplies the latents).

Post-fix rollout (latent now responds to observations; action varies):

```
t=0 is_init=True  |belief|=0.6475 (= prior(0,0,0)) action=[-0.2048]
t=1 is_init=False |belief|=1.0549 action=[ 0.1475]
t=2 is_init=False |belief|=1.2825 action=[ 0.3160]
```

## Impact

With the fix the example learns Pendulum-v1: eval return climbs from ~-1350 (random) to ~-90
(near-optimal) by ~16-20k env steps. Without it the example never learns any task, which also
masks the performance impact of the objective-level bugs.

## Environment

- TorchRL `main` @ `ae421b98d` (example unchanged since #3621)
- Reference: danijar/dreamerv3 @ `e3f0224` (`agent.py:115` policy, `rssm.py:75` `_observe`)
- Affected file: `sota-implementations/dreamer_v3/dreamer_v3.py`
