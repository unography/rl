# Bug: the dreamer_v3 example's acting policy was blind (no obs -> latent)

Separate from the two library (`objectives/dreamer_v3.py`) bugs. This one is in the example
`sota-implementations/dreamer_v3/dreamer_v3.py` (also introduced by #3621) and is what prevented
the example from learning **any** task, which in turn hid the performance impact of the KL bug.

## Symptom

World-model losses fell (reco 0.87 -> 0.03, KL grew with fixed code) but eval return stayed flat
at ~-1350 on Pendulum-v1 (random ~-1300, solved ~-150); actor loss ~0. The policy never learned.

## Root cause (proven)

The policy used for **both** the collector and eval was the bare actor head,
`actor(state, belief) -> action` (`build_actor`). In the real env, `state`/`belief` are supplied
by a `TensorDictPrimer` with `default_value=0` and **nothing ever updates them from
observations**. So the agent conditioned on a constant zero latent at every step.

Rollout probe (deterministic, 6 steps) on the *original* actor:

```
t=0 |state|=0.000 |belief|=0.000 |obs|=1.355 action=[-0.0511]
t=1 |state|=0.000 |belief|=0.000 |obs|=1.936 action=[-0.0511]
...
t=5 |state|=0.000 |belief|=0.000 |obs|=4.772 action=[-0.0511]
```

Observation varies; state/belief stay 0; **action is constant**. The agent is a fixed open-loop
controller -- it cannot perceive the pendulum, so it cannot learn to control it. This is
independent of the KL/reco library bugs.

## Fix

Add `build_actor_realworld(...)`: a recurrent acting policy mirroring the classic dreamer's
`_dreamer_make_actor_real` (`sota-implementations/dreamer/dreamer_utils.py:1166`). Per step:

1. `encoder`   : observation -> encoded_latents
2. `posterior` : (belief, encoded_latents) -> state
3. `actor`     : (state, belief) -> action   (the shared `actor_model` head)
4. `prior`     : (state, belief, action) -> `("next", "belief")`  (advance belief for next step)

The env carries `("next","belief")` to the next root `belief` (registered by the primer), giving
the RSSM recurrence. All modules are shared trained instances -- no new parameters. Wiring:
`build_world_model` also returns `(encoder_net, posterior_net)`; `build_actor` also returns
`actor_net`; the collector and eval use `actor_realworld`; imagination keeps the bare
`actor_model` head (correct -- the world model supplies latents there).

Post-fix rollout probe:

```
t=0 |state|=5.657 |next_belief|=0.743 |obs|=1.355 action=[ 0.2396]
t=1 |state|=5.657 |next_belief|=1.127 |obs|=1.861 action=[-0.2594]
...
t=5 |state|=5.657 |next_belief|=1.787 |obs|=4.649 action=[ 0.5686]
```

state is a real categorical latent (|state| = sqrt(32) one-hots), belief accumulates, and the
action now varies with the observation.

## Result

Pendulum now learns: eval return climbs from ~-1350 to ~-90 (near-optimal) by ~16-20k frames.

## JAX parity

Structurally matches the reference acting path -- JAX `agent.py:115` `policy` ->
`rssm.py:75` `_observe`:

```python
deter = self._core(deter, stoch, action)   # advance belief via prior recurrence, using prev action
x = concat([deter, tokens])                 # combine advanced belief with encoded obs
stoch = self._dist(self._logit('obslogit', x)).sample()   # posterior from (advanced belief, obs)
feat = dict(deter=deter, stoch=stoch)       # act on (belief, posterior)
```

The loop is phase-shifted (we advance the belief at the *end* of a step for use at the *next*;
JAX advances at the *start*), but the data dependencies are identical for every step after the
first. One minor difference: at episode reset JAX masks `(deter,stoch,prevact)->0` and runs
`_core(0,0,0)` once to form the initial belief *before* the first posterior, whereas our fix
feeds `belief=0` into the first posterior. A 1-of-200-steps boundary effect; exact parity would
add an `InitTracker` that runs a prior step on episode start.
