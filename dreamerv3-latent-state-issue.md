## Describe the bug

The DreamerV3 SOTA policy does not use observations when it acts in a real
environment. The collector and the evaluation code give `actor_model` zero
`state` and `belief` tensors. The actor uses only these two tensors. It does not
encode the observation, calculate a posterior state, or update the belief.

The observation changes during an episode, but the latent state and the action
distribution stay constant. Random action samples can still change. This can
hide the fault. During deterministic evaluation, the policy repeats the same
action distribution for all observations.

This fault affects the default Pendulum configuration and the DMC Walker
configuration. It affects CPU and accelerator runs.

## To Reproduce

Install TorchRL from source with the development dependencies. Save this code
as `dreamerv3_latent_repro.py` in the repository root:

```python
from pathlib import Path
import runpy

import torch
from omegaconf import OmegaConf
from torchrl.collectors import Collector
from torchrl.data import Unbounded
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import TensorDictPrimer


root = Path.cwd()
dreamer = runpy.run_path(
    root / "sota-implementations/dreamer_v3/dreamer_v3.py"
)
cfg = OmegaConf.load(root / "sota-implementations/dreamer_v3/config.yaml")
cfg.env.max_episode_steps = 8
torch.manual_seed(cfg.env.seed)

state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
env = TransformedEnv(
    dreamer["make_env"](cfg, cfg.env.seed),
    TensorDictPrimer(
        random=False,
        default_value=0,
        state=Unbounded(state_dim),
        belief=Unbounded(cfg.networks.rnn_hidden_dim),
    ),
)
actor = dreamer["build_actor"](
    cfg=cfg,
    action_dim=env.action_spec.shape[-1],
)
collector = Collector(
    env,
    actor,
    frames_per_batch=32,
    total_frames=32,
    device="cpu",
)

try:
    batch = next(iter(collector))
finally:
    collector.shutdown()


def temporal_range(value):
    value = value.float().reshape(value.shape[0], -1)
    return (value.amax(0) - value.amin(0)).abs().max().item()


print(f'observation_temporal_range={temporal_range(batch["observation"])}')
print(f'state_abs_max={batch["state"].abs().max().item()}')
print(f'belief_abs_max={batch["belief"].abs().max().item()}')
print(f'loc_temporal_range={temporal_range(batch["loc"])}')
print(f'scale_temporal_range={temporal_range(batch["scale"])}')
```

Run the file from the repository root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES='' python dreamerv3_latent_repro.py
```

The observations change, but the latent tensors and the action distribution do
not change:

```text
observation_temporal_range=6.100298881530762
state_abs_max=0.0
belief_abs_max=0.0
loc_temporal_range=0.0
scale_temporal_range=0.0
```

## Expected behavior

The real-world policy must encode each current observation. It must use the
encoded observation and the current belief to calculate a posterior state. The
actor must use this posterior state and belief to select an action. The prior
must write the next belief to `("next", "belief")`.

The posterior state and action distribution must change when the input changes.
The belief must change during an episode and reset to zero at the start of each
episode.

## Screenshots

Not applicable.

## System info

TorchRL was installed from source in a virtual environment.

```text
Python: 3.11.16
TorchRL: 0.14.0+ga46e872e
PyTorch: 2.15.0.dev20260816+cu130
TensorDict: 0.14.0+g8e4daaa
NumPy: 2.4.6
Platform: Linux
```

```python
import sys

import numpy
import torch
import torchrl


print(torchrl.__version__, torch.__version__, numpy.__version__)
print(sys.version, sys.platform)
```

## Additional context

The world-model loss calculates posterior states from replay observations. Thus,
the world model can train on observation data. The actor also trains on latent
states during imagination. The fault is in real-environment collection and
evaluation. These paths call the latent-only actor with constant zero inputs.

The sampled actions can change because the policy samples from `TanhNormal`.
This change does not show that the policy uses the observation. The `loc` and
`scale` values show that the distribution stays constant.

## Reason and Possible fixes

`build_actor()` reads only `state` and `belief`. `TensorDictPrimer` sets these
keys to zero. The collector and evaluation use this actor directly. No module
in the acting path writes the posterior state or `("next", "belief")`.

Build a separate policy for real environments. Share the trained modules. Do
not copy them. Use this sequence:

```text
observation
  -> symlog
  -> encoder
  -> posterior(belief, encoded observation)
  -> actor(state, belief)
  -> prior(state, belief, action)
  -> ("next", "belief")
```

Keep the latent-only actor for imagination. Use the observation-conditioned
policy for both collection and evaluation. Keep `TensorDictPrimer` so that it
resets the state and belief at episode boundaries.

## Checklist

- [x] I have checked that there is no similar issue in the repo (**required**)
- [x] I have read the [documentation](https://github.com/pytorch/rl/tree/main/docs/) (**required**)
- [x] I have provided a minimal working example to reproduce the bug (**required**)
