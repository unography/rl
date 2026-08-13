# DreamerV3 DMC Walker preset silently defaults to CPU on a CUDA host

## Summary

The documented DreamerV3 Walker command silently runs its policy and collected
tensors on CPU even when CUDA is available. The Walker preset inherits
`env.device: cpu` from `config.yaml`, and neither the documented command nor
`config_dmc_walker.yaml` overrides it.

## Reproduction

From revision `109783ec8ed55690ba1c4eda3aa429366bd2073d`, in an environment with
the DreamerV3 DMC dependencies installed:

```bash
MUJOCO_GL=egl python \
  sota-implementations/dreamer_v3/evidence/cuda_default/repro.py
```

The probe uses the checked-in Walker configuration, `make_env()`,
`build_actor()`, and the same `Collector(..., device=cfg.env.device)` wiring as
the documented training command. It collects only 32 frames.

## Observed result

```text
torch=2.11.0+cu128
cuda_available=True
cuda_device_count=1
cuda_device_name=NVIDIA A100-SXM4-80GB
resolved_walker_env_device=cpu
actor_parameter_devices=['cpu']
collector_observation_device=cpu
collector_action_device=cpu
```

The complete captured output is in `observed.txt`.

## Expected result

The advertised 1.1-million-step Walker benchmark should select an available
accelerator by default, with an explicit `env.device=cpu` override remaining
available. Its environment tensors, collector policy, replay data, and learned
modules should use one coherent resolved device.

## Source-level cause

1. `config.yaml` defines `env.device: cpu`.
2. `config_dmc_walker.yaml` does not override that value.
3. The README command only selects `config_dmc_walker`.
4. `dreamer_v3.py` passes `cfg.env.device` to the collector and does not move
   the learned modules to another device.

This report is independent of the separate observation-conditioned latent
state issue.
