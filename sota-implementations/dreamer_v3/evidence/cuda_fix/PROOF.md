# CUDA fix proof

Base revision: `109783ec8ed55690ba1c4eda3aa429366bd2073d`.

## Before

On the NVIDIA A100 test host, the unmodified Walker preset resolved to CPU:

```text
cuda_available=True
cuda_device_name=NVIDIA A100-SXM4-80GB
resolved_walker_env_device=cpu
actor_parameter_devices=['cpu']
collector_observation_device=cpu
collector_action_device=cpu
```

## Fix

- The Walker preset uses `null` as an auto-device sentinel while retaining the
  explicit `env.device=cpu` override.
- `resolve_device()` uses TorchRL's accelerator selection helper.
- Real and model-based environments, learned modules, loss modules, and the
  collector use the same resolved device.
- Replay samples are transferred to the training device before world-model
  updates; newly created latent tensors use that same device.

## After: default Walker preset completes a CUDA update and evaluation

This command keeps `config_dmc_walker`'s device setting untouched. It only
reduces the workload enough to make the proof quick:

```bash
MUJOCO_GL=egl python sota-implementations/dreamer_v3/dreamer_v3.py \
  --config-name=config_dmc_walker \
  collector.frames_per_batch=32 \
  collector.total_frames=32 \
  replay_buffer.buffer_size=256 \
  replay_buffer.batch_size=2 \
  replay_buffer.seq_len=8 \
  replay_buffer.warmup_factor=1 \
  optimization.train_ratio=null \
  optimization.updates_per_batch=1 \
  env.max_episode_steps=32 \
  logger.eval_every=32 \
  logger.eval_episodes=1 \
  logger.output_plot=null \
  logger.metrics_json=null
```

Captured result:

```text
Using training device cuda:0
Initialized LazyTensorStorage with torch.Size([256]) shape
[env_step=   32] eval_reward=+2.05 kl=11.045 reco=1.444 reward=5.541 actor=-0.002
exit_code=0
```

Reaching the final line proves that the default Walker configuration completed
collection, a replay sample, world-model/actor/value optimizer updates, and a
real-environment evaluation without a CPU/CUDA mismatch. The complete concise
capture is in `after.txt`.

## Regression checks

```bash
python -m pytest \
  test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_sota_shares_imagination_parameters \
  test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_dmc_benchmark_aggregation \
  -q
```

Result: `2 passed`.

The complete `test/objectives/test_dreamer_v3.py` file also passes: `58 passed`.
