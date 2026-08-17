## Describe the bug

The DreamerV3 SOTA implementation uses the CPU for all training by default.
This occurs when an accelerator is available. The real DMC environment must run
on the CPU. However, the world model, actor, value model, losses, and policy can
run on the accelerator.

The training loop also keeps detached loss tensors in memory. It keeps five
scalar tensors for each optimization update. The DMC Walker preset does
1,099,000 updates. Therefore, the loop keeps 5,495,000 tensor objects. If
training uses CUDA, these CUDA tensors stay in memory until training ends.

## To Reproduce

Install TorchRL from source. Install the development and DMC dependencies. Use
a machine that has a CUDA GPU. Start the Walker preset:

```bash
source .venv/bin/activate
MUJOCO_GL=egl python sota-implementations/dreamer_v3/dreamer_v3.py \
  --config-name=config_dmc_walker
```

Open a second terminal. Monitor the active CUDA processes:

```bash
watch -n 0.5 \
  'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv'
```

In the affected implementation, the DreamerV3 process does not create a CUDA
context. The training stays on the CPU.

The training loop adds detached tensors to lists after each optimization
update:

```python
loss_hist["kl"].append(model_kl.detach())
loss_hist["reco"].append(m_td["loss_model_reco"].detach())
loss_hist["reward"].append(m_td["loss_model_reward"].detach())
loss_hist["actor"].append(a_td["loss_actor"].detach())
loss_hist["value"].append(v_td["loss_value"].detach())
```

The `detach()` operation removes a tensor from the autograd graph. It does not
move the tensor to the CPU. It does not release the tensor storage.

## Expected behavior

Run the real DMC environments on the CPU. Keep the collector output and replay
buffer on the CPU. Run the models, losses, policy, replay samples, and
model-based imagination environment on an available accelerator by default.
Let the user select the CPU for training.

Do not keep millions of separate loss tensors. Keep only a limited amount of
loss data on the GPU. Keep the full CPU loss history only when a plot is
required.

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
dm-control: 1.0.44
MuJoCo: 3.11.0
Platform: Linux
GPU: NVIDIA A100 80GB PCIe
```

```python
import sys

import numpy
import torch
import torchrl

print(torchrl.__version__, torch.__version__, numpy.__version__)
print(sys.version, sys.platform)
print(torch.cuda.is_available(), torch.cuda.device_count())
```

## Additional context

The DMC physics simulation uses the CPU. This is normal. Keep the real DMC
environment on the CPU. Use the accelerator for policy inference and training.

Keep the replay storage on the CPU because the Walker preset uses a large
buffer. Move each sampled batch to the training device before the loss update.

## Reason and Possible fixes

The configuration uses the environment device as the collector and training
device. The code does not move the training modules to an automatically
selected accelerator. Use separate settings for the real environment and
training.

Use these changes:

- Add an `optimization.device` setting. Use `null` to select an available
  accelerator.
- Keep the real collection and evaluation environments on the CPU.
- Set separate `policy_device`, `env_device`, and `storing_device` values for
  the collector.
- Move the models, losses, and sampled replay batches to the optimization
  device.
- Set `auto_cast_to_device=True` for evaluation rollouts.
- Create `DreamerEnv` on the optimization device. Its base constructor calls
  `world_model.to(self.device)` on shared model parameters.
- Store the loss metrics in one tensor for each batch. Move the loss history to
  the CPU only when the code must make a plot.

## Checklist

- [ ] I have checked that there is no similar issue in the repo (**required**)
- [ ] I have read the [documentation](https://github.com/pytorch/rl/tree/main/docs/) (**required**)
- [x] I have provided a minimal working example to reproduce the bug (**required**)
