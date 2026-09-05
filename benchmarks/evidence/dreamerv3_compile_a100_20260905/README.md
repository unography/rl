# DreamerV3 compiled learner evidence

Status: provisional. These results support continued testing. They are not a
final correctness result or an end-to-end training benchmark.

## Test setup

- Date: 5 September 2026
- GPU: NVIDIA A100-SXM4-80GB
- PyTorch: 2.13.0+cu126
- Precision: bfloat16
- Upstream source: `1d3de3dbd416877c02b2f44fe66a615fe55fb8e9`
- Candidate source: `c1170408cf507d1b2dcb7d9c13ad6280804714d7`
- Batch size: 16
- Sequence length: 64
- Imagination horizon: 15
- Parameters: 640,867
- Measured work: 32 learner updates, including the first update and warmup

The paired runs used the same physical GPU within each run. The benchmark uses
fixed synthetic replay samples. It measures a complete learner update. This
includes the model, actor and value losses, backward pass, optimizer step and
slow-target update. It does not include environment collection or replay I/O.

## Performance

| Run | Eager reference | Compiled candidate | Speedup |
| --- | ---: | ---: | ---: |
| 1, upstream eager | 695.94 ms/update | 15.37 ms/update | 45.28x |
| 2, upstream eager | 838.24 ms/update | 19.80 ms/update | 42.34x |
| 2, candidate eager | 665.23 ms/update | 19.80 ms/update | 33.60x |

The candidate first update took 706.79 seconds in run 1 and 820.74 seconds in
run 2 because it compiled the learner. The run-2 candidate comparison repays
this cost after about 1,270 updates. This estimate applies only to the learner
benchmark.

## Correctness diagnosis

The candidate eager and compiled paths used the same source, GPU and CUDA RNG
state. Their loss error was 0.47%. The final categorical-state mismatch was
0.26%. The final belief RMSE was 0.00332. World-model and value parameter
errors were about 1e-6 on the scaled-RMSE measure.

The strict 32-step check reports `passed: false` for two reasons:

1. Only the return-normalization buffers fail the actor-module limit. The
   eager return span was 0.00000073 and the compiled span was 0.09429. Both
   spans are below the configured minimum scale of 1.0. Both paths therefore
   used the same effective actor scale of 1.0. Actor parameters remained
   close; their largest absolute difference was 0.0000371.
2. Actor momentum differs. The synthetic initial model gives the actor almost
   no learning signal. The eager actor RMS-state magnitude was about 1.02e-14;
   the compiled magnitude was about 1.40e-10. RMS normalization turns these
   very small numerical gradients into unrelated momentum directions. The
   affected optimizer entries, 61 through 73, are the actor parameters.

The other optimizer state is close. World-model momentum had NRMSE 0.0169 and
cosine similarity 0.99986. Value momentum had NRMSE 0.0555 and cosine
similarity 0.99846.

The failed strict gate does not show an optimizer implementation error. It
also must not be changed into a pass by increasing its thresholds. The input
does not provide a useful actor-gradient comparison.

## Required follow-up

- Add a CUDA eager-versus-compiled optimizer test with fixed, nonzero
  gradients.
- Test the complete learner with a nonzero reward and value signal, or with a
  representative trained checkpoint and replay batch.
- Compare losses, gradients, parameters and optimizer state after one or two
  controlled updates.
- Keep the 32-step trajectory comparison as a diagnostic, not a strict
  equality test.
- Run the final candidate for one million environment steps with multiple
  seeds before making a learning-equivalence claim.
- Measure end-to-end throughput before applying the learner speedup to a full
  training-time estimate.

The tensor snapshots used for the diagnosis are stored in the dissertation
artifact repository. They are not committed here because generated
checkpoints do not belong in a TorchRL change. The paired EIDF harness is also
external to this branch. A stable reproducer is required before these numbers
are used in an upstream pull request.
