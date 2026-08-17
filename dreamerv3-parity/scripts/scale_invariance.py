"""Show the DreamerV3 optimizer chain is invariant to a uniform gradient rescale
(so a loss-scale mismatch between JAX and torch cannot explain an LR difference),
except when the rescale moves a tensor across the AGC clipping threshold."""
import ast

import numpy as np
import torch

SRC = "/home/ubuntu/rl/sota-implementations/dreamer_v3/dreamer_v3.py"
tree = ast.parse(open(SRC).read())
node = next(n for n in tree.body if getattr(n, "name", None) == "_DreamerV3Optimizer")
ns = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
Opt = ns["_DreamerV3Optimizer"]

rng = np.random.default_rng(1)
shape = (64, 64)
p0 = (rng.normal(size=shape) * 0.05).astype(np.float32)  # ||p|| ~ 0.2
gseq = [(rng.normal(size=shape)).astype(np.float32) for _ in range(300)]


def run(scale, warmup_steps=1000, steps=300):
    p = torch.nn.Parameter(torch.from_numpy(p0.copy()))
    opt = Opt(
        [p],
        lr=4e-5,
        agc=0.3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-20,
        warmup_steps=warmup_steps,
    )
    traj = []
    clipped = 0
    for g in gseq[:steps]:
        gt = torch.from_numpy(g * scale)
        pn = p.detach().norm().clamp_min(1e-3)
        if gt.norm() > 0.3 * pn:
            clipped += 1
        p.grad = gt
        prev = p.detach().clone()
        opt.step()
        traj.append((p.detach() - prev).clone())
    return torch.stack(traj), clipped


base, cb = run(1.0)
for s in (1e-4, 1e-2, 1e2, 1e4):
    t, c = run(s)
    rel = (t - base).norm() / base.norm()
    print(
        f"grad scale x{s:>8.0e}: rel diff vs x1 = {rel:.3e}  "
        f"(AGC active {c}/300 steps vs {cb}/300 at x1)"
    )

# tiny gradients -> AGC never binds
gseq = [g * 1e-6 for g in gseq]
base2, cb2 = run(1.0)
for s in (1e-3, 1e3):
    t, c = run(s)
    print(
        f"[small-grad regime] scale x{s:>8.0e}: rel diff = "
        f"{(t - base2).norm() / base2.norm():.3e} (AGC active {c}/300 vs {cb2}/300)"
    )
