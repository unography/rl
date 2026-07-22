"""Per-run entry for the KL free-nats ablation on Pendulum.

Runs the sota-implementations/dreamer_v3 example unchanged, except that when the
environment variable ``KL_MODE=buggy`` is set it monkeypatches the library's
``categorical_kl_balanced`` to the *origin/main* behavior (free-nats clamped
per-categorical) BEFORE the example runs. ``KL_MODE=fixed`` (default) leaves the
branch's summed-then-clamped version in place.

``DreamerV3ModelLoss.forward`` calls ``categorical_kl_balanced`` via the module
global, so replacing the attribute on the module is sufficient -- no edit to the
library source, no divergence between the two arms except that one function.

Usage (from repo root):
    KL_MODE=buggy .venv/bin/python dreamerv3-parity-notes/kl_ablation_entry.py <hydra overrides>
"""
from __future__ import annotations

import os
import runpy
import sys

import torch
import torchrl.objectives.dreamer_v3 as D


def _buggy_categorical_kl_balanced(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.8,
    free_bits: float = 1.0,
    unimix: float = 0.0,
    beta_dyn: float | None = None,
    beta_rep: float | None = None,
) -> torch.Tensor:
    """The branch's ``categorical_kl_balanced`` with ONLY the clamp granularity reverted.

    This is a byte-for-byte copy of the fixed branch function except that the two
    ``.sum(-1).clamp_min(...)`` lines become ``.clamp_min(...)`` (the origin/main
    per-categorical behavior). Keeping unimix / two-scale weighting identical in
    both arms means the SOLE difference between fixed and buggy is where the
    free-nats floor is applied -- a clean controlled ablation of the bug.
    """
    posterior = torch.softmax(posterior_logits, dim=-1)
    prior = torch.softmax(prior_logits, dim=-1)

    if unimix:
        num_classes = posterior.shape[-1]
        posterior = (1 - unimix) * posterior + unimix / num_classes
        prior = (1 - unimix) * prior + unimix / num_classes

    eps = 1e-8
    posterior = posterior.clamp(min=eps)
    prior = prior.clamp(min=eps)

    post_sg = posterior.detach()
    kl_term1 = (post_sg * (post_sg.log() - prior.log())).sum(-1)

    prior_sg = prior.detach()
    kl_term2 = (posterior * (posterior.log() - prior_sg.log())).sum(-1)

    # BUG (origin/main): clamp per categorical, then mean over [batch, categoricals].
    # The fixed branch inserts ``.sum(-1)`` before ``.clamp_min`` on these two lines.
    kl_dyn = kl_term1.clamp_min(free_bits).mean()
    kl_rep = kl_term2.clamp_min(free_bits).mean()

    if beta_dyn is not None and beta_rep is not None:
        return beta_dyn * kl_dyn + beta_rep * kl_rep
    return alpha * kl_dyn + (1.0 - alpha) * kl_rep


mode = os.environ.get("KL_MODE", "fixed")
if mode == "buggy":
    D.categorical_kl_balanced = _buggy_categorical_kl_balanced
    sys.stderr.write("[ablation] KL_MODE=buggy -> patched categorical_kl_balanced (per-categorical clamp)\n")
elif mode == "fixed":
    sys.stderr.write("[ablation] KL_MODE=fixed -> branch categorical_kl_balanced (summed clamp)\n")
else:
    raise SystemExit(f"KL_MODE must be 'fixed' or 'buggy', got {mode!r}")

# Hand the remaining CLI args to the example's hydra main and run it as __main__.
_EXAMPLE = "sota-implementations/dreamer_v3/dreamer_v3.py"
sys.argv = [_EXAMPLE, *sys.argv[1:]]
runpy.run_path(_EXAMPLE, run_name="__main__")
