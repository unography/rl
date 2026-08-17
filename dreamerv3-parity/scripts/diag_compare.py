"""Term-by-term comparison of JAX train/* scalars vs Torch diagnostics."""
import json
import statistics
from collections import defaultdict

JAX_PATHS = {
    0: "/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/metrics.jsonl",
    1: "/tmp/jaxrun-seed1/metrics.jsonl",
    2: "/tmp/jaxrun-seed2/metrics.jsonl",
}
TORCH_PATHS = {s: f"/tmp/dv3-seed{s}/metrics.jsonl" for s in (0, 1, 2)}

# jax key -> torch key
PAIRS = [
    ("train/loss/dyn", "loss_dynamic"),
    ("train/loss/rep", "loss_representation"),
    ("train/loss/con", "loss_continue"),
    ("train/loss/rew", "loss_reward"),
    ("train/loss/policy", "loss_actor"),
    ("train/loss/value", "loss_value"),
    ("train/loss/repval", "loss_replay_value"),
    ("train/val", "val"),
    ("train/slowval", "slowval"),
    ("train/ret", "ret"),
    ("train/ret_max", "ret_max"),
    ("train/ret_rate", "ret_rate"),
    ("train/adv", "adv"),
    ("train/adv_mag", "adv_mag"),
    ("train/adv_std", "adv_std"),
    ("train/ent/action", "ent_action"),
    ("train/weight", "weight"),
    ("train/con", "con"),
    ("train/rew", "rew"),
]

BUCKET = 8192


def load(path, stepkey, keys, prefix_ok=lambda r: True):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if stepkey not in r or not prefix_ok(r):
            continue
        b = int(r[stepkey]) // BUCKET
        for k in keys:
            if k in r and isinstance(r[k], (int, float)):
                out[b][k].append(r[k])
    return out


jax = {s: load(p, "step", [k for k, _ in PAIRS]) for s, p in JAX_PATHS.items()}
torch_ = {
    s: load(
        p, "action_steps", [v for _, v in PAIRS], lambda r: r.get("type") == "train"
    )
    for s, p in TORCH_PATHS.items()
}

buckets = sorted(
    set().union(*[set(d) for d in jax.values()])
    & set().union(*[set(d) for d in torch_.values()])
)

print(f"buckets of {BUCKET} steps: {[b*BUCKET for b in buckets]}\n")
hdr = f"{'metric':>22s} | " + " | ".join(
    f"{b*BUCKET//1000:>3d}k jax   torch  ratio" for b in buckets
)
print(hdr)
print("-" * len(hdr))
for jk, tk in PAIRS:
    cells = []
    for b in buckets:
        jv = [v for s in jax for v in jax[s][b].get(jk, [])]
        tv = [v for s in torch_ for v in torch_[s][b].get(tk, [])]
        if not jv or not tv:
            cells.append(f"{'-':>22s}")
            continue
        jm, tm = statistics.mean(jv), statistics.mean(tv)
        ratio = tm / jm if abs(jm) > 1e-9 else float("nan")
        cells.append(f"{jm:8.3f} {tm:8.3f} {ratio:6.2f}")
    print(f"{jk:>22s} | " + " | ".join(cells))
