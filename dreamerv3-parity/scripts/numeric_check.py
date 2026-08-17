import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/rl")
from torchrl.modules.models.model_based_v3 import (
    DreamerV3MLP,
    RSSMPosteriorV3,
    RSSMPriorV3,
)

P = pickle.load(
    open(
        "/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/ckpt/20260808T203817F087503/agent.pkl",
        "rb",
    )
)["params"]
P = {k: np.asarray(v, np.float64) for k, v in P.items() if not k.startswith("opt/")}


def silu(x):
    return x / (1 + np.exp(-x))


def rmsnorm(x, scale, eps=1e-4):
    m2 = np.square(x).mean(-1, keepdims=True)
    return x * (1.0 / np.sqrt(m2 + eps)) * scale


def lin(x, name):
    return x @ P[f"{name}/kernel"] + P[f"{name}/bias"]


def blocklin(x, name, g):
    k = P[f"{name}/kernel"]  # (g, in/g, out/g)
    xs = x.reshape(*x.shape[:-1], g, x.shape[-1] // g)
    y = np.einsum("...ki,kio->...ko", xs, k)
    y = y.reshape(*y.shape[:-2], -1)
    return y + P[f"{name}/bias"]


def flat2group(x, g):
    return x.reshape(*x.shape[:-1], g, x.shape[-1] // g)


def group2flat(x):
    return x.reshape(*x.shape[:-2], -1)


G = 8


def jax_core(deter, stoch, action):
    stoch = stoch.reshape(stoch.shape[0], -1)
    action = action / np.maximum(1, np.abs(action))
    x0 = silu(rmsnorm(lin(deter, "dyn/dynin0"), P["dyn/dynin0norm/scale"]))
    x1 = silu(rmsnorm(lin(stoch, "dyn/dynin1"), P["dyn/dynin1norm/scale"]))
    x2 = silu(rmsnorm(lin(action, "dyn/dynin2"), P["dyn/dynin2norm/scale"]))
    x = np.concatenate([x0, x1, x2], -1)[..., None, :].repeat(G, -2)
    x = group2flat(np.concatenate([flat2group(deter, G), x], -1))
    x = silu(rmsnorm(blocklin(x, "dyn/dynhid0", G), P["dyn/dynhid0norm/scale"]))
    x = blocklin(x, "dyn/dyngru", G)
    gates = np.split(flat2group(x, G), 3, -1)
    reset, cand, update = (group2flat(t) for t in gates)
    reset = 1 / (1 + np.exp(-reset))
    cand = np.tanh(reset * cand)
    update = 1 / (1 + np.exp(-(update - 1)))
    return update * cand + (1 - update) * deter


def jax_prior(deter):
    x = silu(rmsnorm(lin(deter, "dyn/prior0"), P["dyn/prior0norm/scale"]))
    x = silu(rmsnorm(lin(x, "dyn/prior1"), P["dyn/prior1norm/scale"]))
    return lin(x, "dyn/priorlogit").reshape(-1, 32, 4)


def jax_post(deter, tokens):
    x = np.concatenate([deter, tokens], -1)
    x = silu(rmsnorm(lin(x, "dyn/obs0"), P["dyn/obs0norm/scale"]))
    return lin(x, "dyn/obslogit").reshape(-1, 32, 4)


def jax_enc(obs):
    x = np.sign(obs) * np.log1p(np.abs(obs))
    for i in range(3):
        x = silu(rmsnorm(lin(x, f"enc/mlp{i}"), P[f"enc/mlp{i}norm/scale"]))
    return x


def jax_dec(stoch, deter):
    inp = np.concatenate([stoch, deter], -1)
    x = inp
    for i in range(3):
        x = silu(rmsnorm(lin(x, f"dec/mlp/linear{i}"), P[f"dec/mlp/norm{i}/scale"]))
    return np.concatenate(
        [
            lin(x, f"dec/vec/{k}/pred").reshape(x.shape[0], -1)
            for k in ("height", "orientations", "velocity")
        ],
        -1,
    )


def jax_rew(deter, stoch):
    x = np.concatenate([deter, stoch], -1)
    x = silu(rmsnorm(lin(x, "rew/mlp/linear0"), P["rew/mlp/norm0/scale"]))
    return lin(x, "rew/head/logits")


def jax_pol(deter, stoch):
    x = np.concatenate([deter, stoch], -1)
    for i in range(3):
        x = silu(rmsnorm(lin(x, f"pol/mlp/linear{i}"), P[f"pol/mlp/norm{i}/scale"]))
    mean = np.tanh(lin(x, "pol/head/action/mean"))
    std = lin(x, "pol/head/action/stddev")
    std = (1.0 - 0.1) / (1 + np.exp(-(std + 2.0))) + 0.1
    return mean, std


# ---------------- torch side ----------------
td = torch.float64
torch.set_default_dtype(td)
prior = RSSMPriorV3(
    action_shape=torch.Size([6]),
    hidden_dim=64,
    rnn_hidden_dim=512,
    num_categoricals=32,
    num_classes=4,
    action_dim=6,
    unimix=0.01,
    recurrent_model="block_gru",
    num_blocks=8,
    num_layers=1,
    prior_num_layers=2,
    norm_eps=1e-4,
).double()
post = RSSMPosteriorV3(
    hidden_dim=64,
    num_categoricals=32,
    num_classes=4,
    rnn_hidden_dim=512,
    obs_embed_dim=64,
    unimix=0.01,
    use_rms_norm=True,
    num_layers=1,
    norm_eps=1e-4,
).double()
enc = DreamerV3MLP(24, None, depth=3, num_cells=64, norm_eps=1e-4).double()
dec_bb = DreamerV3MLP(640, None, depth=3, num_cells=64, norm_eps=1e-4).double()
dec_heads = torch.nn.ModuleList([torch.nn.Linear(64, n) for n in (1, 14, 9)]).double()
rew = DreamerV3MLP(
    640, 255, depth=1, num_cells=64, outscale=0.0, norm_eps=1e-4
).double()
pol_bb = DreamerV3MLP(640, None, depth=3, num_cells=64, norm_eps=1e-4).double()
pol_mean = torch.nn.Linear(64, 6).double()
pol_std = torch.nn.Linear(64, 6).double()


def setlin(mod, name):
    mod.weight.data.copy_(torch.tensor(P[f"{name}/kernel"].T))
    mod.bias.data.copy_(torch.tensor(P[f"{name}/bias"]))


def setnorm(mod, name):
    mod.weight.data.copy_(torch.tensor(P[f"{name}/scale"]))


def setblock(mod, name):
    mod.weight.data.copy_(torch.tensor(P[f"{name}/kernel"]))
    mod.bias.data.copy_(torch.tensor(P[f"{name}/bias"]))


r = prior.rnn
setlin(r.belief_projection[0], "dyn/dynin0")
setnorm(r.belief_projection[1], "dyn/dynin0norm")
setlin(r.state_projection[0], "dyn/dynin1")
setnorm(r.state_projection[1], "dyn/dynin1norm")
setlin(r.action_projection[0], "dyn/dynin2")
setnorm(r.action_projection[1], "dyn/dynin2norm")
setblock(r.hidden_layers[0], "dyn/dynhid0")
setnorm(r.hidden_layers[1], "dyn/dynhid0norm")
setblock(r.gates, "dyn/dyngru")
pp = prior.rnn_to_prior_projector
setlin(pp[0], "dyn/prior0")
setnorm(pp[1], "dyn/prior0norm")
setlin(pp[3], "dyn/prior1")
setnorm(pp[4], "dyn/prior1norm")
setlin(pp[6], "dyn/priorlogit")
op = post.obs_rnn_to_post_projector
setlin(op[0], "dyn/obs0")
setnorm(op[1], "dyn/obs0norm")
setlin(op[3], "dyn/obslogit")
for i in range(3):
    setlin(enc.model[3 * i], f"enc/mlp{i}")
    setnorm(enc.model[3 * i + 1], f"enc/mlp{i}norm")
    setlin(dec_bb.model[3 * i], f"dec/mlp/linear{i}")
    setnorm(dec_bb.model[3 * i + 1], f"dec/mlp/norm{i}")
    setlin(pol_bb.model[3 * i], f"pol/mlp/linear{i}")
    setnorm(pol_bb.model[3 * i + 1], f"pol/mlp/norm{i}")
for h, k in zip(dec_heads, ("height", "orientations", "velocity")):
    setlin(h, f"dec/vec/{k}/pred")
setlin(rew.model[0], "rew/mlp/linear0")
setnorm(rew.model[1], "rew/mlp/norm0")
setlin(rew.model[3], "rew/head/logits")
setlin(pol_mean, "pol/head/action/mean")
setlin(pol_std, "pol/head/action/stddev")

rng = np.random.default_rng(0)
B = 5
deter = rng.normal(size=(B, 512))
stoch = rng.normal(size=(B, 32, 4))
act = rng.normal(size=(B, 6)) * 2
obs = rng.normal(size=(B, 24)) * 3


def cmp(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    print(f"{name:28s} max|diff|={np.abs(a-b).max():.3e}  scale={np.abs(a).mean():.4f}")


with torch.no_grad():
    t_deter = torch.tensor(deter)
    t_stoch = torch.tensor(stoch.reshape(B, -1))
    t_act = torch.tensor(act)
    cmp(
        "block-GRU core",
        jax_core(deter, stoch, act),
        prior.rnn(t_stoch, t_deter, t_act).numpy(),
    )
    d2 = jax_core(deter, stoch, act)
    cmp(
        "prior logits",
        jax_prior(d2),
        prior.rnn_to_prior_projector(torch.tensor(d2)).reshape(B, 32, 4).numpy(),
    )
    tok = jax_enc(obs)
    cmp("encoder", tok, enc(torch.tensor(np.sign(obs) * np.log1p(np.abs(obs)))).numpy())
    cmp(
        "posterior logits",
        jax_post(d2, tok),
        post.obs_rnn_to_post_projector(
            torch.cat([torch.tensor(d2), torch.tensor(tok)], -1)
        )
        .reshape(B, 32, 4)
        .numpy(),
    )
    t_dec = torch.cat(
        [dec_bb(torch.tensor(stoch.reshape(B, -1)), torch.tensor(deter))], -1
    )
    t_dec = torch.cat([h(t_dec) for h in dec_heads], -1)
    cmp("decoder", jax_dec(stoch.reshape(B, -1), deter), t_dec.numpy())
    cmp(
        "reward logits",
        jax_rew(deter, stoch.reshape(B, -1)),
        rew(torch.tensor(deter), torch.tensor(stoch.reshape(B, -1))).numpy(),
    )
    jm, js = jax_pol(deter, stoch.reshape(B, -1))
    h = pol_bb(torch.tensor(deter), torch.tensor(stoch.reshape(B, -1)))
    cmp("policy mean", jm, pol_mean(h).tanh().numpy())
    cmp("policy std", js, ((1.0 - 0.1) * torch.sigmoid(pol_std(h) + 2) + 0.1).numpy())
