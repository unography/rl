import numpy as np
import torch

rng = np.random.default_rng(0)
N, H = 7, 15
lam = 0.95
horizon = 333.0

rew = rng.normal(size=(N, H + 1)).astype(np.float32)  # rew at features 0..H
con = rng.uniform(0.9, 0.999, size=(N, H + 1)).astype(np.float32)
val = rng.normal(size=(N, H + 1)).astype(np.float32)


# ---------------- JAX transcription ----------------
def jax_lambda_return(last, term, rew, val, boot, disc, lam):
    rets = [boot[:, -1]]
    live = (1 - last * 0 + 0) * 0  # placeholder
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    intermediate = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(intermediate[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


contdisc = True
disc = 1.0 if contdisc else 1 - 1 / horizon
weight_jax = np.cumprod(disc * con, 1) / disc
last = np.zeros_like(con)
term = 1 - con
ret_jax = jax_lambda_return(last, term, rew, val, val, disc, lam)
adv_jax = ret_jax - val[:, :-1]

# ---------------- torch transcription (candidate) ----------------
gamma = 1.0
reward_t = torch.tensor(rew[:, 1:]).unsqueeze(-1)  # next rewards, features 1..H
value_t = torch.tensor(val[:, 1:]).unsqueeze(-1)  # next values
cont_t = torch.tensor(con[:, 1:]).unsqueeze(-1)  # next continuations
root_cont = torch.tensor(con[:, :-1]).unsqueeze(-1)  # features 0..H-1
root_val = torch.tensor(val[:, :-1]).unsqueeze(-1)

next_return = value_t[..., -1, :]
returns = []
for r, v, c in zip(
    reversed(reward_t.unbind(-2)),
    reversed(value_t.unbind(-2)),
    reversed(cont_t.unbind(-2)),
):
    next_return = r + gamma * c * ((1 - lam) * v + lam * next_return)
    returns.append(next_return)
ret_torch = torch.stack(returns[::-1], dim=-2)

discount_torch = torch.cat(
    [root_cont[..., :1, :], gamma * root_cont[..., 1:, :]], dim=-2
).cumprod(dim=-2)

print(
    "lambda return max abs diff:", np.abs(ret_jax - ret_torch.squeeze(-1).numpy()).max()
)
print(
    "weight max abs diff:",
    np.abs(weight_jax[:, :-1] - discount_torch.squeeze(-1).numpy()).max(),
)
adv_torch = (ret_torch - root_val).squeeze(-1).numpy()
print("adv max abs diff:", np.abs(adv_jax - adv_torch).max())

# ---------------- repl_loss ----------------
T = 20
rew2 = rng.normal(size=(4, T)).astype(np.float32)
boot2 = rng.normal(size=(4, T)).astype(np.float32)
last2 = (rng.uniform(size=(4, T)) < 0.1).astype(np.float32)
term2 = (rng.uniform(size=(4, T)) < 0.05).astype(np.float32)
val2 = rng.normal(size=(4, T)).astype(np.float32)
disc2 = 1 - 1 / horizon
ret2_jax = jax_lambda_return(last2, term2, rew2, val2, boot2, disc2, lam)


def torch_replay_target(reward, done, terminated, bootstrap, horizon, lmbda):
    discount = 1 - 1 / horizon
    live = (~terminated[..., 1:]).to(reward.dtype) * discount
    continuation = (~done[..., 1:]).to(reward.dtype) * lmbda
    intermediate = reward[..., 1:] + (1 - continuation) * live * bootstrap[..., 1:]
    next_return = bootstrap[..., -1]
    returns = []
    for i in reversed(range(intermediate.shape[-1])):
        next_return = (
            intermediate[..., i] + live[..., i] * continuation[..., i] * next_return
        )
        returns.append(next_return)
    return torch.stack(returns[::-1], -1)


ret2_torch = torch_replay_target(
    torch.tensor(rew2),
    torch.tensor(last2).bool(),
    torch.tensor(term2).bool(),
    torch.tensor(boot2),
    horizon,
    lam,
).numpy()
print("repl lambda return max abs diff:", np.abs(ret2_jax - ret2_torch).max())
