# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compile-friendly DreamerV3 learner coordination.

The tensor path reuses the RSSM and loss calculations from TorchRL. It avoids
TensorDict changes inside the compiled graph.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch.nn import functional as F

from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.modules.models.model_based import (
    _dreamer_v3_rms_norm,
    _DreamerV3RMSNorm,
    DreamerV3MLP,
    RSSMRolloutV3,
    two_hot_cross_entropy,
)
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
    symexp,
    symlog,
)
from torchrl.objectives.dreamer_v3 import _replay_value_target, categorical_kl_terms


class _CoreInputs(NamedTuple):
    observation: torch.Tensor
    action: torch.Tensor
    reset: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    terminated: torch.Tensor
    state: torch.Tensor
    belief: torch.Tensor


class _CoreDraws(NamedTuple):
    rssm_uniforms: torch.Tensor
    action_noise: torch.Tensor
    prior_uniforms: torch.Tensor


class _ModelOutputs(NamedTuple):
    total_model_loss: torch.Tensor
    model_kl: torch.Tensor
    loss_reco: torch.Tensor
    loss_reward: torch.Tensor
    next_states: torch.Tensor
    next_beliefs: torch.Tensor


class _CoreOutputs(NamedTuple):
    total_loss: torch.Tensor
    metrics: torch.Tensor
    post_state: torch.Tensor
    post_belief: torch.Tensor
    return_low: torch.Tensor
    return_high: torch.Tensor


def _normal_log_prob(
    loc: torch.Tensor, scale: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Independent-normal log-density, written as ``torch.distributions`` does."""
    var = scale**2
    log_scale = scale.log()
    log_prob = (
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )
    return log_prob.sum(-1)


def _normal_entropy(scale: torch.Tensor) -> torch.Tensor:
    """Independent-normal entropy, written as ``torch.distributions`` does."""
    return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale)).sum(-1)


def _linear_quantile(values: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute linear quantiles without a data-dependent scalar check."""
    sorted_values = values.sort(dim=0).values
    ranks = q * (values.shape[0] - 1)
    lower = ranks.to(torch.long)
    upper = ranks.ceil().to(torch.long)
    weights = ranks - lower
    shape = (-1, *([1] * (values.ndim - 1)))
    return sorted_values[lower].lerp(sorted_values[upper], weights.reshape(shape))


def _functional_mlp(
    mlp: DreamerV3MLP, parameters: dict[str, torch.Tensor], *inputs: torch.Tensor
) -> torch.Tensor:
    """Run a DreamerV3 MLP with substituted parameters."""
    value = inputs[0] if len(inputs) == 1 else torch.cat(inputs, -1)
    for index, layer in enumerate(mlp.model):
        if isinstance(layer, torch.nn.Linear):
            value = F.linear(
                value,
                parameters[f"model.{index}.weight"],
                parameters.get(f"model.{index}.bias"),
            )
        elif isinstance(layer, _DreamerV3RMSNorm):
            value = _dreamer_v3_rms_norm(
                value, parameters[f"model.{index}.weight"], layer.eps
            )
        else:
            value = layer(value)
    return value


def _detached_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach() for name, parameter in module.named_parameters()}


class TensorLearnerCore:
    """Coordinate one fixed-shape learner update on tensors."""

    def __init__(
        self,
        *,
        world_model: TensorDictModuleBase,
        model_loss: DreamerV3ModelLoss,
        actor_loss: DreamerV3ActorLoss,
        value_loss: DreamerV3ValueLoss,
        cfg: DictConfig,
        state_dim: int,
    ):
        self.state_dim = state_dim
        self.belief_dim = cfg.networks.rnn_hidden_dim

        rollout = world_model[1]
        if not isinstance(rollout, RSSMRolloutV3):
            raise ValueError(
                "The tensor learner needs the standard DreamerV3 RSSM rollout."
            )
        self.rollout = rollout
        self.prior_net = rollout.rssm_prior.module
        self.encoder_net = world_model[0][1].module
        self.decoder_net = world_model[2][0].module
        self.reward_net = world_model[3][0].module
        self.reward_decoder = world_model[3][2].module
        self.continuation_net = world_model[4][0].module

        if (
            model_loss.kl_mode != "separate"
            or not model_loss.reward_two_hot
            or model_loss.reco_loss != "l2"
            or model_loss.global_average
            or model_loss.detach_output
            or model_loss.lambda_continue <= 0
        ):
            raise ValueError(
                "The tensor learner core reproduces the example's model loss "
                "configuration only (separate KL terms, two-hot reward, L2 "
                "reconstruction, per-event sums, continuation loss)."
            )
        self.model_loss = model_loss
        self.reward_bins = model_loss.reward_bins

        self.actor_loss = actor_loss
        self.value_loss = value_loss
        env = actor_loss.model_based_env
        custom_env = (
            type(env) is not DreamerEnv
            or getattr(env, "world_model_params", None) is not None
            or getattr(env, "world_model_buffers", None) is not None
            or getattr(env, "_post_step_mdp_hooks", None) is not None
            or any(
                name in env.__dict__
                for name in (
                    "_reset",
                    "_step",
                    "any_done",
                    "maybe_reset",
                    "reset",
                    "rollout",
                    "step",
                )
            )
        )
        if custom_env:
            raise ValueError("The tensor learner needs the standard DreamerEnv.")
        if value_loss.value_loss != "two_hot":
            raise ValueError("The tensor learner needs the two-hot critic.")
        if not actor_loss.use_reinforce:
            raise ValueError("The tensor learner needs the REINFORCE actor loss.")
        self.actor_net = actor_loss.actor_model[0].module
        self.value_net = value_loss.value_model[0].module
        self.value_decoder = value_loss.value_model[2].module
        self.value_bins = value_loss.value_bins
        self.imagination_horizon = actor_loss.imagination_horizon
        estimator = actor_loss.value_estimator
        self.gamma = estimator.gamma
        self.lmbda = estimator.lmbda
        self.replay_horizon = float(cfg.optimization.continuation_horizon)
        self.replay_lmbda = float(cfg.optimization.lmbda)
        self.replay_value_loss_weight = float(cfg.optimization.replay_value_loss_weight)
        self.action_dim = self.actor_net.action_dim
        self.num_categoricals = self.prior_net.num_categoricals
        self._target_parameters = self._collect_target_parameters(value_loss)

        self._compiled_forward: Callable | None = None

    @staticmethod
    def _collect_target_parameters(
        value_loss: DreamerV3ValueLoss,
    ) -> dict[str, torch.Tensor]:
        if not value_loss.slow_critic_regularization:
            return {}
        target = value_loss.target_value_model_params
        prefix = ("module", "0", "module")
        parameters = {}
        for name, _ in value_loss.value_model[0].module.named_parameters():
            key = prefix + tuple(name.split("."))
            parameters[name] = target.get(key)
        return parameters

    def prepare_inputs(self, sample: TensorDictBase) -> _CoreInputs:
        """Read the replay sample into contiguous tensors."""
        action = sample.get("action")
        reset = sample.get("is_init")
        while reset.ndim > action.ndim and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        while reset.ndim < action.ndim:
            reset = reset.unsqueeze(-1)
        return _CoreInputs(
            observation=sample.get(("next", "observation")),
            action=action,
            reset=reset,
            reward=sample.get(("next", "reward")),
            done=sample.get(("next", "done")),
            terminated=sample.get(("next", "terminated")),
            state=sample.get("state")[..., 0, :].contiguous(),
            belief=sample.get("belief")[..., 0, :].contiguous(),
        )

    def draw(self, inputs: _CoreInputs) -> _CoreDraws:
        """Draw every random input in the order of the reference path."""
        rssm_uniforms = self.rollout._draw_uniforms(inputs.action)
        batch = inputs.action.shape[:-1].numel()
        device = inputs.action.device
        action_noise = []
        prior_uniforms = []
        for _ in range(self.imagination_horizon):
            # The actor draws its reparameterized noise, then the prior draws
            # the categorical uniform, at every imagined step.
            action_noise.append(
                torch.empty((batch, self.action_dim), device=device).normal_()
            )
            prior_uniforms.append(
                torch.rand((batch, self.num_categoricals), device=device)
            )
        return _CoreDraws(
            rssm_uniforms,
            torch.stack(action_noise),
            torch.stack(prior_uniforms),
        )

    def compile(self, **compile_kwargs: Any) -> None:
        """Compile the complete loss-producing forward."""
        compile_kwargs.setdefault("dynamic", False)
        compile_kwargs.setdefault("mode", "reduce-overhead")
        self._compiled_forward = torch.compile(self._forward, **compile_kwargs)

    def _model_forward(
        self,
        inputs: _CoreInputs,
        rssm_uniforms: torch.Tensor,
    ) -> _ModelOutputs:
        model_loss = self.model_loss
        embedding = self.encoder_net(symlog(inputs.observation))
        (
            _,
            _,
            _,
            prior_logits,
            posterior_logits,
            next_states,
            next_beliefs,
        ) = self.rollout._loop(
            inputs.state,
            inputs.belief,
            inputs.action,
            embedding,
            inputs.reset,
            rssm_uniforms,
        )
        reco_pixels = symexp(self.decoder_net(next_states, next_beliefs).float())
        reward_logits = self.reward_net(next_beliefs, next_states).float()
        continue_pred = self.continuation_net(next_beliefs, next_states).float()
        dynamic, representation = categorical_kl_terms(
            posterior_logits,
            prior_logits,
            free_nats=model_loss.free_bits,
            unimix=model_loss.unimix,
        )
        dynamic = model_loss.lambda_kl * model_loss.lambda_dynamic * dynamic
        representation = (
            model_loss.lambda_kl * model_loss.lambda_representation * representation
        )
        model_kl = dynamic + representation
        reconstruction = (symlog(inputs.observation) - symlog(reco_pixels)).square()
        reconstruction = (
            reconstruction.reshape(*inputs.action.shape[:-1], -1).sum(-1).mean()
        )
        reconstruction = model_loss.lambda_reco * reconstruction
        reward = (
            model_loss.lambda_reward
            * two_hot_cross_entropy(
                reward_logits, inputs.reward, self.reward_bins
            ).mean()
        )
        continuation_target = (
            ~inputs.terminated
        ).float() * model_loss.continue_target_scale
        continuation = model_loss.lambda_continue * F.binary_cross_entropy_with_logits(
            continue_pred.squeeze(-1), continuation_target.squeeze(-1)
        )
        total_model_loss = model_kl + reconstruction + reward + continuation
        return _ModelOutputs(
            total_model_loss=total_model_loss,
            model_kl=model_kl,
            loss_reco=reconstruction,
            loss_reward=reward,
            next_states=next_states,
            next_beliefs=next_beliefs,
        )

    def run(
        self,
        inputs: _CoreInputs,
        draws: _CoreDraws,
        return_low: torch.Tensor,
        return_high: torch.Tensor,
    ) -> _CoreOutputs:
        """Run the complete loss-producing forward."""
        if self._compiled_forward is not None:
            outputs = self._compiled_forward(inputs, draws, return_low, return_high)
            # CUDA graphs reuse output buffers on the next call.
            return _CoreOutputs(
                outputs.total_loss,
                outputs.metrics.clone(),
                outputs.post_state.clone(),
                outputs.post_belief.clone(),
                outputs.return_low.clone(),
                outputs.return_high.clone(),
            )
        return self._forward(inputs, draws, return_low, return_high)

    def _forward(
        self,
        inputs: _CoreInputs,
        draws: _CoreDraws,
        return_low: torch.Tensor,
        return_high: torch.Tensor,
    ) -> _CoreOutputs:
        model = self._model_forward(inputs, draws.rssm_uniforms)
        (
            actor_loss,
            value_loss,
            replay_loss,
            new_low,
            new_high,
        ) = self._behavior_forward(inputs, draws, model, return_low, return_high)
        total_loss = (
            model.total_model_loss.squeeze()
            + actor_loss
            + value_loss
            + self.replay_value_loss_weight * replay_loss
        )
        metrics = torch.stack(
            (
                model.model_kl.detach().reshape(()),
                model.loss_reco.detach().reshape(()),
                model.loss_reward.detach().reshape(()),
                actor_loss.detach().reshape(()),
                value_loss.detach().reshape(()),
                replay_loss.detach().reshape(()),
            )
        )
        return _CoreOutputs(
            total_loss=total_loss,
            metrics=metrics,
            post_state=model.next_states.detach(),
            post_belief=model.next_beliefs.detach(),
            return_low=new_low,
            return_high=new_high,
        )

    def _value(
        self, belief: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the online critic's logits and decoded value."""
        logits = self.value_net(belief, state).float()
        return logits, self.value_decoder(logits)

    def _target_value(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        logits = _functional_mlp(
            self.value_net, self._target_parameters, belief, state
        ).float()
        return self.value_decoder(logits)

    def _imagine(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        draws: _CoreDraws,
    ) -> tuple[torch.Tensor, ...]:
        """Roll the actor through the prior for the imagination horizon."""
        prior_net = self.prior_net
        reward_net = self.reward_net
        prior_parameters = _detached_parameters(prior_net)
        reward_parameters = _detached_parameters(reward_net)
        states, beliefs, actions, next_states, next_beliefs, rewards = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for step in range(self.imagination_horizon):
            loc, scale = self.actor_net(state, belief)
            action = loc + draws.action_noise[step] * scale
            _, next_state, next_belief = torch.func.functional_call(
                prior_net,
                prior_parameters,
                (state, belief, action),
                {"_uniform": draws.prior_uniforms[step]},
            )
            reward_logits = torch.func.functional_call(
                reward_net, reward_parameters, (next_belief, next_state)
            )
            reward = self.reward_decoder(reward_logits.float())
            states.append(state)
            beliefs.append(belief)
            actions.append(action)
            next_states.append(next_state)
            next_beliefs.append(next_belief)
            rewards.append(reward)
            state, belief = next_state, next_belief
        return tuple(
            torch.stack(values, -2)
            for values in (states, beliefs, actions, next_states, next_beliefs, rewards)
        )

    def _lambda_target(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        continuation: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.gamma.to(reward)
        lmbda = self.lmbda.to(reward)
        next_return = value[..., -1, :]
        returns = []
        for reward_t, value_t, continuation_t in zip(
            reversed(reward.unbind(-2)),
            reversed(value.unbind(-2)),
            reversed(continuation.unbind(-2)),
        ):
            next_return = reward_t + gamma * continuation_t * (
                (1 - lmbda) * value_t + lmbda * next_return
            )
            returns.append(next_return)
        return torch.stack(returns[::-1], dim=-2)

    def _return_scale(
        self,
        returns: torch.Tensor,
        return_low: torch.Tensor,
        return_high: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_loss = self.actor_loss
        if not actor_loss.return_normalization:
            scale = torch.ones((), dtype=returns.dtype, device=returns.device)
            return scale, return_low, return_high
        retnorm = actor_loss.retnorm
        if actor_loss.training:
            flat = returns.detach().reshape(-1, *retnorm.shape).to(return_low.dtype)
            batch_low, batch_high = _linear_quantile(flat, retnorm._q.to(flat.dtype))
            return_low = return_low.lerp(batch_low, retnorm.rate)
            return_high = return_high.lerp(batch_high, retnorm.rate)
        scale = (return_high - return_low).clamp_min(retnorm.min_scale).squeeze(-1)
        return scale, return_low, return_high

    def _behavior_forward(
        self,
        inputs: _CoreInputs,
        draws: _CoreDraws,
        model: _ModelOutputs,
        return_low: torch.Tensor,
        return_high: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        actor_loss_module = self.actor_loss
        post_state = model.next_states.detach().reshape(-1, self.state_dim)
        post_belief = model.next_beliefs.detach().reshape(-1, self.belief_dim)

        with torch.no_grad():
            (
                states,
                beliefs,
                actions,
                next_states,
                next_beliefs,
                rewards,
            ) = self._imagine(post_state, post_belief, draws)
            _, next_value = self._value(next_beliefs, next_states)
            first_continuation = torch.sigmoid(
                self.continuation_net(beliefs[..., :1, :], states[..., :1, :]).float()
            )
            continuation = torch.sigmoid(
                self.continuation_net(next_beliefs, next_states).float()
            )
            root_continuation = torch.cat(
                [first_continuation, continuation[..., :-1, :]], dim=-2
            )
            lambda_target = self._lambda_target(rewards, next_value, continuation)
            if actor_loss_module.discount_loss:
                gamma = self.gamma.to(lambda_target.device)
                discount = torch.cat(
                    [
                        root_continuation[..., :1, :],
                        gamma * root_continuation[..., 1:, :],
                    ],
                    dim=-2,
                ).cumprod(dim=-2)
            else:
                discount = torch.ones_like(lambda_target)
            discount = discount.detach()

        loc, scale = self.actor_net(states.detach(), beliefs.detach())
        log_prob = _normal_log_prob(loc, scale, actions.detach()).unsqueeze(-1)
        with torch.no_grad():
            _, baseline = self._value(beliefs, states)
        return_scale, new_low, new_high = self._return_scale(
            lambda_target, return_low, return_high
        )
        advantage = (lambda_target - baseline).detach() / return_scale
        actor_loss = -(discount * log_prob * advantage).mean()
        if actor_loss_module.entropy_bonus > 0:
            entropy = _normal_entropy(scale).unsqueeze(-1)
            actor_loss = (
                actor_loss
                - actor_loss_module.entropy_bonus * (discount * entropy).mean()
            )

        lambda_target = lambda_target.detach()
        discount_weight = discount.squeeze(-1)
        value_logits, _ = self._value(beliefs.detach(), states.detach())
        target = lambda_target.squeeze(-1)
        loss = two_hot_cross_entropy(value_logits, target, self.value_bins)
        regularization = self.value_loss.slow_critic_regularization
        if regularization:
            with torch.no_grad():
                slow_target = self._target_value(beliefs.detach(), states.detach())
            loss = loss + regularization * two_hot_cross_entropy(
                value_logits, slow_target.squeeze(-1), self.value_bins
            )
        value_loss = (discount_weight * loss).mean()

        bootstrap = lambda_target[..., 0, 0].reshape(inputs.action.shape[:-1])
        replay_target = _replay_value_target(
            inputs.reward,
            inputs.done,
            inputs.terminated,
            bootstrap,
            self.replay_horizon,
            self.replay_lmbda,
        )
        done = inputs.done.squeeze(-1)
        weight = ~done[..., :-1]
        replay_logits, _ = self._value(model.next_beliefs, model.next_states)
        prediction = replay_logits[..., :-1, :]
        replay_loss = two_hot_cross_entropy(prediction, replay_target, self.value_bins)
        if regularization:
            with torch.no_grad():
                slow_target = self._target_value(model.next_beliefs, model.next_states)[
                    ..., :-1, 0
                ]
            replay_loss = replay_loss + regularization * two_hot_cross_entropy(
                prediction, slow_target, self.value_bins
            )
        replay_loss = (weight.to(replay_loss.dtype) * replay_loss).mean()

        return actor_loss, value_loss, replay_loss, new_low, new_high

    def commit_return_normalization(
        self, return_low: torch.Tensor, return_high: torch.Tensor
    ) -> None:
        """Write the returned normalizer state into the actor-loss buffers."""
        retnorm = self.actor_loss.retnorm
        with torch.no_grad():
            retnorm.low.copy_(return_low)
            retnorm.high.copy_(return_high)
