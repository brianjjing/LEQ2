"""The guardian penalty must reach LEQ's update (Algorithm 1, line 16), not only the
dataset-expansion rollout, whose rewards the update never reads.

Run from the repo root: JAX_PLATFORMS=cpu conda run -n LEQ2 python test_guardian_penalty.py
"""
from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from algos.leq.learner import Learner
from common import Batch, Model

OBS, ACT, N = 3, 2, 8


class ToyWorld(nn.Module):
    @nn.compact
    def __call__(self, key, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        next_obs = observations + 0.1 * jnp.tanh(nn.Dense(OBS)(x))
        reward = next_obs.sum(-1)
        return next_obs, reward, jnp.zeros_like(reward), {}


def params_after_update(log_prob):
    """log_prob None = no guardian; far above thr (0) = weight 0; far below = weight 1."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, OBS)).astype(np.float32)
    act = rng.uniform(-1, 1, size=(N, ACT)).astype(np.float32)
    guardian = None
    if log_prob is not None:
        scorer = SimpleNamespace(score_samples=lambda x: np.full(len(x), log_prob))
        guardian = {"model": scorer, "thr": 0.0}
    agent = Learner(
        0,
        obs,
        act,
        model=Model.create(ToyWorld(), [jax.random.PRNGKey(0), jax.random.PRNGKey(1), obs, act]),
        max_steps=10,
        hidden_dims=(8, 8),
        horizon_length=2,
        num_repeat=1,
        expectile=0.4,
        actor_update="lambda-return",
        critic_update="lambda-return",
        scaler=(np.zeros((1, OBS + ACT), np.float32), np.ones((1, OBS + ACT), np.float32)),
        reward_scaler=(1.0, 0.0),
        guardian=guardian,
        guardian_penalty_coef=0.5,
    )
    batch = Batch(obs, act, np.zeros(N, np.float32), np.ones(N, np.float32), obs, None)
    agent.update(batch, batch, 0.25)
    return jax.tree_util.tree_leaves((agent.critic.params, agent.actor.params))


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


if __name__ == "__main__":
    base = params_after_update(None)
    assert same(base, params_after_update(1e3)), "a zero penalty changed the update"
    assert not same(base, params_after_update(-1e3)), "the penalty never reached the update"
    print("ok")
