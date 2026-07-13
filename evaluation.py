from typing import Dict, List

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle as pkl
import gym
import numpy as np
import copy
import time
from tqdm import tqdm
from functools import partial

from common import PRNGKey, Model


@jax.jit
def step_imagine(
    key: PRNGKey,
    model_eval: Model,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    states: jnp.ndarray,
):
    is_first = jnp.ones(obs.shape[0])
    next_states = model_eval(key, obs, action, is_first, states)
    return states


@partial(jax.jit, static_argnames=["action_dim", "state_dim"])
def step_imagine_first(
    key: PRNGKey, model_eval: Model, obs: jnp.ndarray, action_dim: int, state_dim: int
):
    batch_size = obs.shape[0]
    action = jnp.zeros((batch_size, action_dim))
    is_first = jnp.zeros(batch_size)
    states = jnp.zeros((batch_size, state_dim))
    next_states = model_eval(key, obs, action, is_first, states)
    return states


def evaluate(
    seed: int,
    agent: nn.Module,
    envs: List[gym.Env],
    video_path: str,
    step: int,
    model_eval=None,
    debug=False,
) -> Dict[str, float]:
    stats = {"return": [], "length": []}
    states, actions, rewards, observations = [], [], [], []

    _observations, dones = [], []
    num_episodes = len(envs)
    s = time.time()
    key = jax.device_put(PRNGKey(seed))
    for env in envs:
        _observations.append(env.reset())
        observations.append([])
        dones.append(False)
        states.append([])
        actions.append([])
        rewards.append([])
    # print(observations)
    _observations = np.array(_observations)

    dones = np.array(dones, dtype=bool)
    for j in tqdm(range(10000)):
        if np.all(dones):
            break
        if model_eval is not None:
            key, rng = jax.random.split(key)
            if j == 0:
                action_dim, state_dim = env.action_space.shape[-1], 32 * 32 + 200
                _states = step_imagine_first(
                    key, model_eval, _observations, action_dim, state_dim
                )
            else:
                _states = step_imagine(
                    key, model_eval, _observations, _actions, jax.device_put(_states)
                )
            _states = jax.device_get(_states)
        else:
            _states = _observations
        _actions = agent.sample_actions(key, _states, temperature=0.0)
        _actions = np.array(_actions)
        for i in range(len(envs)):
            if dones[i]:
                continue
            observations[i].append(np.copy(_observations[i]))
            states[i].append(np.copy(_states[i]))
            actions[i].append(np.copy(_actions[i]))
            obs, reward, done, info = envs[i].step(_actions[i])
            _observations[i] = obs
            rewards[i].append(reward)
            if done:
                dones[i] = True
                stats["return"].append(info["episode"]["return"])
                stats["length"].append(info["episode"]["length"])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    observations = np.concatenate(observations, axis=0)
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    if debug:
        _states, _actions = jax.device_put(states), jax.device_put(actions)
        q_values = agent.critic(_states, _actions)
        q_values = jax.device_get(q_values)
        trajectory = (observations, states, actions, rewards)
        print("Saving to:", video_path, step)
        np.save(os.path.join(video_path, f"q_values_{step}.npz"), q_values)
        with open(os.path.join(video_path, f"traj_{step}.pkl"), "wb") as F:
            pkl.dump(trajectory, F)
    return stats


LEQ2_ROOT = os.path.dirname(os.path.abspath(__file__))

def main(_):
    from absl import flags
    import wrappers
    from dataset_utils import D4RLDataset, NeoRLDataset
    from dynamics.termination_fns import get_termination_fn
    from dynamics.ensemble_model_learner import get_world_model
    from algos.leq.learner import Learner

    FLAGS = flags.FLAGS
    assert FLAGS.checkpoint is not None, "Must pass --checkpoint <path/to/seed_step.pkl>"

    is_neorl = FLAGS.env_name.split("-")[1] == "v3"

    if is_neorl:
        import neorl
        task, version, data_type = FLAGS.env_name.split("-")
        env = neorl.make(task + "-" + version)
        dataset = NeoRLDataset(env, data_type, FLAGS.discount)
        reward_scaler = (1.0, 0.0)
    else:
        import d4rl
        import d4rl_ext
        env = gym.make(FLAGS.env_name)
        dataset = D4RLDataset(env, FLAGS.discount)
        env_lower = FLAGS.env_name.lower()
        if "antmaze" in env_lower:
            dataset.rewards -= 1.0
            reward_scaler = (1.0, -1.0)
        elif any(x in env_lower for x in ["halfcheetah", "hopper", "walker2d"]):
            if "random" in env_lower:
                reward_scaler = (1.0, 0.0)
            else:
                scale, bias = normalize_d4rl(dataset)
                reward_scaler = (scale, bias)
        else:
            reward_scaler = (1.0, 0.0)

    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    model_path = os.path.join(
        LEQ2_ROOT,
        "../OfflineRL-Kit2/models/dynamics-ensemble/",
        str(FLAGS.seed),
        FLAGS.env_name,
    )
    termination_fn = get_termination_fn(task=FLAGS.env_name)
    with jax.transfer_guard("allow"):
        model, scaler = get_world_model(
            model_path, obs_dim, action_dim, reward_scaler, termination_fn
        )

    eval_envs = []
    if is_neorl:
        task, version, _ = FLAGS.env_name.split("-")
        gym_env_name = task + "-" + version
        for i in range(FLAGS.eval_episodes):
            e = gym.make(gym_env_name, exclude_current_positions_from_observation=False)
            e.get_normalized_score = lambda x, n=task: get_normalized_score_neorl(x, n)
            e = wrappers.EpisodeMonitor(e)
            e = wrappers.SinglePrecision(e)
            e.seed(FLAGS.seed + i)
            e.action_space.seed(FLAGS.seed + i)
            e.observation_space.seed(FLAGS.seed + i)
            eval_envs.append(e)
    else:
        for i in range(FLAGS.eval_episodes):
            e = gym.make(FLAGS.env_name)
            e = wrappers.EpisodeMonitor(e)
            e = wrappers.SinglePrecision(e)
            e.seed(FLAGS.seed + i)
            e.action_space.seed(FLAGS.seed + i)
            e.observation_space.seed(FLAGS.seed + i)
            eval_envs.append(e)

    kwargs = dict(FLAGS.config)
    data_batch = dataset.sample(256)
    agent = Learner(
        FLAGS.seed,
        jax.device_put(data_batch.observations),
        jax.device_put(data_batch.actions),
        max_steps=FLAGS.max_steps,
        model=model,
        env_name=FLAGS.env_name,
        scaler=scaler,
        reward_scaler=reward_scaler,
        horizon_length=FLAGS.horizon_length,
        expectile=FLAGS.expectile,
        hidden_dims=tuple([FLAGS.layer_size] * FLAGS.num_layers),
        discount=FLAGS.discount,
        lamb=FLAGS.lamb,
        num_repeat=FLAGS.num_repeat,
        actor_update=FLAGS.actor_update,
        critic_update=FLAGS.critic_update,
        maintain_model=False,
        **kwargs,
    )

    with open(FLAGS.checkpoint, "rb") as f:
        params = pkl.load(f)
    agent.actor = agent.actor.replace(params=params["actor"])
    agent.critic = agent.critic.replace(params=params["critic"])

    stats = evaluate(FLAGS.seed, agent, eval_envs, "", 0, model_eval=None, debug=False)
    print(f"\nenv={FLAGS.env_name}  seed={FLAGS.seed}  episodes={FLAGS.eval_episodes}")
    print(f"mean_return:  {stats['return']:.2f}")
    print(f"mean_length:  {stats['length']:.1f}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, LEQ2_ROOT)

    from absl import app, flags
    from ml_collections import config_flags

    flags.DEFINE_string("env_name", "HalfCheetah-v3-medium", "Environment name.")
    flags.DEFINE_string("checkpoint", None, "Path to .pkl checkpoint file to evaluate.")
    flags.DEFINE_integer("seed", 200, "Random seed (must match the dynamics model seed).")
    flags.DEFINE_float("expectile", 0.5, "Expectile used during training.")
    flags.DEFINE_integer("eval_episodes", 10, "Number of eval episodes.")
    flags.DEFINE_integer("num_layers", 3, "Policy MLP layers.")
    flags.DEFINE_integer("layer_size", 256, "Policy MLP width.")
    flags.DEFINE_float("discount", 0.997, "Discount factor.")
    flags.DEFINE_float("lamb", 0.95, "Lambda.")
    flags.DEFINE_integer("horizon_length", 10, "Horizon length.")
    flags.DEFINE_integer("max_steps", int(1e6), "Used for Learner init only.")
    flags.DEFINE_integer("num_repeat", 1, "Num repeat.")
    flags.DEFINE_string("actor_update", "lambda-return", "Actor update type.")
    flags.DEFINE_string("critic_update", "lambda-return", "Critic update type.")
    config_flags.DEFINE_config_file("config", "configs/config.py")

    app.run(main)