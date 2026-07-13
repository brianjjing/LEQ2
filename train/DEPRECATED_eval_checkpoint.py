<<<<<<< HEAD
"""Evaluate a saved LEQ2 checkpoint."""

import os
import pickle as pkl
=======
"""Re-evaluate a saved LEQ checkpoint (same setup as train_LEQ.py)."""

import os
import pickle
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
import sys

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
<<<<<<< HEAD
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

LEQ2_TRAIN = os.path.dirname(os.path.abspath(__file__))
LEQ2_ROOT = os.path.dirname(LEQ2_TRAIN)
sys.path.insert(0, LEQ2_ROOT)

import gym
import jax
import numpy as np
=======

LEQ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LEQ_ROOT)

import d4rl
import d4rl_ext
import gym
import jax
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
from absl import app, flags
from ml_collections import config_flags

import wrappers
<<<<<<< HEAD
from dataset_utils import D4RLDataset, NeoRLDataset, split_into_trajectories
=======
from dataset_utils import D4RLDataset, split_into_trajectories
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
from dynamics.termination_fns import get_termination_fn
from dynamics.ensemble_model_learner import get_world_model
from evaluation import evaluate
from algos.leq.learner import Learner

<<<<<<< HEAD
FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "HalfCheetah-v3-medium", "Environment name.")
flags.DEFINE_string("checkpoint", None, "Path to .pkl checkpoint file to evaluate.")
flags.DEFINE_integer("seed", 200, "Random seed (must match the dynamics model seed).")
=======

def normalize(dataset):
    trajs = split_into_trajectories(
        dataset.observations, dataset.actions, dataset.rewards,
        dataset.masks, dataset.dones_float, dataset.next_observations,
    )
    def compute_returns(traj):
        return sum(rew for _, _, rew, _, _, _ in traj)
    trajs.sort(key=compute_returns)
    scale = 1000.0 / (compute_returns(trajs[-1]) - compute_returns(trajs[0]))
    dataset.rewards *= scale
    dataset.returns_to_go *= scale
    return scale, 0.0

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "antmaze-medium-play-v2", "Environment name.")
flags.DEFINE_string("checkpoint", None, "Path to .pkl with actor/critic params.")
flags.DEFINE_integer("seed", 3, "Random seed.")
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
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


<<<<<<< HEAD
def get_normalized_score_neorl(x, env_name):
    scores = {
        "HalfCheetah": (12284, -298),
        "Hopper":      (3294,  5),
        "Walker2d":    (5143,  1),
    }
    max_score, min_score = scores[env_name]
    return (x - min_score) / (max_score - min_score)


def normalize_d4rl(dataset):
    trajs = split_into_trajectories(
        dataset.observations, dataset.actions, dataset.rewards,
        dataset.masks, dataset.dones_float, dataset.next_observations,
    )
    def compute_returns(traj):
        return sum(rew for _, _, rew, _, _, _ in traj)
    trajs.sort(key=compute_returns)
    scale = 1000.0 / (compute_returns(trajs[-1]) - compute_returns(trajs[0]))
    dataset.rewards *= scale
    dataset.returns_to_go *= scale
    return scale, 0.0


def main(_):
    assert FLAGS.checkpoint is not None, "Must pass --checkpoint <path/to/seed_step.pkl>"

    is_neorl = FLAGS.env_name.split("-")[1] == "v3"

    # --- Build env and dataset ---
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

    # --- Load dynamics model ---
    model_path = os.path.join(
        LEQ2_ROOT,
        "../OfflineRL-Kit2/models/dynamics-ensemble/",
        str(FLAGS.seed),
        FLAGS.env_name,
    )
    termination_fn = get_termination_fn(task=FLAGS.env_name)
=======
def make_dataset(env_name, discount):
    env = gym.make(env_name)
    dataset = D4RLDataset(env, discount)
    env_lower = env_name.lower()
    if "antmaze" in env_lower:
        dataset.rewards -= 1.0
        reward_scaler = (1.0, -1.0)
    elif any(x in env_lower for x in ["halfcheetah", "hopper", "walker2d"]):
        if "random" in env_lower:
            reward_scaler = (1.0, 0.0)
        else:
            reward_scale, reward_bias = normalize(dataset)
            reward_scaler = (reward_scale, reward_bias)
    else:
        reward_scaler = (1.0, 0.0)
    return dataset, reward_scaler


def main(_):
    assert FLAGS.checkpoint is not None, "Must pass --checkpoint"

    env = gym.make(FLAGS.env_name)
    obs_dim, action_dim = env.observation_space.shape[-1], env.action_space.shape[-1]
    termination_fn = get_termination_fn(task=FLAGS.env_name)

    model_path = os.path.join(
        LEQ_ROOT,
        "../OfflineRL-Kit/models/dynamics-ensemble/",
        str(FLAGS.seed),
        FLAGS.env_name,
    )
    dataset, reward_scaler = make_dataset(FLAGS.env_name, FLAGS.discount)
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
    with jax.transfer_guard("allow"):
        model, scaler = get_world_model(
            model_path, obs_dim, action_dim, reward_scaler, termination_fn
        )

<<<<<<< HEAD
    # --- Build eval envs ---
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

    # --- Build agent (architecture must match training) ---
    kwargs = dict(FLAGS.config)
    data_batch = dataset.sample(256)
=======
    eval_envs = []
    for i in range(FLAGS.eval_episodes):
        eval_env = gym.make(FLAGS.env_name)
        eval_env = wrappers.EpisodeMonitor(eval_env)
        eval_env = wrappers.SinglePrecision(eval_env)
        seed = FLAGS.seed + i
        eval_env.seed(seed)
        eval_env.action_space.seed(seed)
        eval_env.observation_space.seed(seed)
        eval_envs.append(eval_env)

    data_batch = dataset.sample(256)
    kwargs = dict(FLAGS.config)
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
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
<<<<<<< HEAD
        hidden_dims=tuple([FLAGS.layer_size] * FLAGS.num_layers),
=======
        hidden_dims=tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)]),
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
        discount=FLAGS.discount,
        lamb=FLAGS.lamb,
        num_repeat=FLAGS.num_repeat,
        actor_update=FLAGS.actor_update,
        critic_update=FLAGS.critic_update,
        maintain_model=False,
        **kwargs,
    )

<<<<<<< HEAD
    # --- Load checkpoint ---
    with open(FLAGS.checkpoint, "rb") as f:
        params = pkl.load(f)
    agent.actor = agent.actor.replace(params=params["actor"])
    agent.critic = agent.critic.replace(params=params["critic"])

    # --- Evaluate ---
    stats = evaluate(FLAGS.seed, agent, eval_envs, "", 0, model_eval=None, debug=False)
    print(f"\nenv={FLAGS.env_name}  seed={FLAGS.seed}  episodes={FLAGS.eval_episodes}")
    print(f"mean_return:  {stats['return']:.2f}")
    print(f"mean_length:  {stats['length']:.1f}")


if __name__ == "__main__":
    app.run(main)
=======
    with open(FLAGS.checkpoint, "rb") as f:
        params = pickle.load(f)
    agent.actor = agent.actor.replace(params=params["actor"])
    agent.critic = agent.critic.replace(params=params["critic"])

    stats = evaluate(FLAGS.seed, agent, eval_envs, "", 0, model_eval=None, debug=False)
    print(f"env={FLAGS.env_name} seed={FLAGS.seed} episodes={FLAGS.eval_episodes}")
    print(f"mean_episode_return (D4RL normalized score): {stats['return']}")
    print(f"mean_episode_length: {stats['length']}")


if __name__ == "__main__":
    app.run(main)
>>>>>>> 202636658ad0d1ee76e6014b2de92c0016f893af
