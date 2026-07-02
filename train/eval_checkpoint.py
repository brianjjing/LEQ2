"""Re-evaluate a saved LEQ checkpoint (same setup as train_LEQ.py)."""

import os
import pickle
import sys

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

LEQ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LEQ_ROOT)

import d4rl
import d4rl_ext
import gym
import jax
from absl import app, flags
from ml_collections import config_flags

import wrappers
from dataset_utils import D4RLDataset
from dynamics.termination_fns import get_termination_fn
from dynamics.ensemble_model_learner import get_world_model
from evaluation import evaluate
from algos.leq.learner import Learner

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "antmaze-medium-play-v2", "Environment name.")
flags.DEFINE_string("checkpoint", None, "Path to .pkl with actor/critic params.")
flags.DEFINE_integer("seed", 3, "Random seed.")
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


def make_dataset(env_name, discount):
    env = gym.make(env_name)
    dataset = D4RLDataset(env, discount)
    if "antmaze" in env_name.lower():
        dataset.rewards -= 1.0
        reward_scaler = (1.0, -1.0)
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
    with jax.transfer_guard("allow"):
        model, scaler = get_world_model(
            model_path, obs_dim, action_dim, reward_scaler, termination_fn
        )

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
        hidden_dims=tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)]),
        discount=FLAGS.discount,
        lamb=FLAGS.lamb,
        num_repeat=FLAGS.num_repeat,
        actor_update=FLAGS.actor_update,
        critic_update=FLAGS.critic_update,
        maintain_model=False,
        **kwargs,
    )

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
