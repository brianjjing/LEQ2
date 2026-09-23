"""Re-evaluate a saved LEQ checkpoint (same setup as train_LEQ.py)."""

import contextlib
import io
import os
import pickle
import random
import sys

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

LEQ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LEQ_ROOT)
sys.path.append(os.path.join(LEQ_ROOT, "../GORMPO_abiomed/abiomed_env"))  # rl_env (abiomed-v0 only)

import d4rl
import d4rl_ext
import gym
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import wrappers
from dataset_utils import AbiomedDataset, D4RLDataset, split_into_trajectories
from dynamics.termination_fns import get_termination_fn
from dynamics.ensemble_model_learner import get_world_model
from evaluation import evaluate
from algos.leq.learner import Learner


class GymnasiumToGymWrapper(gym.Wrapper):
    """Copied from train_LEQ.py (importing it would re-define its absl flags): Gymnasium
    reset()/step() -> the old-Gym API that wrappers.EpisodeMonitor and evaluate() expect."""

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0]
        return result

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            if "episode" not in info:
                info["episode"] = {}
            return obs, reward, done, info
        return result


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
flags.DEFINE_string(
    "dataset_path",
    os.path.join(LEQ_ROOT, "../GORMPO_abiomed/synthetic_data/SAC_5000eps_stochastic.npz"),
    "abiomed-v0 only: .npz dataset (LEQ_MCS.sh's default); evaluation only uses its shapes.",
)
flags.DEFINE_integer("episode_steps", 6, "abiomed-v0 only: episode length (helpers/evaluate.py's default; train_LEQ.py's final eval uses 1000).")
flags.DEFINE_list("eval_seeds", None, "Comma-separated eval seeds, one evaluate() pass each (default: --seed).")
config_flags.DEFINE_config_file("config", "configs/config.py")


def make_dataset(env_name, discount):
    if "abiomed" in env_name:
        return AbiomedDataset(FLAGS.dataset_path, discount), (1.0, 0.0)
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


def make_env(seed=None, world_model=None):
    """gym.make(env_name); for abiomed-v0, train_LEQ.py's env (world_model: share one across eval envs)."""
    if "abiomed" not in FLAGS.env_name:
        return gym.make(FLAGS.env_name)
    from rl_env import AbiomedRLEnv, AbiomedRLEnvFactory

    kw = dict(
        max_steps=FLAGS.episode_steps,
        action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=True,
        seed=seed,
    )
    if world_model is None:
        env = AbiomedRLEnvFactory.create_env(model_name="10min_1hr_all_data", device="cuda:0", **kw)
    else:
        with contextlib.redirect_stdout(io.StringIO()):  # AbiomedRLEnv prints its action bounds on every init
            env = AbiomedRLEnv(world_model=world_model, **kw)
    return GymnasiumToGymWrapper(env)


def main(_):
    assert FLAGS.checkpoint is not None, "Must pass --checkpoint"

    env = make_env(FLAGS.seed)
    obs_dim, action_dim = env.observation_space.shape[-1], env.action_space.shape[-1]
    termination_fn = get_termination_fn(task=FLAGS.env_name)

    model_path = os.path.join(
        LEQ_ROOT,
        "../OfflineRL-Kit2/models/dynamics-ensemble/",
        str(FLAGS.seed),
        FLAGS.env_name,
    )
    dataset, reward_scaler = make_dataset(FLAGS.env_name, FLAGS.discount)
    with jax.transfer_guard("allow"):
        model, scaler = get_world_model(
            model_path, obs_dim, action_dim, reward_scaler, termination_fn
        )

    world_model = getattr(env.unwrapped, "world_model", None)  # abiomed: all eval envs share one
    eval_envs = []
    for i in range(FLAGS.eval_episodes):
        eval_env = make_env(FLAGS.seed + i, world_model)
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

    returns = []
    for s in [int(x) for x in FLAGS.eval_seeds or [FLAGS.seed]]:
        random.seed(s)  # abiomed draws each episode's start state from this global RNG
        stats = evaluate(s, agent, eval_envs, "", 0, model_eval=None, debug=False)
        returns.append(stats["return"])
        print(f"eval_seed={s} mean_return={stats['return']:.4f} mean_length={stats['length']}")
    print(f"env={FLAGS.env_name} seed={FLAGS.seed} episodes={FLAGS.eval_episodes} eval_seeds={len(returns)}")
    print(f"mean_episode_return (D4RL: normalized score): {np.mean(returns):.4f} +/- {np.std(returns):.4f}")


if __name__ == "__main__":
    app.run(main)
