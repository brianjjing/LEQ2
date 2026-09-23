"""Evaluate a saved LEQ2 checkpoint on the MCS (Abiomed) world-model env.

Same flags, protocol and CSV metrics as GORMPO_abiomed/cormpo/helpers/evaluate.py, but the
policy is a LEQ2 checkpoint (the {"actor", "critic"} .pkl that train_LEQ.py --debug saves):

    conda run -n LEQ2 python train/eval_checkpoint_mcs.py --policy_path tmp/EP/models/abiomed-v0/42/0.5/42_200000.pkl --seeds 1 4 9 11 14 15 16 17 18 19 --eval_episodes 1000 --devid 2

D4RL / NeoRL checkpoints: use `python evaluation.py --checkpoint ...` instead.
"""

import argparse
import csv
import datetime
import os
import pickle as pkl
import sys
import time

LEQ2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GORMPO_ABIOMED_DIR = os.environ.get(
    "GORMPO_ABIOMED_DIR", os.path.join(LEQ2_ROOT, "..", "GORMPO_abiomed")
)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--policy_path", type=str, required=True, help="LEQ2 .pkl checkpoint")
parser.add_argument("--devid", type=int, default=7, help="Which GPU device index to use")
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="List of seeds for evaluation")
parser.add_argument("--eval_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=6, help="Episode length (helpers/evaluate.py's default)")
parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
parser.add_argument("--num_layers", type=int, default=3, help="Actor MLP layers (train_LEQ.py default)")
parser.add_argument("--layer_size", type=int, default=256, help="Actor MLP width (train_LEQ.py default)")
parser.add_argument("--out", type=str, default=None, help="Results CSV (default: tmp/eval/<checkpoint>_<time>.csv)")
args = parser.parse_args()

# Must precede anything that initialises CUDA: pins jax and torch to --devid, and stops
# JAX preallocating 75% of a GPU that other jobs are sharing.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devid)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, LEQ2_ROOT)
sys.path.append(GORMPO_ABIOMED_DIR)

import jax
import jax.numpy as jnp
import numpy as np
import torch

import policy
from abiomed_env.cost_func import (
    compute_acp_cost_model,
    unstable_percentage_model,
    weaning_score_model,
    weaning_score_model_gradient,
)
from abiomed_env.rl_env import AbiomedRLEnvFactory
from common import Model, PRNGKey


def load_policy(path, obs_dim, action_dim, hidden_dims):
    """Deterministic act(obs) for a LEQ2 checkpoint. The actor is built exactly as in
    algos/leq/learner.py; its obs scaler is one of its params, so the .pkl carries it."""
    actor_def = policy.NormalTanhPolicy(
        hidden_dims,
        obs_dim,
        action_dim,
        log_std_scale=1e-3,
        log_std_min=-5.0,
        dropout_rate=None,
        state_dependent_std=False,
        tanh_squash_distribution=True,
        use_norm=False,
    )
    actor = Model.create(actor_def, inputs=[PRNGKey(0), jnp.zeros((1, obs_dim))])
    with open(path, "rb") as f:
        actor = actor.replace(params=pkl.load(f)["actor"])
    key = PRNGKey(0)

    def act(obs):
        # temperature 0 -> deterministic (key unused), same as evaluation.py's evaluate()
        _, action = policy.sample_actions(key, actor, obs[None], 0.0)
        return np.clip(jax.device_get(action)[0], -1, 1)

    return act


def evaluate(env, act, episodes):
    """helpers/evaluate.py's _evaluate(), minus its wandb/plot side effects."""
    # While evaluating, turn off reward shaping
    env.gamma1 = env.gamma2 = env.gamma3 = 0

    returns, lengths, acp, wean, wean_thr, unstable = [], [], [], [], [], []
    for ep in range(episodes):
        # first start is pinned to idx=1590, later ones are random -- as in helpers/evaluate.py
        obs, _ = env.reset(idx=1590 if ep == 0 else None)
        ep_states, ep_return, done = [], 0.0, False
        while not done:
            next_obs, reward, terminal, truncated, _ = env.step(act(obs))
            ep_return += reward
            ep_states.append(obs)
            obs = next_obs
            done = terminal or truncated

        states = np.array(ep_states)
        acp.append(compute_acp_cost_model(env.world_model, env.episode_actions, states))
        wean.append(weaning_score_model_gradient(env.world_model, states, env.episode_actions)[0])
        wean_thr.append(weaning_score_model(env.world_model, states, env.episode_actions))
        unstable.append(unstable_percentage_model(env.world_model, states))
        returns.append(ep_return)
        lengths.append(len(ep_states))

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "mean_acp": np.mean(acp),
        "mean_unsafe_hours": np.mean(unstable),
        "mean_wean_score": np.mean(wean),
        "mean_wean_score_thr": np.mean(wean_thr),
    }


def main():
    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        max_steps=args.max_steps,
        action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=True,
        seed=args.seeds[0],
        device="cuda:0" if torch.cuda.is_available() else "cpu",  # CUDA_VISIBLE_DEVICES remaps --devid to 0
    )
    act = load_policy(
        args.policy_path,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        (args.layer_size,) * args.num_layers,
    )

    results = []
    for seed in args.seeds:
        # seeds random/np/torch: same start-state sequence as helpers/evaluate.py's fresh env per seed
        env.seed(seed)
        t0 = time.time()
        res = {**evaluate(env, act, args.eval_episodes), "seed": seed}
        results.append(res)
        print(
            f"seed {seed}: "
            + "  ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "seed")
            + f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    print(f"\n{args.policy_path}: {len(results)} seeds x {args.eval_episodes} episodes, max_steps={args.max_steps}")
    print("mean +/- std across seeds:")
    for k in results[0]:
        if k != "seed":
            v = [r[k] for r in results]
            print(f"  {k:20s} {np.mean(v):10.4f} +/- {np.std(v):.4f}")

    out = args.out or os.path.join(
        LEQ2_ROOT, "tmp", "eval",
        f"{os.path.splitext(os.path.basename(args.policy_path))[0]}_{datetime.datetime.now():%m%d_%H%M%S}.csv",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
