#!/usr/bin/env python3
"""Builds a real-data MCS offline dataset (.npz) for LEQ's --dataset_path.

Same windowing/reward logic as OfflineRL-Kit2's
run_example/run_combo_dbg_mcs.py::build_abiomed_dataset (COMBO already trains
on this real data successfully) -- copied rather than imported since COMBO
and LEQ2 live in separate conda envs. One deliberate difference: terminals is
ones(), not COMBO's zeros(). Each row here is an independently-sampled real
window, not a step in a chained trajectory (verified: 0/12050 adjacent rows
in the raw pkl are temporally continuous). COMBO's algorithm doesn't care.
LEQ's AbiomedDataset (dataset_utils.py) infers episode boundaries from
row-to-row continuity when terminals==0, which would silently mislabel every
row here -- ones() makes the true 1-step-episode structure explicit instead.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/brian/repos/GORMPO_abiomed/abiomed_env")
from rl_env import AbiomedRLEnvFactory


def build_real_dataset(env, timesteps=6, feat=12):
    wm = env.world_model
    splits = [wm.data_train, wm.data_val]  # data_test stays held out
    data   = torch.cat([torch.as_tensor(s.data)   for s in splits], dim=0).float()
    pl     = torch.cat([torch.as_tensor(s.pl)     for s in splits], dim=0).float()
    labels = torch.cat([torch.as_tensor(s.labels) for s in splits], dim=0).float()

    obs = data.reshape(-1, timesteps * feat)
    next_obs = torch.cat(
        [labels.reshape(-1, timesteps, feat - 1), pl.reshape(-1, timesteps, 1)], dim=2
    ).reshape(-1, timesteps * feat)

    pl_unnorm = np.asarray(wm.unnorm_pl(pl))
    pl_mode = np.array([np.bincount(np.rint(r).astype(int)).argmax() for r in pl_unnorm]).reshape(-1, 1)
    actions = np.asarray(wm.normalize_pl(torch.as_tensor(pl_mode, dtype=torch.float32))).reshape(-1, 1)

    nb = next_obs.reshape(-1, timesteps, feat)
    rewards = np.array([env._compute_reward(nb[i]) for i in range(nb.shape[0])], dtype=np.float32)

    obs_np, next_np = obs.cpu().numpy().astype(np.float32), next_obs.cpu().numpy().astype(np.float32)
    assert obs_np.shape[1] == timesteps * feat and actions.shape[1] == 1
    assert len(obs_np) == len(next_np) == len(actions) == len(rewards)
    return {
        "observations": obs_np,
        "next_observations": next_np,
        "actions": actions.astype(np.float32),
        "rewards": rewards,
        "terminals": np.ones(len(obs_np), dtype=np.float32),
    }


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/brian/repos/GORMPO_abiomed/synthetic_data/real_train_val.npz"

    env = AbiomedRLEnvFactory.create_env(
        model_name="10min_1hr_all_data",
        model_path="/home/brian/repos/OfflineRL-Kit2/abiomed_env/data/10min_1hr_all_data_model.pth",
        data_path="/public/gormpo/10min_1hr_all_data.pkl",  # full 17,865-window data, not the 300-window sample
        max_steps=6, action_space_type="continuous",
        reward_type="smooth", normalize_rewards=True,
        seed=42, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataset = build_real_dataset(env)
    np.savez(out_path, **dataset)
    print(f"saved {len(dataset['observations'])} transitions -> {out_path}")
