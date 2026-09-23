# Change log: MCS (abiomed-v0) support for the LEQ checkpoint eval (2026-09-21)

Goal: evaluate saved LEQ+MCS checkpoints (no DBG) on abiomed-v0. Nothing was committed or staged.
`evaluation.py`, `train/train_LEQ.py`, `algos/` and the sibling repos were NOT modified.

## Modified
`train/eval_checkpoint.py` - your merge-conflict-resolved LEQ eval (pre-patch copy: `tmp/eval/eval_checkpoint.before_mcs.py`).
It still calls `evaluation.evaluate()` unchanged; MCS support is added on top:

| # | Change | Why |
|---|---|---|
| 1 | imports: `contextlib`, `io`, `random`, `numpy`, `AbiomedDataset` | needed by the additions below |
| 2 | `sys.path.append(.../GORMPO_abiomed/abiomed_env)` | to import `rl_env`, the same env train_LEQ.py uses |
| 3 | `GymnasiumToGymWrapper` copied from train_LEQ.py | `evaluate()` / `EpisodeMonitor` need the old-Gym API; importing train_LEQ.py would re-define its absl flags |
| 4 | flags `--dataset_path` (default: LEQ_MCS.sh's npz), `--episode_steps` (6), `--eval_seeds` (comma list, default `[--seed]`) | dataset only supplies shapes to `Learner`; 6 = `helpers/evaluate.py` default (train_LEQ.py's final eval uses 1000) |
| 5 | `make_dataset()`: abiomed branch -> `AbiomedDataset`, reward scaler `(1.0, 0.0)` | same as train_LEQ.py |
| 6 | new `make_env()`, used for the dims env and every eval env | D4RL unchanged (`gym.make`). abiomed: one `AbiomedRLEnvFactory` env, all eval envs share its world model, so `--eval_episodes 1000` = 1000 light envs, not 1000 GPU models |
| 7 | dynamics path `../OfflineRL-Kit/` -> `../OfflineRL-Kit2/` | Kit only has abiomed seeds 1-3; the checkpoints' dynamics (42/123/456) live in Kit2, which every other LEQ2 script uses |
| 8 | final block: loop over eval seeds (`random.seed(s)` -> `evaluate()`), print per-seed + mean +/- std | `random` drives abiomed's episode start states (the world-model step is deterministic) |

## Created
- `train/eval_checkpoint_mcs.py` - standalone evaluator following GORMPO_abiomed `helpers/evaluate.py` (10 eval seeds x 1000 episodes, plus ACP / unsafe hours / weaning metrics). Superseded by the patch above, but it is the only one that returns those extra metrics. Delete if not wanted.
- `tmp/eval/*.csv` - its results (training seeds 42 / 123 / 456 at 200k). `tmp/eval/eval_checkpoint.before_mcs.py` - backup of your file before the patch.
- `CHANGELOG.md` - this file.

## Replaced outside my edits
- `train/DEPRECATED_eval_checkpoint.py`: I rewrote it early in the session; it was then renamed to `train/eval_checkpoint.py` with the conflict resolved to the incoming (D4RL-only) side (git shows the old path as deleted). My rewrite lives on as `train/eval_checkpoint_mcs.py`.

## Verification (seed-42 checkpoint unless noted)
- Standalone loader reproduces the actions logged during training to <3e-6 on 20 non-saturated (seed, step) pairs (seeds 123/456). Seed 42's policy outputs a constant +1.0 at all ten checkpoints.
- Patched eval, 10 eval seeds x 1000 episodes: return 1.2932 +/- 0.1139, vs 1.2893 +/- 0.1141 from the standalone script.
- D4RL regression: hopper-medium-v2, seed 42, expectile 0.1, 1M ckpt -> 103.85 normalized (5 episodes); the original path still works.

## Run
Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` on a shared GPU, otherwise JAX grabs 75% of it.
```
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n LEQ2 python train/eval_checkpoint.py \
  --env_name abiomed-v0 --seed 42 --checkpoint tmp/EP/models/abiomed-v0/42/0.5/42_200000.pkl \
  --eval_episodes 1000 --eval_seeds=1,4,9,11,14,15,16,17,18,19
```
`--seed` is the training seed (selects the dynamics dir); `--eval_seeds` are the evaluation seeds; add `--episode_steps 1000` for train_LEQ's long-episode eval.

## Diff of `train/eval_checkpoint.py`
```diff
--- before_mcs
+++ after_mcs
@@ -1,7 +1,10 @@
 """Re-evaluate a saved LEQ checkpoint (same setup as train_LEQ.py)."""
 
+import contextlib
+import io
 import os
 import pickle
+import random
 import sys
 
 os.environ["XLA_FLAGS"] = (
@@ -10,22 +13,45 @@
 
 LEQ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 sys.path.insert(0, LEQ_ROOT)
+sys.path.append(os.path.join(LEQ_ROOT, "../GORMPO_abiomed/abiomed_env"))  # rl_env (abiomed-v0 only)
 
 import d4rl
 import d4rl_ext
 import gym
 import jax
+import numpy as np
 from absl import app, flags
 from ml_collections import config_flags
 
 import wrappers
-from dataset_utils import D4RLDataset, split_into_trajectories
+from dataset_utils import AbiomedDataset, D4RLDataset, split_into_trajectories
 from dynamics.termination_fns import get_termination_fn
 from dynamics.ensemble_model_learner import get_world_model
 from evaluation import evaluate
 from algos.leq.learner import Learner
 
 
+class GymnasiumToGymWrapper(gym.Wrapper):
+    """Copied from train_LEQ.py (importing it would re-define its absl flags): Gymnasium
+    reset()/step() -> the old-Gym API that wrappers.EpisodeMonitor and evaluate() expect."""
+
+    def reset(self, **kwargs):
+        result = self.env.reset(**kwargs)
+        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
+            return result[0]
+        return result
+
+    def step(self, action):
+        result = self.env.step(action)
+        if len(result) == 5:
+            obs, reward, terminated, truncated, info = result
+            done = bool(terminated or truncated)
+            if "episode" not in info:
+                info["episode"] = {}
+            return obs, reward, done, info
+        return result
+
+
 def normalize(dataset):
     trajs = split_into_trajectories(
         dataset.observations, dataset.actions, dataset.rewards,
@@ -54,10 +80,19 @@
 flags.DEFINE_integer("num_repeat", 1, "Num repeat.")
 flags.DEFINE_string("actor_update", "lambda-return", "Actor update type.")
 flags.DEFINE_string("critic_update", "lambda-return", "Critic update type.")
+flags.DEFINE_string(
+    "dataset_path",
+    os.path.join(LEQ_ROOT, "../GORMPO_abiomed/synthetic_data/SAC_5000eps_stochastic.npz"),
+    "abiomed-v0 only: .npz dataset (LEQ_MCS.sh's default); evaluation only uses its shapes.",
+)
+flags.DEFINE_integer("episode_steps", 6, "abiomed-v0 only: episode length (helpers/evaluate.py's default; train_LEQ.py's final eval uses 1000).")
+flags.DEFINE_list("eval_seeds", None, "Comma-separated eval seeds, one evaluate() pass each (default: --seed).")
 config_flags.DEFINE_config_file("config", "configs/config.py")
 
 
 def make_dataset(env_name, discount):
+    if "abiomed" in env_name:
+        return AbiomedDataset(FLAGS.dataset_path, discount), (1.0, 0.0)
     env = gym.make(env_name)
     dataset = D4RLDataset(env, discount)
     env_lower = env_name.lower()
@@ -75,16 +110,37 @@
     return dataset, reward_scaler
 
 
+def make_env(seed=None, world_model=None):
+    """gym.make(env_name); for abiomed-v0, train_LEQ.py's env (world_model: share one across eval envs)."""
+    if "abiomed" not in FLAGS.env_name:
+        return gym.make(FLAGS.env_name)
+    from rl_env import AbiomedRLEnv, AbiomedRLEnvFactory
+
+    kw = dict(
+        max_steps=FLAGS.episode_steps,
+        action_space_type="continuous",
+        reward_type="smooth",
+        normalize_rewards=True,
+        seed=seed,
+    )
+    if world_model is None:
+        env = AbiomedRLEnvFactory.create_env(model_name="10min_1hr_all_data", device="cuda:0", **kw)
+    else:
+        with contextlib.redirect_stdout(io.StringIO()):  # AbiomedRLEnv prints its action bounds on every init
+            env = AbiomedRLEnv(world_model=world_model, **kw)
+    return GymnasiumToGymWrapper(env)
+
+
 def main(_):
     assert FLAGS.checkpoint is not None, "Must pass --checkpoint"
 
-    env = gym.make(FLAGS.env_name)
+    env = make_env(FLAGS.seed)
     obs_dim, action_dim = env.observation_space.shape[-1], env.action_space.shape[-1]
     termination_fn = get_termination_fn(task=FLAGS.env_name)
 
     model_path = os.path.join(
         LEQ_ROOT,
-        "../OfflineRL-Kit/models/dynamics-ensemble/",
+        "../OfflineRL-Kit2/models/dynamics-ensemble/",
         str(FLAGS.seed),
         FLAGS.env_name,
     )
@@ -94,9 +150,10 @@
             model_path, obs_dim, action_dim, reward_scaler, termination_fn
         )
 
+    world_model = getattr(env.unwrapped, "world_model", None)  # abiomed: all eval envs share one
     eval_envs = []
     for i in range(FLAGS.eval_episodes):
-        eval_env = gym.make(FLAGS.env_name)
+        eval_env = make_env(FLAGS.seed + i, world_model)
         eval_env = wrappers.EpisodeMonitor(eval_env)
         eval_env = wrappers.SinglePrecision(eval_env)
         seed = FLAGS.seed + i
@@ -133,10 +190,14 @@
     agent.actor = agent.actor.replace(params=params["actor"])
     agent.critic = agent.critic.replace(params=params["critic"])
 
-    stats = evaluate(FLAGS.seed, agent, eval_envs, "", 0, model_eval=None, debug=False)
-    print(f"env={FLAGS.env_name} seed={FLAGS.seed} episodes={FLAGS.eval_episodes}")
-    print(f"mean_episode_return (D4RL normalized score): {stats['return']}")
-    print(f"mean_episode_length: {stats['length']}")
+    returns = []
+    for s in [int(x) for x in FLAGS.eval_seeds or [FLAGS.seed]]:
+        random.seed(s)  # abiomed draws each episode's start state from this global RNG
+        stats = evaluate(s, agent, eval_envs, "", 0, model_eval=None, debug=False)
+        returns.append(stats["return"])
+        print(f"eval_seed={s} mean_return={stats['return']:.4f} mean_length={stats['length']}")
+    print(f"env={FLAGS.env_name} seed={FLAGS.seed} episodes={FLAGS.eval_episodes} eval_seeds={len(returns)}")
+    print(f"mean_episode_return (D4RL: normalized score): {np.mean(returns):.4f} +/- {np.std(returns):.4f}")
 
 
 if __name__ == "__main__":
```

# Change log: LEQ no-DBG MCS expectile runs 0.1 / 0.3 / 0.4 (seed 42), 2026-09-21 evening

No repo code changed. Everything lives on `/data` (root disk is 96% full); logs are symlinked into `tmp/EP_expectile_search/`.

| What | Detail |
|---|---|
| Runs | `train/train_LEQ.py --env_name abiomed-v0 --seed 42 --expectile {0.1,0.3,0.4} --dataset_path .../SAC_5000eps_stochastic.npz --save_dir /data/leq2_runs/mcs_nodbg_expectile_seed42/ --debug` (same flags as `bash_scr/mult_seed/LEQ_MCS.sh`), with `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| GPUs | one per card: 0.1 -> GPU 2, 0.3 -> GPU 7, 0.4 -> GPU 4 (most free memory, ~2.8 GB per run). The first launch had all three on GPU 2 (1.8 GB headroom); it was killed at ~step 23k and relaunched. Its logs: `logs/killed_first_launch/` |
| Logs | `/data/leq2_runs/mcs_nodbg_expectile_seed42/logs/`, symlinked into `tmp/EP_expectile_search/` |
| Cap at 200k | `stop_at_step.sh` (in the run dir) terminates each run's session once `42_200000.pkl` is completely written. Not `--max_steps 200000`: that also feeds the actor's cosine LR schedule (`algos/leq/learner.py:336`), so the checkpoint would differ from a 1M-step run's 200k (e.g. the 0.5 baseline's). Side effect: the end-of-training "Final score" eval never runs. Log: `logs/stop_at_200k.log` |
| Tested | dummy-run test: partial checkpoint ignored, complete checkpoint stops only that session, stale PID refused |

# Change log: LEQ + DBG on real MCS data, expectile 0.4, seed 42, all 5 guardians (2026-09-23)

## Code change: `algos/leq/learner.py` `rollout()`
Guardian scoring now runs in 8192-row chunks, not one `score_samples` call on all ~250k rollout rows. This is the same fix as OfflineRL-Kit2's `_ChunkedScorer` (COMBO, `--guardian-chunk-size`).
- **Why:** the one-shot call OOM'd neuralode (12.3 GiB allocated) in the Sep 17 expectile search. realnvp's `score_samples` also has no `no_grad`, so it held an autograd graph over all rows.
- **Check** (seed-42 guardians, rows from `real_train_val.npz`, chunked vs one-shot):

| guardian | result |
|---|---|
| kde | identical |
| realnvp | differs by at most 2e-2 (float noise) |
| vae, ddpm | stochastic scorers; for vae, chunked-vs-one-shot mean abs error was 1.06 vs its own run-to-run 1.03 |
| neuralode | repeat calls on the same batch are identical; dopri5 picks step sizes per batch, so chunked vs one-shot differs by up to 2.7 on a logp of about 80 |

## Runs
- **Flags:** the same as `bash_scr/expectile_search/LEQ_EXPECTILE_MCS.sh` (`real_train_val.npz`, `dynamics-ensemble-real/42/abiomed-v0`, its per-guardian coef table, `--eval_episodes 10 --debug`), with `--expectile 0.4 --seed 42`, plus `XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1`.
- **Why not the script:** its fixed save dir, `tmp/EP_dbg_expectile_search/<g>/`, already holds the Sep 17-18 kde/vae/realnvp `42/0.4` checkpoints and logs, and a rerun would overwrite them.

| guardian | coef | GPU |
|---|---|---|
| neuralode | 0.2 | 5 |
| kde | 0.2 | 3 |
| realnvp | 0.2 | 1 |
| ddpm | 0.4 | 4 |
| vae | 0.1 | 7 |

- **Outputs:** `/data/leq2_runs/mcs_dbg_real_e0.4_seed42/<guardian>/{models,videos}`. Logs are in `.../logs/`, symlinked into `tmp/EP_dbg_mcs_real_e0.4/`.
- **Stop at 200k:** `stop_at_step.sh`, copied from the no-DBG run dir, runs once per job. It keeps the 1M-step LR schedule, so the 200k checkpoints are comparable to the no-DBG e0.4 run. The stopper logs are `logs/stop_at_200k_<guardian>.log`.

## Outcome
- **Finished:** realnvp, vae, kde and neuralode reached 200k (at 10:27, 10:45, 10:55 and 14:12 UTC). Their `42_200000.pkl` files are byte-identical (md5 `0e14301f…`), to each other and to the Sep 17-18 kde/realnvp/vae `42/0.4` checkpoints. Eval return is identical at every step: 50k −12.43, 100k −23.72, 150k −46.93, 200k −42.85.
- **Cause:** the guardian penalty never affects training. `Learner.rollout()` only penalizes the rewards stored in `rollout_dataset`. `lambda_update_q` (`algos/leq/critic.py`) and `DPG_lambda_update_actor` (`algos/leq/actor.py`) read only `model_batch.observations` and take their rewards from the dynamics model. Every LEQ+DBG run so far is effectively plain LEQ.
- **ddpm:** killed at about step 42k (17:00 UTC), because it can't produce a different result. Its only checkpoint is `42_0.pkl`.

# Change log: guardian penalty moved into LEQ's imagined rollouts (2026-09-23)

## Code change
The penalty now goes where LEQ's Algorithm 1 samples the rewards it trains on: line 16, the imagined rollouts. The old placement was line 11, the dataset expansion; line 9 keeps only the states from that rollout.
- **`algos/leq/learner.py` `_penalize()`:** inside `_update_jit`, the dynamics model is wrapped so every imagined reward becomes r − coef·clip(tanh(0.1·(thr − log p(s′, a))), 0), the same formula as before. This covers all three `--actor_update`/`--critic_update` modes. The guardian runs on the host through `jax.pure_callback`. Its inputs are `stop_gradient`ed, so under the actor's `jacrev` the penalty is a constant. The actor feels the penalty through the λ-return values and the penalized critic, not through ∂penalty/∂a.
- **`Learner`:** builds the host-side `penalty_fn` from the guardian dict. The dataset-expansion rollout and evaluation still use the raw model.
- **Removed:** the penalty in `Learner.rollout()` (including this morning's 8192-row chunking), and `HashableGuardian` in `train/train_LEQ.py`. The guardian is no longer a jit argument.
- **Unchanged:** the `--guardian_*` flags and every `bash_scr/` script.
- **Test:** `test_guardian_penalty.py` checks that a zero penalty leaves one update bit-identical and a non-zero penalty changes it. It fails on the old code and passes for all three update modes. Run it with `JAX_PLATFORMS=cpu conda run -n LEQ2 python test_guardian_penalty.py` (about 45 s).

## Smoke test
MCS real data, e0.4, seed 42, 1000 steps (`/data/leq2_runs/dbg_fix_smoke/`).

| guardian | coef | it/s | checkpoints at 500 and 1000 | critic `reward_model`, mean over steps 100–1000 |
|---|---|---|---|---|
| none | – | 61–66 | – | +0.041 |
| vae | 0.1 | 17–27 | all distinct | −0.026 |
| kde | 0.2 | ~8.7 | all distinct | −0.031 |
| realnvp | 0.2 | ~5 | all distinct | −0.156 (≈ coef: nearly every imagined step gets w ≈ 1) |
| neuralode | 0.2 | ~0.024 (42 s/step) | stopped at step 5 | – |
| ddpm | 0.4 | ~0.04 (25 s/step) | stopped at step 10 | – |

- **neuralode / ddpm:** at these rates, 200k steps would take about 97 and 58 days. Batching per trajectory and skipping the `jacrev` pass would give at most about 15×. These two need a JAX-side stand-in for the guardian, such as an MLP fit to its scores.

- **Coefs:** the per-guardian coefficients were picked while the penalty had no effect, so they are untuned.
- **Cost:** 30 host round trips per update step (10 imagined steps × critic, actor forward, actor `jacrev`). The `jacrev` pass's scoring is wasted, and batching per trajectory would cut the round trips by 10×; see the `ponytail:` note in `_penalize`.
