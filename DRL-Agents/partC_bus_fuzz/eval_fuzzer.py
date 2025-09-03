# partC_bus_fuzz/eval_fuzzer.py (FINAL)
import os, argparse, numpy as np
from stable_baselines3 import SAC, TD3, PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit

from bus_fuzz_env import BusFuzzEnv, BusFuzzConfig

FRAME_RUNS_SUBDIR = "C_bus"
ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO, "a2c": A2C, "ddpg": DDPG}

def make_env(cfg):
    def _f():
        env = BusFuzzEnv(cfg=cfg)
        env = TimeLimit(env, max_episode_steps=cfg.horizon_steps)
        return env
    return _f

def load_vecnorm(venv, models_dir, algo, seed=None):
    tried = []
    if seed is not None:
        tried.append(os.path.join(models_dir, f"{algo}_s{seed}_vecnorm.pkl"))
    tried.append(os.path.join(models_dir, f"{algo}_vecnorm.pkl"))
    for p in tried:
        if os.path.exists(p):
            venv = VecNormalize.load(p, venv)
            venv.training = False
            venv.norm_reward = False
            return venv, p
    return venv, None

def load_model(algo, venv, models_dir, seed=None):
    tried = []
    if seed is not None:
        tried.append(os.path.join(models_dir, f"{algo}_s{seed}.zip"))
    tried.append(os.path.join(models_dir, f"{algo}.zip"))
    for path in tried:
        if os.path.exists(path):
            cls = ALGOS[algo]
            return cls.load(path, env=venv), path
    raise FileNotFoundError("Model not found. Tried: " + " | ".join(tried))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS.keys()), default="sac")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models", FRAME_RUNS_SUBDIR))

    cfg = BusFuzzConfig()
    venv = VecMonitor(DummyVecEnv([make_env(cfg)]))
    venv, stats_path = load_vecnorm(venv, models_dir, args.algo, args.seed)

    model, model_path = load_model(args.algo, venv, models_dir, args.seed)

    returns, lengths = evaluate_policy(model, venv, n_eval_episodes=args.episodes,
                                       deterministic=True, render=False, return_episode_rewards=True)
    print(f"Loaded model: {model_path}")
    if stats_path:
        print(f"Loaded VecNormalize stats: {stats_path}")
    print(f"Episode returns: {np.round(returns, 3)}")
    print(f"Avg return over {args.episodes} eps: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"Avg length: {np.mean(lengths):.1f}")

if __name__ == "__main__":
    main()