# partA_2d_env/fault_campaign.py (FINAL)
from __future__ import annotations

import os, argparse, csv, time
import numpy as np
from typing import Optional

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from gymnasium.wrappers import TimeLimit

from two_d_env import TwoDEnvHTTP
from fault_env import FaultInjectEnv

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
FRAME_RUNS_SUBDIR = "A_2d"

def make_env(host, port, horizon, faults):
    def _f():
        env = TwoDEnvHTTP(host=host, port=port)
        if horizon and horizon > 0:
            env = TimeLimit(env, max_episode_steps=horizon)
        env = FaultInjectEnv(env, **faults)
        return env
    return _f

def load_vecnorm(venv, models_dir, algo, seed=None):
    tried = []
    if seed is not None:
        tried.append(os.path.join(models_dir, f"{algo}_2denv_s{seed}_vecnorm.pkl"))
    tried.append(os.path.join(models_dir, f"{algo}_2denv_vecnorm.pkl"))
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
        tried.append(os.path.join(models_dir, f"{algo}_2denv_s{seed}.zip"))
    tried.append(os.path.join(models_dir, f"{algo}_2denv.zip"))
    for path in tried:
        if os.path.exists(path):
            cls = ALGOS[algo]
            return cls.load(path, env=venv), path
    raise FileNotFoundError("Model not found. Tried: " + " | ".join(tried))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--horizon", type=int, default=0)

    # faults
    ap.add_argument("--obs-noise-std", type=float, default=0.0)
    ap.add_argument("--obs-drop-p", type=float, default=0.0)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--sticky-p", type=float, default=0.0)
    ap.add_argument("--reward-flip-p", type=float, default=0.0)
    ap.add_argument("--reward-offset", type=float, default=0.0)

    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models", FRAME_RUNS_SUBDIR))

    faults = dict(
        obs_drop_p=args.obs_drop_p,
        obs_noise_std=args.obs_noise_std,
        delay_ms=args.delay_ms,
        sticky_p=args.sticky_p,
        reward_flip_p=args.reward_flip_p,
        reward_offset=args.reward_offset,
    )

    # Build env
    venv = VecMonitor(DummyVecEnv([make_env(args.host, args.port, args.horizon, faults)]))
    venv, stats_path = load_vecnorm(venv, models_dir, args.algo, args.seed)

    # Load model
    model, model_path = load_model(args.algo, venv, models_dir, args.seed)

    # Evaluate
    returns, lengths, infer_ms_all = [], [], []
    for ep in range(args.episodes):
        obs, _ = venv.reset()
        done = False
        ep_r = 0.0
        ep_l = 0
        infer_ms = []
        while not done:
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            infer_ms.append((time.perf_counter() - t0) * 1000.0)

            obs, r, term, trunc, info = venv.step(action)
            done = bool(term or trunc)
            ep_r += float(r)
            ep_l += 1
        returns.append(ep_r)
        lengths.append(ep_l)
        infer_ms_all += infer_ms

    p95 = float(np.percentile(infer_ms_all, 95)) if infer_ms_all else float("nan")

    # CSV out
    tag = f"od{args.obs_drop_p}_on{args.obs_noise_std}_dm{args.delay_ms}_sp{args.sticky_p}_rfp{args.reward_flip_p}_ro{args.reward_offset}"
    safe_tag = tag.replace(".", "p")
    csv_name = f"fault_{args.algo}_s{args.seed}_{safe_tag}.csv"
    csv_path = os.path.join(models_dir, csv_name)
    os.makedirs(models_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo","seed","episodes","avg_return","std_return","avg_len","p95_infer_ms",
                    "obs_drop_p","obs_noise_std","delay_ms","sticky_p","reward_flip_p","reward_offset",
                    "model_path","vecnorm_path"])
        w.writerow([args.algo, args.seed, args.episodes, np.mean(returns), np.std(returns),
                    np.mean(lengths), p95,
                    args.obs_drop_p, args.obs_noise_std, args.delay_ms, args.sticky_p, args.reward_flip_p, args.reward_offset,
                    model_path, stats_path or ""])

    print(f"Loaded model: {model_path}")
    if stats_path:
        print(f"Loaded VecNormalize stats: {stats_path}")
    print(f"Avg return over {args.episodes} eps: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"Avg length: {np.mean(lengths):.1f}")
    print(f"Inference p95: {p95:.3f} ms")
    print(f"Wrote CSV -> {csv_path}")

if __name__ == "__main__":
    main()