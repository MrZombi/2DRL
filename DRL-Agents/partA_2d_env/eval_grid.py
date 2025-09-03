import os, argparse, numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from gymnasium.wrappers import TimeLimit

from two_d_env import TwoDEnvHTTP

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
FRAME_RUNS_SUBDIR = "A_2d"

def make_env(host, port, horizon):
    def _f():
        env = TwoDEnvHTTP(host=host, port=port)
        if horizon and horizon > 0:
            env = TimeLimit(env, max_episode_steps=horizon)
        return env
    return _f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models", FRAME_RUNS_SUBDIR))

    venv = VecMonitor(DummyVecEnv([make_env(args.host, args.port, args.horizon)]))
    stats_try = [os.path.join(models_dir, f"{args.algo}_2denv_s{args.seed}_vecnorm.pkl"),
                 os.path.join(models_dir, f"{args.algo}_2denv_vecnorm.pkl")]
    for p in stats_try:
        if os.path.exists(p):
            venv = VecNormalize.load(p, venv); venv.training=False; venv.norm_reward=False; break

    model_try = [os.path.join(models_dir, f"{args.algo}_2denv_s{args.seed}.zip"),
                 os.path.join(models_dir, f"{args.algo}_2denv.zip")]
    for mp in model_try:
        if os.path.exists(mp):
            model = ALGOS[args.algo].load(mp, env=venv); break
    else:
        raise FileNotFoundError("Model not found in: " + " | ".join(model_try))

    rets, lens = [], []
    for _ in range(args.episodes):
        obs, _ = venv.reset()
        done = False
        ep_r = 0.0; ep_l = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = venv.step(action)
            done = bool(term or trunc)
            ep_r += float(r); ep_l += 1
        rets.append(ep_r); lens.append(ep_l)

    print(f"Avg return over {args.episodes} eps: {np.mean(rets):.2f} ± {np.std(rets):.2f}")
    print(f"Avg length: {np.mean(lens):.1f}")
    print(f"Used model: {mp}")

if __name__ == "__main__":
    main()
