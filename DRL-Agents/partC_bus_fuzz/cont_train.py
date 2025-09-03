# partC_bus_fuzz/cont_train.py (FINAL)
import os, argparse, csv, json
import numpy as np
import gymnasium as gym
import torch

from gymnasium.wrappers import TimeLimit, RescaleAction, TransformObservation
from stable_baselines3 import SAC, TD3, PPO, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_checker import check_env
from dataclasses import asdict

from bus_fuzz_env import BusFuzzEnv, BusFuzzConfig
from callbacks import FuzzTB

TOTAL_STEPS_DEFAULT = 200_000
MAX_EP_STEPS = 25
FRAME_RUNS_SUBDIR = "C_bus"

ALGOS = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "a2c": A2C,
    "ddpg": DDPG,
}

def make_dirs(base_dir):
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models", FRAME_RUNS_SUBDIR))
    runs_dir   = os.path.normpath(os.path.join(base_dir, "..", "runs",   FRAME_RUNS_SUBDIR))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    return models_dir, runs_dir

def make_env(seed, cfg):
    def _f():
        env = BusFuzzEnv(cfg)

        # 1) Beobachtungen auf float32 casten (verhindert Gym-Warnungen)
        if isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.dtype != np.float32:
            env = TransformObservation(env, lambda o: np.asarray(o, dtype=np.float32, copy=False))

        # 2) Actions auf [-1, 1] normalisieren (sauber für alle Algos)
        if isinstance(env.action_space, gym.spaces.Box):
            lo, hi = env.action_space.low, env.action_space.high
            if not (np.allclose(lo, -1.0) and np.allclose(hi, 1.0)):
                env = RescaleAction(env, -1.0, 1.0)

        env = TimeLimit(env, max_episode_steps=cfg.horizon_steps)
        return env
    return _f

def train(algo: str, total_steps: int, resume: bool, run_name: str | None, seed: int,
          n_envs: int = 8, device: str = "auto"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir, runs_dir = make_dirs(base_dir)

    policy_kwargs = dict(net_arch=[128, 128])

    cfg = BusFuzzConfig(horizon_steps=MAX_EP_STEPS, seed=seed)
    check_env(BusFuzzEnv(cfg=cfg))

    venv = VecMonitor(DummyVecEnv([make_env(seed + i, cfg) for i in range(n_envs)]))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    np.random.seed(seed)
    torch.manual_seed(seed)

    tb_name = run_name or f"{algo}_s{seed}"
    algo_cls = ALGOS[algo]

    # per-algo HP
    hp = {}
    if algo == "sac":
        hp = dict(learning_rate=3e-4, gamma=0.99, tau=0.005,
                  buffer_size=500_000, batch_size=256,
                  train_freq=(1, "step"), learning_starts=5_000,
                  ent_coef="auto")
    elif algo == "td3":
        n_actions = venv.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        hp = dict(learning_rate=1e-3, gamma=0.99, tau=0.005,
                  buffer_size=500_000, batch_size=256,
                  train_freq=(1, "step"), learning_starts=5_000,
                  action_noise=action_noise, policy_delay=2,
                  target_noise_clip=0.5)
    elif algo == "ppo":
        hp = dict(n_steps=2048, batch_size=256, n_epochs=10,
                  learning_rate=3e-4, ent_coef=0.01, gamma=0.99, clip_range=0.2)
    elif algo == "a2c":
        hp = dict(learning_rate=3e-4, gamma=0.99)
    elif algo == "ddpg":
        n_actions = venv.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        hp = dict(learning_rate=1e-3, gamma=0.99, tau=0.005,
                  buffer_size=500_000, batch_size=256,
                  train_freq=(1, "step"), learning_starts=5_000,
                  action_noise=action_noise)
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    common_kwargs = dict(
        policy="MlpPolicy",
        env=venv,
        tensorboard_log=os.path.join(runs_dir, tb_name),
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs
    )
    model = algo_cls(**common_kwargs, **hp)

    # dump config
    meta = dict(
        algo=algo, seed=seed, n_envs=n_envs, total_steps=total_steps,
        policy_kwargs=policy_kwargs,
        hp=hp,
        cfg=asdict(cfg)
    )
    config_path = os.path.join(runs_dir, f"{tb_name}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # callbacks
    tb_cb = FuzzTB(log_every_steps=1000)

    # learn
    model.learn(total_timesteps=total_steps, tb_log_name=tb_name,
                progress_bar=True, reset_num_timesteps=not resume,
                log_interval=1, callback=[tb_cb])

    # save model + vecnorm
    model_path = os.path.join(models_dir, f"{algo}_s{seed}.zip")
    model.save(model_path)
    model.save(os.path.join(models_dir, f"{algo}.zip"))

    vecnorm_seed = os.path.join(models_dir, f"{algo}_s{seed}_vecnorm.pkl")
    vecnorm_generic = os.path.join(models_dir, f"{algo}_vecnorm.pkl")
    venv.save(vecnorm_seed)
    venv.save(vecnorm_generic)

    # CSV summary
    csv_path = os.path.join(models_dir, f"{algo}_s{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["algo","seed","total_steps","n_envs","tb_run","config_json","vecnorm"])
        w.writerow([algo, seed, total_steps, n_envs, tb_name, config_path, vecnorm_seed])

    print(f"[DONE] saved model -> {model_path}\n[DONE] CSV -> {csv_path}"
          f"\n[DONE] VecNormalize -> {vecnorm_seed}\nLogs -> {os.path.join(runs_dir, tb_name)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS.keys()), default="sac")
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS_DEFAULT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    train(args.algo, args.total_steps, args.resume, args.run_name, args.seed, args.n_envs, args.device)


if __name__ == "__main__":
    main()