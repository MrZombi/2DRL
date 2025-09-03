import os, argparse, json, numpy as np, torch
from dataclasses import asdict

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit

from two_d_env import TwoDEnvHTTP
from callbacks import DiscreteTB

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
FRAME_RUNS_SUBDIR = "A_2d"
POLICY_MLP = "MlpPolicy"

def make_dirs(base_dir):
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models", FRAME_RUNS_SUBDIR))
    runs_dir   = os.path.normpath(os.path.join(base_dir, "..", "runs",   FRAME_RUNS_SUBDIR))
    os.makedirs(models_dir, exist_ok=True); os.makedirs(runs_dir, exist_ok=True)
    return models_dir, runs_dir

def make_env(host, port, horizon, agent_tag, run_id, timeout_s):
    def _f():
        env = TwoDEnvHTTP(host=host, port=port, agent_tag=agent_tag, run_id=run_id, timeout_s=timeout_s)
        if horizon and horizon > 0:
            env = TimeLimit(env, max_episode_steps=horizon)
        return env
    return _f

def train(algo: str, total_steps: int, resume: bool, run_name: str|None, seed: int,
          n_envs: int, device: str, host: str, base_port: int, horizon: int, timeout_s: float):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir, runs_dir = make_dirs(base_dir)

    env_fns = [make_env(host, base_port + i, horizon, agent_tag=algo, run_id=run_name or f"{algo}_s{seed}", timeout_s=timeout_s)
               for i in range(n_envs)]
    venv = VecMonitor(DummyVecEnv(env_fns))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    np.random.seed(seed); torch.manual_seed(seed)

    algo_cls = ALGOS[algo]
    tb_name = run_name or f"{algo}_s{seed}"

    hp = {}
    if algo == "ppo":
        hp = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, clip_range=0.2)
    elif algo == "a2c":
        hp = dict(n_steps=5, learning_rate=3e-4, gamma=0.99)
    elif algo == "dqn":
        hp = dict(
            learning_rate=1e-4, buffer_size=100_000, learning_starts=10_000,
            train_freq=4, batch_size=128, gamma=0.99, target_update_interval=10_000,
            exploration_fraction=0.2, exploration_final_eps=0.05
        )
    else:
        raise ValueError("Unsupported algo")

    model = algo_cls(
        policy=POLICY_MLP, env=venv, tensorboard_log=os.path.join(runs_dir, tb_name),
        verbose=1, device=device, policy_kwargs=dict(net_arch=[128,128]), **hp
    )

    meta = dict(
        algo=algo, seed=seed, n_envs=n_envs, total_steps=total_steps,
        hp=hp, policy_kwargs=dict(net_arch=[128,128]), cfg=dict(host=host, base_port=base_port, horizon=horizon, timeout_s=timeout_s)
    )
    config_path = os.path.join(runs_dir, f"{tb_name}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    tb_cb = DiscreteTB(log_every_steps=1000)

    model.learn(total_timesteps=total_steps, tb_log_name=tb_name,
                progress_bar=True, reset_num_timesteps=not resume,
                log_interval=1, callback=[tb_cb])

    model_path = os.path.join(models_dir, f"{algo}_2denv_s{seed}.zip")
    model.save(model_path); model.save(os.path.join(models_dir, f"{algo}_2denv.zip"))
    venv.save(os.path.join(models_dir, f"{algo}_2denv_s{seed}_vecnorm.pkl"))
    venv.save(os.path.join(models_dir, f"{algo}_2denv_vecnorm.pkl"))

    print(f"[DONE] saved model -> {model_path}")
    print(f"[DONE] logs -> {os.path.join(runs_dir, tb_name)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    ap.add_argument("--total-steps", type=int, default=65536)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    args = ap.parse_args()
    train(args.algo, args.total_steps, args.resume, args.run_name, args.seed,
          args.n_envs, args.device, args.host, args.base_port, args.horizon, args.timeout_s)

if __name__ == "__main__":
    main()