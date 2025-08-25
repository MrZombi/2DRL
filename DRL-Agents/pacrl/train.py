# train.py — one-process training entry (mit Monitor/Eval)
import os, time, json, argparse
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from exploration_wrapper import ExplorationRewardWrapper
from env import Arcade2DEnv
from .callbacks import ExplorationCSVCallback


def make_env(agent_name: str, run_id: int, seed: int,
             render: bool, frame_skip: int, render_every: int):
    def _thunk():
        base = Arcade2DEnv(
            server_url=os.environ.get("ARCADE2D_SERVER", "http://127.0.0.1:8000"),
            seed=seed,
            render=render,
            frame_skip=frame_skip,
            render_every=render_every,
            agent_name=agent_name,
            run_id=run_id,
        )
        return ExplorationRewardWrapper(
            base,
            w_visit=0.3, w_frontier=0.4, w_novel=0.2,
            grid_size=(28, 31), quadrants=(4, 4),
        )
    return _thunk

from pacrl.env import Arcade2DEnv  # unser Gym-Wrapper

def build_model(algo: str, env, net_width: int = 256, lr: float = 3e-4):
    if algo == "dqn":
        policy_kwargs = dict(net_arch=[net_width, net_width])
        return DQN("MlpPolicy", env,
                   learning_rate=lr, buffer_size=200_000, learning_starts=5_000,
                   batch_size=256, gamma=0.99, train_freq=4, target_update_interval=5_000,
                   exploration_fraction=0.20, exploration_final_eps=0.05,
                   policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/DuelingDQN/")
    elif algo == "ppo":
        policy_kwargs = dict(net_arch=dict(pi=[net_width, net_width], vf=[net_width, net_width]))
        return PPO("MlpPolicy", env,
                   n_steps=1024, batch_size=256, n_epochs=10,
                   gae_lambda=0.95, gamma=0.99, learning_rate=lr, clip_range=0.2, ent_coef=0.01,
                   policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/PPO/")
    elif algo == "a2c":
        policy_kwargs = dict(net_arch=dict(pi=[net_width, net_width], vf=[net_width, net_width]))
        return A2C("MlpPolicy", env,
                   n_steps=5, gamma=0.99, learning_rate=7e-4,
                   ent_coef=0.0, vf_coef=0.5, gae_lambda=1.0,
                   use_rms_prop=True, normalize_advantage=True,
                   policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/A2C/")
    else:
        raise ValueError("algo must be one of: dqn, ppo, a2c")

def main():
    import os, time, json, argparse
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.utils import set_random_seed
    from pacrl.env import Arcade2DEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo", "a2c"], required=True)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    # Render/Speed-Flags:
    ap.add_argument("--render", action="store_true", help="show C++ window during training")
    ap.add_argument("--render-every", type=int, default=1, dest="render_every",
                    help="draw every Nth tick on the server")
    ap.add_argument("--frame-skip", type=int, default=1, dest="frame_skip",
                    help="simulate N ticks per /step on the server")
    args = ap.parse_args()

    name_map = {"dqn": "DuelingDQN", "ppo": "PPO", "a2c": "A2C"}
    agent_name = name_map[args.algo]
    run_id = int(time.time())
    set_random_seed(args.seed)

    def make_env(agent_name: str, run_id: int, seed: int):
        def _thunk():
            base = Arcade2DEnv(
                server_url=os.environ.get("ARCADE2D_SERVER", "http://127.0.0.1:8000"),
                seed=seed,
                render=args.render,
                frame_skip=args.frame_skip,
                render_every=args.render_every,
                agent_name=agent_name,
                run_id=run_id,
            )
            # Exploration-Wrapper davor hängen:
            wrapped = ExplorationRewardWrapper(
                base,
                w_visit=0.3, w_frontier=0.4, w_novel=0.2,
                grid_size=(28, 31), quadrants=(4, 4)  # ggf. anpassen
            )
            return wrapped

        return _thunk

    # Trainings-Env
    env = DummyVecEnv([make_env(agent_name, run_id, args.seed)])
    env = VecMonitor(env)

    # Modell bauen (deine build_model-Funktion bleibt unverändert)
    model = build_model(args.algo, env, net_width=args.width, lr=args.lr)

    # Metadaten speichern…
    os.makedirs("models", exist_ok=True)
    meta = {
        "algo": args.algo, "agent_name": agent_name, "run_id": run_id,
        "seed": args.seed, "timesteps": args.timesteps, "net_width": args.width, "lr": args.lr,
        "server_url": os.environ.get("ARCADE2D_SERVER", "http://127.0.0.1:8000"),
        "frame_skip": args.frame_skip, "render": args.render, "render_every": args.render_every,
    }
    with open(f"models/{agent_name}_run{run_id}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Training
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    cb = ExplorationCSVCallback(agent=agent_name, run_id=str(run_id))
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=True)

    # Separates Eval-Env (headless)
    eval_env = Monitor(
        Arcade2DEnv(
            server_url=os.environ.get("ARCADE2D_SERVER", "http://127.0.0.1:8000"),
            seed=args.seed, render=False, frame_skip=1, render_every=1,
            agent_name=f"{agent_name}-eval", run_id=run_id
        )
    )
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    print(f"[{agent_name}] Eval reward: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()

    # Speichern
    model.save(f"models/{agent_name}_run{run_id}.zip")

if __name__ == "__main__":
    main()