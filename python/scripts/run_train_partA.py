
#!/usr/bin/env python
from __future__ import annotations
import argparse, os, ast
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from callbacks.episode_csv_logger import EpisodeCSVLogger
from callbacks.tb_telemetry_callback import TBTelemetryCallback
from env.two_d_env import TwoDEnv, EnvConfig

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def parse_net_arch(s: str | None):
    if not s:
        return None
    try:
        # allow simple comma list "64,64"
        if isinstance(s, str) and "," in s and not any(ch in s for ch in "{}[]"):
            return [int(x) for x in s.split(",") if x.strip()]
        v = ast.literal_eval(s)
        return v
    except Exception:
        msg = (
            '--net-arch must be a Python literal like "[64,64]" '
            'or \'{"pi":[64,64],"vf":[64,64]}\', or a comma list like "128,128"'
        )
        raise SystemExit(msg)


def make_env(*, host: str, port: int, seed: int, frame_skip: int,
             render: bool, render_every: int, run_id: str):
    def _thunk():
        cfg = EnvConfig(host=host, port=port, seed=seed, frame_skip=frame_skip,
                        render=render, render_every=render_every, agent="sb3", run_id=run_id)
        env = TwoDEnv(cfg)
        return Monitor(env)
    return _thunk


def build_vec_env(n_envs: int, *, host: str, base_port: int, seed: int, frame_skip: int,
                  render: bool, render_every: int, run_id: str):
    fns = []
    for i in range(n_envs):
        port = base_port + i
        fns.append(make_env(host=host, port=port, seed=seed + i, frame_skip=frame_skip,
                            render=render, render_every=render_every, run_id=run_id))
    return DummyVecEnv(fns)


def main():
    ap = argparse.ArgumentParser()
    # core
    ap.add_argument("--algo", choices=ALGOS.keys(), default="ppo")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-envs", type=int, default=4)

    # server
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-every", type=int, default=1)

    # logging / ckpt
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--tb-dir", default="tb/partA")
    ap.add_argument("--csv-file", default="logs/partA/episodes.csv")
    ap.add_argument("--save-every", type=int, default=0)

    # normalization (support long+short flags)
    ap.add_argument("--normalize-obs", action="store_true")
    ap.add_argument("--normalize-rew", action="store_true")
    ap.add_argument("--norm-obs", action="store_true")
    ap.add_argument("--norm-rew", action="store_true")

    # model hyperparams (support long+short where sensible)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)

    # on-policy specifics (PPO/A2C)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)   # PPO
    ap.add_argument("--n-epochs", type=int, default=None)     # PPO
    ap.add_argument("--net-arch", type=str, default=None)

    # off-policy specifics (DQN)
    ap.add_argument("--buffer-size", type=int, default=None)
    ap.add_argument("--train-freq", type=int, default=None)
    ap.add_argument("--gradient-steps", type=int, default=None)
    ap.add_argument("--learning-starts", type=int, default=None)
    ap.add_argument("--target-update-interval", type=int, default=None)

    args = ap.parse_args()

    run_id = args.run_id or f"partA_{args.algo}_seed{args.seed}"
    logs_base = os.path.dirname(args.csv_file) or "logs/partA"
    os.makedirs(logs_base, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    # env
    venv = build_vec_env(
        n_envs=max(1, args.n_envs),
        host=args.host, base_port=args.base_port, seed=args.seed,
        frame_skip=args.frame_skip, render=args.render, render_every=args.render_every,
        run_id=run_id,
    )

    # normalization
    normalize_obs = args.normalize_obs or args.norm_obs
    normalize_rew = args.normalize_rew or args.norm_rew
    if normalize_obs or normalize_rew:
        venv = VecNormalize(venv, norm_obs=normalize_obs, norm_reward=normalize_rew, clip_obs=10.0)

    # algo + policy kwargs
    Algo = ALGOS[args.algo]
    policy_kwargs: Dict[str, Any] = {}
    if args.net_arch:
        policy_kwargs["net_arch"] = parse_net_arch(args.net_arch)

    # consolidate common kwargs
    kwargs: Dict[str, Any] = dict(verbose=1, tensorboard_log=args.tb_dir, policy_kwargs=policy_kwargs or None)
    lr = args.learning_rate if args.learning_rate is not None else args.lr
    if lr is not None:
        kwargs["learning_rate"] = lr
    if args.gamma is not None:
        kwargs["gamma"] = args.gamma

    # algo-specific
    if args.algo in ("ppo", "a2c"):
        if args.n_steps is not None: kwargs["n_steps"] = args.n_steps
        if args.algo == "ppo":
            if args.batch_size is not None: kwargs["batch_size"] = args.batch_size
            if args.n_epochs is not None: kwargs["n_epochs"] = args.n_epochs
    elif args.algo == "dqn":
        if args.buffer_size is not None: kwargs["buffer_size"] = args.buffer_size
        if args.learning_starts is not None: kwargs["learning_starts"] = args.learning_starts
        if args.train_freq is not None: kwargs["train_freq"] = args.train_freq
        if args.gradient_steps is not None: kwargs["gradient_steps"] = args.gradient_steps
        if args.batch_size is not None: kwargs["batch_size"] = args.batch_size
        if args.target_update_interval is not None: kwargs["target_update_interval"] = args.target_update_interval

    model = Algo("MlpPolicy", venv, **kwargs)

    # callbacks
    tb_cb = TBTelemetryCallback(
        keys=["pellets_left","ghosts_eaten","power_timer","power_collected","died","cleared","timeout"],
        prefix="env",
        dump_every=1000,
    )
    cbs: List[Any] = [tb_cb]

    try:
        csv_cb = EpisodeCSVLogger(
            save_dir=os.path.dirname(args.csv_file),
            filename=os.path.basename(args.csv_file),
            run_id=run_id,
            seed=args.seed,
            preset="2d",
            mode="agent",
            algo=args.algo,
            total_timesteps=args.steps,
        )
        cbs.append(csv_cb)
    except Exception:
        # if signature differs in your project, skip CSV and keep TB-only
        pass

    if args.save_every and args.save_every > 0:
        ckpt_dir = os.path.join(logs_base, "ckpt", run_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        cbs.append(CheckpointCallback(save_freq=args.save_every, save_path=ckpt_dir, name_prefix="model"))

    # learn
    try:
        model.learn(total_timesteps=args.steps, callback=cbs, log_interval=10, progress_bar=True, tb_log_name=run_id)
    except TypeError:
        model.learn(total_timesteps=args.steps, callback=cbs, log_interval=10, tb_log_name=run_id)

    # save
    model_path = Path(os.path.join(logs_base, f"{run_id}.zip"))
    model.save(str(model_path))
    if isinstance(venv, VecNormalize):
        venv.save(str(model_path.with_suffix(".vecnormalize.pkl")))


if __name__ == "__main__":
    main()
