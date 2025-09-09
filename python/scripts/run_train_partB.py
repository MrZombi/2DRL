import argparse, os, sys, ast, pathlib
from typing import List, Dict, Any
from pathlib import Path
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks.episode_csv_logger import EpisodeCSVLogger
from callbacks.tb_telemetry_callback import TBTelemetryCallback
from env.thermal_env import ThermalEnv, ThermalEnvConfig
from scripts.config_presets_partB import PRESETS_B, CURRICULUM_B

ALGOS = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC}

def parse_net_arch(s: str | None):
    if not s: return None
    try:
        if "," in s:
            return [int(x) for x in s.split(",") if x.strip()]
        return ast.literal_eval(s)
    except Exception:
        return None

def make_env(seed: int, mode: str, preset: str, stage: str, ablate: List[str]):
    p = PRESETS_B.get(preset, PRESETS_B["baseline"]).copy()
    cfg = ThermalEnvConfig(
        n_cores=p["n_cores"], dt=0.1, episode_len=1024,
        arrival_rate_hz=p["arrival_rate_hz"], deadline_s=0.5,
        service_coeff_tps_per_ghz=60.0, queue_limit_jobs=200,
        freq_min_ghz=0.5, freq_max_ghz=4.0, freq_step_rel_max=0.10,
        pwm_step_max=0.15, cooling_coeff_base=0.06, cooling_coeff_fan=0.22, temp_noise_std=0.03,
        base_power_w=8.0, core_dyn_power_coeff=20.0, dyn_power_exp=1.4,
        fan_power_base_w=0.6, fan_power_coeff_w=5.0,
        use_battery=p["use_battery"], battery_capacity_Wh=p["battery_capacity_Wh"],
        battery_init_Wh=None, power_cap_w=p["power_cap_w"],
        T_init_C=45.0, ambient_C=p["ambient_C"], ambient_drift_C_per_s=0.0,
        T_safe_C=p["T_safe_C"], throttle_freq_drop_rel=0.2,
        util_low_thresh=0.25, util_high_thresh=0.95, queue_high_thresh=0.85,
        seed=seed, mode="control", preset=preset, obs_lag_steps=p.get("obs_lag_steps", 0),
        early_stop_patience=0, ablate=ablate,
    )
    env = ThermalEnv(cfg)
    cur = CURRICULUM_B.get(stage, CURRICULUM_B["mid"])
    env.cfg.freq_step_rel_max = cur["freq_step_rel_max"]
    env.cfg.pwm_step_max = cur["pwm_step_max"]
    return Monitor(env)

def build_vec_env(n_envs: int, **kwargs_env):
    fns = []
    base_seed = kwargs_env["seed"]
    for i in range(n_envs):
        def _make(i=i):
            k = kwargs_env.copy()
            k["seed"] = base_seed + i
            return make_env(**k)
        fns.append(_make)
    return DummyVecEnv(fns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--preset", type=str, default="baseline", choices=list(PRESETS_B.keys()))
    ap.add_argument("--stage", type=str, default="mid", choices=list(CURRICULUM_B.keys()))
    ap.add_argument("--ablate", type=str, default="")
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--normalize-obs", action="store_true")
    ap.add_argument("--normalize-rew", action="store_true")
    ap.add_argument("--id", type=str, default="")
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--net-arch", type=str, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--n-epochs", type=int, default=None)
    ap.add_argument("--buffer-size", type=int, default=None)
    ap.add_argument("--train-freq", type=int, default=None)
    ap.add_argument("--gradient-steps", type=int, default=None)
    ap.add_argument("--learning-starts", type=int, default=None)
    ap.add_argument("--target-update-interval", type=int, default=None)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--csv-file", type=str, default="logs/partB/episodes.csv")
    ap.add_argument("--tb-dir", type=str, default="tb/partB")
    args = ap.parse_args()
    logs_base = os.path.dirname(args.csv_file) or "logs/partB"
    os.makedirs(logs_base, exist_ok=True)

    ablate = [s.strip() for s in args.ablate.split(",") if s.strip()]
    run_id = f"{args.algo}_s{args.seed}_{args.preset}_{args.stage}" + (f"_{args.id}" if args.id else "")
    os.makedirs(args.tb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    logs_base = os.path.dirname(args.csv_file) or "logs/partB"
    os.makedirs(logs_base, exist_ok=True)

    venv = build_vec_env(
        n_envs=max(1, args.n_envs),
        seed=args.seed, mode="control", preset=args.preset, stage=args.stage, ablate=ablate
    )
    venv = VecNormalize(venv, norm_obs=args.normalize_obs, norm_reward=args.normalize_rew, clip_obs=5.0)

    Algo = ALGOS[args.algo]
    policy_kwargs: Dict[str, Any] = {}
    na = parse_net_arch(args.net_arch)
    if na is not None: policy_kwargs["net_arch"] = na
    kwargs: Dict[str, Any] = dict(
        verbose=1,
        tensorboard_log=args.tb_dir,
        policy_kwargs=policy_kwargs
    )

    if args.learning_rate is not None: kwargs["learning_rate"] = args.learning_rate
    if args.gamma is not None: kwargs["gamma"] = args.gamma

    if args.algo == "ppo":
        if args.n_steps is not None: kwargs["n_steps"] = args.n_steps
        if args.batch_size is not None: kwargs["batch_size"] = args.batch_size
        if args.n_epochs is not None: kwargs["n_epochs"] = args.n_epochs

    elif args.algo == "a2c":
        if args.n_steps is not None: kwargs["n_steps"] = args.n_steps

    else:
        if args.buffer_size is not None: kwargs["buffer_size"] = args.buffer_size
        if args.learning_starts is not None: kwargs["learning_starts"] = args.learning_starts
        if args.train_freq is not None: kwargs["train_freq"] = args.train_freq
        if args.gradient_steps is not None: kwargs["gradient_steps"] = args.gradient_steps
        if args.batch_size is not None: kwargs["batch_size"] = args.batch_size
        kwargs.pop("target_update_interval", None)

    model = None
    if args.resume:
        model_path = pathlib.Path(args.resume)
        if not model_path.exists():
            print(f"[WARN] resume path not found: {model_path}", file=sys.stderr)
        else:
            vn_path = model_path.with_suffix(".vecnormalize.pkl")  # passend zum Save
            if vn_path.exists():
                venv = VecNormalize.load(str(vn_path), venv)
            model = Algo.load(str(model_path), env=venv, tensorboard_log=args.tb_dir)
            print(f"[INFO] Resumed from {model_path}")

    if model is None:
        model = Algo("MlpPolicy", venv, **kwargs)

    csv_cb = EpisodeCSVLogger(
        save_dir=os.path.dirname(args.csv_file),
        filename=os.path.basename(args.csv_file),
        run_id=run_id
    )
    tb_cb = TBTelemetryCallback()
    cbs = [csv_cb, tb_cb]
    if args.save_every and args.save_every > 0:
        ckpt_dir = os.path.join(logs_base, "ckpt", run_id)
        model_path = Path(os.path.join(logs_base, f"{run_id}.zip"))
        cbs.append(CheckpointCallback(save_freq=args.save_every, save_path=ckpt_dir, name_prefix="model"))

    try:
        import tqdm, rich  # noqa: F401
        use_pb = True
    except Exception:
        use_pb = False

    model.learn(
        total_timesteps=args.steps,
        callback=cbs, log_interval=10, progress_bar=use_pb,
        tb_log_name=run_id
    )
    model_path = Path(os.path.join(logs_base, f"{run_id}.zip"))
    model.save(str(model_path))
    venv.save(str(model_path.with_suffix(".vecnormalize.pkl")))

if __name__ == "__main__":
    main()
