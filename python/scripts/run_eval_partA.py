from __future__ import annotations
import argparse, os, time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor

try:
    from torch.utils.tensorboard import SummaryWriter  # TB-first
except Exception:  # pragma: no cover
    SummaryWriter = None  # fällt zurück auf Console-only

from env.two_d_env import TwoDEnv, EnvConfig

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), required=True)
    ap.add_argument("--model", required=True, help="Pfad zum SB3-Model (.zip)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)

    # Server/Env
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-every", type=int, default=1)

    # TensorBoard
    ap.add_argument("--tb-dir", default="tb/partA_eval")
    ap.add_argument("--run-id", default=None)

    args = ap.parse_args()

    run_id = args.run_id or f"eval_partA_{args.algo}_{int(time.time())}"
    if SummaryWriter and args.tb_dir:
        os.makedirs(args.tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tb_dir)
    else:
        writer = None

    # Env
    cfg = EnvConfig(host=args.host, port=args.port, seed=args.seed, frame_skip=args.frame_skip,
                    render=args.render, render_every=args.render_every, agent="eval", run_id=run_id)
    env = Monitor(TwoDEnv(cfg))

    # Model
    Algo = ALGOS[args.algo]
    model = Algo.load(args.model, print_system_info=False)

    returns = []
    lengths = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(int(action))
            done = term or trunc
            ep_ret += float(rew)
            ep_len += 1
            if done:
                returns.append(ep_ret)
                lengths.append(ep_len)
                if writer:
                    writer.add_scalar("eval/return", ep_ret, ep)
                    writer.add_scalar("eval/length", ep_len, ep)
                    # logge einige env-Infos, falls vorhanden
                    for k in ("pellets_left", "ghosts_eaten", "power_timer", "died", "cleared", "timeout"):
                        if k in info:
                            try:
                                writer.add_scalar(f"eval/{k}", float(info[k]), ep)
                            except Exception:
                                pass
                print(f"[EVAL] ep={ep} return={ep_ret:.3f} len={ep_len} reason={info.get('terminated_reason','-')}")
                break

    env.close()
    if writer:
        writer.flush(); writer.close()

    if returns:
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        mean_l = float(np.mean(lengths))
        print(f"[SUMMARY] episodes={len(returns)} mean_return={mean_r:.3f}±{std_r:.3f} mean_len={mean_l:.1f}")


if __name__ == "__main__":
    main()