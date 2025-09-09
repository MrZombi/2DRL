#!/usr/bin/env python
import argparse, os, csv, numpy as np
from stable_baselines3.common.monitor import Monitor
from env.bus_env import BusEnv, EnvConfig
from scripts.config_presets_partC import PRESETS, CURRICULUM

def make_env(seed: int, mode: str, preset: str, stage: str, deterministic: bool):
    p = PRESETS.get(preset, PRESETS["baseline"]).copy()
    if deterministic:
        p["event_rate_hz"] = 0.0
        p["bit_error_rate"] = 0.0
    cfg = EnvConfig(
        n_flows=p["n_flows"], dt=0.01, bandwidth_Bps=p["bandwidth_Bps"], header_overhead=16,
        size_scale=1.0, arbitration=True, bit_error_rate=p["bit_error_rate"], queue_limit_bytes=4096,
        event_rate_hz=p["event_rate_hz"], episode_len=512, early_stop_patience=128,
        mode=mode, preset=preset, seed=seed, obs_lag_steps=p.get("obs_lag_steps", 0),
    )
    env = BusEnv(cfg)
    cur = CURRICULUM.get(stage, CURRICULUM["mid"])
    env.delta_period_rel_max = cur["delta_period_rel_max"]
    env.delta_jitter_rel_max = cur["delta_jitter_rel_max"]
    env.delta_offset_rel_max = cur["delta_offset_rel_max"]
    env.delta_bg_abs_max     = cur["delta_bg_abs_max"]
    env.delta_noise_abs_max  = cur["delta_noise_abs_max"]
    env.delta_scale_abs_max  = cur["delta_scale_abs_max"]
    return Monitor(env)

def run_episode(env) -> dict:
    obs, info = env.reset()
    ep_ret, ep_len = 0.0, 0
    done = False
    while not done:
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        ep_ret += r; ep_len += 1
        done = (term or trunc)
    summ = info.get("episode_summary", {})
    return {"ret": ep_ret, "len": ep_len, "uniq_miss": int(summ.get("uniq_flows_missed", 0)), "first_miss_t": float(summ.get("first_miss_t", -1.0))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[111,222,333])
    ap.add_argument("--presets", type=str, nargs="+", default=["baseline","stress_high_bg"])
    ap.add_argument("--stage", type=str, default="mid")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", type=str, default="logs/eval_sweep.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["preset","seed","stage","deterministic","ret","len","uniq_miss","first_miss_t"])
        for preset in args.presets:
            for sd in args.seeds:
                env = make_env(sd, "explore", preset, args.stage, args.deterministic)
                res = run_episode(env)
                w.writerow([preset, sd, args.stage, int(args.deterministic), res["ret"], res["len"], res["uniq_miss"], res["first_miss_t"]])
                env.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
