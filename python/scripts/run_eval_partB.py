
#!/usr/bin/env python
import argparse, os, csv, numpy as np
from stable_baselines3.common.monitor import Monitor
from env.thermal_env import ThermalEnv, ThermalEnvConfig
from scripts.config_presets_partB import PRESETS_B, CURRICULUM_B

def make_env(seed: int, preset: str, stage: str):
    p = PRESETS_B.get(preset, PRESETS_B["baseline"]).copy()
    cfg = ThermalEnvConfig(
        n_cores=p["n_cores"], dt=0.1, episode_len=256,
        arrival_rate_hz=p["arrival_rate_hz"], deadline_s=0.5,
        service_coeff_tps_per_ghz=60.0, queue_limit_jobs=200,
        power_cap_w=p["power_cap_w"], use_battery=p["use_battery"], battery_capacity_Wh=p["battery_capacity_Wh"],
        ambient_C=p["ambient_C"], T_safe_C=p["T_safe_C"], seed=seed, preset=preset, mode="control",
        obs_lag_steps=p.get("obs_lag_steps", 0),
    )
    env = ThermalEnv(cfg)
    cur = CURRICULUM_B.get(stage, CURRICULUM_B["mid"])
    env.cfg.freq_step_rel_max = cur["freq_step_rel_max"]
    env.cfg.pwm_step_max = cur["pwm_step_max"]
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
    return {"ret": ep_ret, "len": ep_len, "uniq_miss": int(summ.get("uniq_flows_missed", 0)),
            "first_miss_t": float(summ.get("first_miss_t", -1.0))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[111,222,333])
    ap.add_argument("--presets", type=str, nargs="+", default=list(PRESETS_B.keys()))
    ap.add_argument("--stage", type=str, default="mid")
    ap.add_argument("--out", type=str, default="logs/eval_partB.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["preset","seed","stage","ret","len","uniq_miss","first_miss_t"])
        for preset in args.presets:
            for sd in args.seeds:
                env = make_env(sd, preset, args.stage)
                res = run_episode(env)
                w.writerow([preset, sd, args.stage, res["ret"], res["len"], res["uniq_miss"], res["first_miss_t"]])
                env.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
