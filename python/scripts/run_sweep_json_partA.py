import argparse, json, subprocess, os, shlex, datetime, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Pfad zu sweep JSON")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="0 = alle")
    args = ap.parse_args()

    cfg = json.load(open(args.cfg, "r", encoding="utf-8"))
    algos = cfg.get("algos", ["ppo"])
    seeds = cfg.get("seeds", [123])
    n_envs = int(cfg.get("n_envs", 1))
    normalize_obs = bool(cfg.get("normalize_obs", False))
    normalize_rew = bool(cfg.get("normalize_rew", False))
    steps = int(cfg.get("steps", 200000))
    common = cfg.get("common", {})
    overrides = cfg.get("algo_overrides", {})

    os.makedirs("logs/sweeps", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(f"logs/sweeps/manifest_partA_{stamp}.csv")
    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write("algo,seed,steps,n_envs,normalize_obs,normalize_rew,args_line\n")
        runs = 0
        for algo in algos:
            for sd in seeds:
                if args.max_runs and runs >= args.max_runs:
                    break
                cmd = [sys.executable,"-m","scripts.run_train_partA","--algo",algo,"--seed",str(sd),
                       "--steps",str(steps),"--n-envs",str(n_envs)]
                if normalize_obs: cmd.append("--normalize-obs")
                if normalize_rew: cmd.append("--normalize-rew")
                for key, val in common.items():
                    cmd += [f"--{key.replace('_','-')}", str(val)]
                for key, val in overrides.get(algo, {}).items():
                    cmd += [f"--{key.replace('_','-')}", str(val)]
                line = " ".join(shlex.quote(x) for x in cmd)
                manifest.write(f"{algo},{sd},{steps},{n_envs},{int(normalize_obs)},{int(normalize_rew)},{line}\n")
                manifest.flush()
                print(f"[RUN] {line}")
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
                runs += 1
    print("Sweep manifest:", str(manifest_path))

if __name__ == "__main__":
    main()
