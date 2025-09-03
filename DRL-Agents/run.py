import argparse, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A","B","C"], required=True)
    ap.add_argument("--algo", help="ppo/a2c/dqn für A; sac/td3/ppo für B/C", required=True)
    ap.add_argument("--mode", choices=["train","eval","faults"], default="train")
    ap.add_argument("--episodes", type=int, default=20)   # für eval/faults
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.part == "A":
        if args.mode == "train":
            script = {"ppo":"train_ppo.py","a2c":"train_a2c.py","dqn":"train_dqn.py"}[args.algo]
            return subprocess.call([sys.executable, os.path.join(here, "partA_2d_env", script)])
        elif args.mode == "eval":
            return subprocess.call([sys.executable, os.path.join(here, "partA_2d_env", "eval_latency.py")])
        elif args.mode == "faults":
            return subprocess.call([sys.executable, os.path.join(here, "partA_2d_env", "fault_campaign.py"),
                                    "--algo", args.algo, "--episodes", str(args.episodes)])

    if args.part == "B":
        script = {"sac":"train_sac.py","td3":"train_td3.py","ppo":"train_ppo_cont.py"}[args.algo]
        return subprocess.call([sys.executable, os.path.join(here, "partB_thermal", script)])

    if args.part == "C":
        if args.mode == "train":
            script = {"sac":"train_fuzzer_sac.py","td3":"train_fuzzer_td3.py","ppo":"train_fuzzer_ppo.py"}[args.algo]
            return subprocess.call([sys.executable, os.path.join(here, "partC_bus_fuzz", script)])
        elif args.mode == "eval":
            return subprocess.call([sys.executable, os.path.join(here, "partC_bus_fuzz", "eval_fuzzer.py"),
                                    "--algo", args.algo])
        elif args.mode == "faults":
            return subprocess.call([sys.executable, os.path.join(here, "partC_bus_fuzz", "fault_campaign_bus.py"),
                                    "--algo", args.algo, "--episodes", str(args.episodes)])

if __name__ == "__main__":
    sys.exit(main())
