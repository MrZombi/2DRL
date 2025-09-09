import argparse, os, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/episodes.csv")
    ap.add_argument("--out", default="eval/plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "episode" not in df.columns: raise SystemExit("episodes.csv ohne 'episode'")

    def plt_save(x, y, name, ylabel):
        plt.figure()
        plt.plot(df[x], df[y])
        plt.xlabel(x); plt.ylabel(ylabel); plt.title(name)
        png = os.path.join(args.out, f"{name}.png")
        svg = os.path.join(args.out, f"{name}.svg")
        plt.savefig(png, bbox_inches="tight"); plt.savefig(svg, bbox_inches="tight"); plt.close()
        print(" -", png); print(" -", svg)

    print("Gespeichert:")
    if "ep_return" in df.columns:
        plt_save("episode","ep_return","return_over_episodes","Return")
    if "uniq_flows_missed" in df.columns:
        plt_save("episode","uniq_flows_missed","uniq_misses_over_episodes","uniq_misses")
    if "mean_util" in df.columns:
        plt_save("episode","mean_util","util_over_episodes","util")

if __name__ == "__main__":
    main()
