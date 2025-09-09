import argparse, pandas as pd, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/episodes.csv")
    ap.add_argument("--out", default="logs/summary_agg.csv")
    ap.add_argument("--last-k", type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV nicht gefunden: {args.csv}")
    df = pd.read_csv(args.csv)
    if "run_id" not in df:
        raise SystemExit("Spalte 'run_id' fehlt in episodes.csv")
    if "algo" not in df.columns:
        df["algo"] = df["run_id"].str.split("_").str[0]

    rows = []
    for run, grp in df.groupby("run_id"):
        grp = grp.sort_values("episode")
        tail = grp.tail(args.last_k)
        rows.append({
            "run_id": run,
            "algo": tail["algo"].iloc[0] if "algo" in tail.columns else "",
            "episodes": int(tail["episode"].max() if "episode" in tail else len(tail)),
            "ep_return_mean_lastK": float(tail["ep_return"].mean() if "ep_return" in tail else 0.0),
            "uniq_misses_mean_lastK": float(tail["uniq_flows_missed"].mean() if "uniq_flows_missed" in tail else 0.0),
            "mean_util_lastK": float(tail["mean_util"].mean() if "mean_util" in tail else 0.0),
            "cooldown_hits_lastK": float(tail["cooldown_hits"].mean() if "cooldown_hits" in tail else 0.0),
        })
    out = pd.DataFrame(rows).sort_values(["algo","ep_return_mean_lastK"], ascending=[True, False])
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
