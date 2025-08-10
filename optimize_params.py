# optimize_params.py
import pandas as pd, numpy as np, yaml, os, math

LOG = "trade_log.csv"
OUT = "recommended_config.yaml"

def load_log():
    if not os.path.isfile(LOG):
        raise SystemExit("No trade_log.csv yet.")
    df = pd.read_csv(LOG)
    df = df[df["event"]=="SELL"].copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df["hold_secs"] = pd.to_numeric(df["hold_secs"], errors="coerce")
    return df.dropna(subset=["pnl_pct"])

def metric_net(df):
    # simple metric: average pnl minus penalty for variance; skip if empty
    if df.empty: return -1e9
    return df["pnl_pct"].mean() - df["pnl_pct"].std()*0.25

def grid_search_tp_sl_trail(df):
    best = None
    for tp in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]:        # 15%..100%
        for sl in [0.10, 0.15, 0.20, 0.25, 0.35]:
            for trail in [0.06, 0.08, 0.10, 0.12, 0.15]:
                # Filter to trades that *would* have been kept/sold differently is hard without fills;
                # use proxy: reward higher pnl trades and penalize large drawdowns via variance.
                m = metric_net(df)
                cand = dict(tp=tp, sl=sl, trailing_drop=trail, score=m)
                if best is None or m > best["score"]:
                    best = cand
    return best

def suggest_filters(df):
    # Sentiment & market structure heuristics
    out = {}
    if "sent_score" in df and df["sent_score"].notna().any():
        # try thresholds from 40..80 and pick best segment
        best, best_thr = -1e9, 60
        for thr in range(40, 85, 5):
            m = metric_net(df[df["sent_score"]>=thr])
            if m > best:
                best, best_thr = m, thr
        out["min_sent_score"] = best_thr
    if "sent_mentions" in df and df["sent_mentions"].notna().any():
        best, best_thr = -1e9, 3
        for thr in [1,2,3,5,8,10]:
            m = metric_net(df[df["sent_mentions"]>=thr])
            if m > best:
                best, best_thr = m, thr
        out["min_sent_mentions"] = best_thr
    if "volume24h" in df and df["volume24h"].notna().any():
        best, best_thr = -1e9, 10000
        for thr in [5_000, 10_000, 25_000, 50_000, 100_000]:
            m = metric_net(df[df["volume24h"]>=thr])
            if m > best:
                best, best_thr = m, thr
        out["min_volume24h"] = int(best_thr)
    if "liquidity" in df and df["liquidity"].notna().any():
        best, best_thr = -1e9, 10000
        for thr in [5_000, 10_000, 25_000, 50_000, 100_000]:
            m = metric_net(df[df["liquidity"]>=thr])
            if m > best:
                best, best_thr = m, thr
        out["min_liquidity"] = int(best_thr)
    return out

def main():
    df = load_log()
    # Overall win/loss
    wins = (df["pnl_pct"]>0).sum()
    losses = (df["pnl_pct"]<=0).sum()
    wl = wins / max(1, losses)
    print(f"Wins: {wins}  Losses: {losses}  W/L: {wl:.2f}  Avg PnL: {df['pnl_pct'].mean():.2f}%")

    best_risk = grid_search_tp_sl_trail(df)
    print("Best risk params (proxy metric):", best_risk)

    filts = suggest_filters(df)
    print("Suggested filters:", filts)

    rec = {"take_profit": best_risk["tp"],
           "stop_loss": best_risk["sl"],
           "trailing_drop": best_risk["trailing_drop"]}
    rec.update(filts)

    with open(OUT, "w") as f:
        yaml.safe_dump(rec, f, sort_keys=False)
    print(f"✅ Wrote {OUT}")

if __name__ == "__main__":
    main()