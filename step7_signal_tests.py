"""Step 7: does any extra signal help, 3, 6 or 12 months ahead?

For each signal: model with housing data only vs. housing data + that signal, same rows,
same month-by-month test. A signal "helps" if it lowers forecast error significantly.

    python step7_signal_tests.py
    python step7_signal_tests.py --raw data/raw_real
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED, RAW, RESULTS, TEST_START
from step3_model import ridge

HORIZONS = (3, 6, 12)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical order
INK, MUTED, GRID = "#1E2421", "#5E6863", "#E6E9E7"


def dm_test(err_a, err_b, dates, lag):
    """Diebold-Mariano: is model B's squared error lower than A's? Newey-West for overlap."""
    d = pd.Series(err_a ** 2 - err_b ** 2).groupby(dates.values).mean().values
    T = len(d)
    dc = d - d.mean()
    var = dc @ dc / T + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / T) if var > 0 else 0.0
    return float(stat), float(1 - stats.norm.cdf(stat))


def walk(df, cols, target, h):
    known = df.dropna(subset=[target])
    out = []
    for m in sorted(known.loc[known["date"] >= TEST_START, "date"].unique()):
        train = known[known["date"] <= pd.Timestamp(m) - pd.DateOffset(months=h)]
        test = known[known["date"] == m]
        if len(train) < 200 or train["date"].nunique() < 12:
            continue
        p = ridge().fit(train[cols], train[target]).predict(test[cols])
        out.append(pd.DataFrame({"date": test["date"].values, "metro": test["metro"].values,
                                 "y": test[target].values, "p": p}))
    return pd.concat(out) if out else None


def main(raw):
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    fs = pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"]
    feats = {k: v.split(",") for k, v in fs.items()}

    # forward price change h months ahead, vs. the average metro (from the full Zillow series)
    z = pd.read_csv(raw / "zillow_clean.csv", parse_dates=["date"]).sort_values(["metro", "date"])
    for h in HORIZONS:
        z[f"fwd_{h}"] = np.log(z.groupby("metro")["zhvi"].shift(-h)) - np.log(z["zhvi"])
        z[f"y_{h}"] = z[f"fwd_{h}"] - z.groupby("date")[f"fwd_{h}"].transform("mean")
    df = panel.merge(z[["metro", "date"] + [f"y_{h}" for h in HORIZONS]], on=["metro", "date"], how="left")

    base = feats["A"]
    search = [c for c in feats["B"] if c not in base]
    signals = {"Out-of-town search": [c for c in search if "outoftown" in c],
               "Local buyer search": [c for c in search if "outoftown" not in c],
               "Home Depot / Lowe's searches": [c for c in feats["C"] if c not in feats["B"]]}
    if "spending" in feats:
        signals["Home-improvement store sales (Census)"] = feats["spending"]
    if "jobs" in feats:
        signals["Metro jobs and unemployment (BLS)"] = feats["jobs"]

    rows = []
    for name, add in signals.items():
        add = [c for c in add if c in df]
        if not add:
            continue
        sub = df.dropna(subset=base + add)
        for h in HORIZONS:
            t = f"y_{h}"
            a, b = walk(sub, base, t, h), walk(sub, base + add, t, h)
            if a is None or b is None:
                continue
            m = a.merge(b, on=["date", "metro", "y"], suffixes=("_a", "_b"))
            rc = lambda x, col: x[col].rank().corr(x["y"].rank())
            ra = m.groupby("date").apply(lambda x: rc(x, "p_a"), include_groups=False).mean()
            rb = m.groupby("date").apply(lambda x: rc(x, "p_b"), include_groups=False).mean()
            stat, p = dm_test((m["y"] - m["p_a"]).values, (m["y"] - m["p_b"]).values, m["date"], h - 1)
            rows.append({"signal": name, "horizon_months": h, "months_tested": int(m["date"].nunique()),
                         "first_month": f"{m['date'].min():%b %Y}",
                         "housing_only": round(float(ra), 3), "with_signal": round(float(rb), 3),
                         "gain": round(float(rb - ra), 3), "p_value": round(p, 3),
                         "helps": bool(p < 0.05 and rb > ra)})
            print(f"{name:40s} {h:2d}m  {ra:.3f} -> {rb:.3f}  p={p:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "signal_tests.csv", index=False)
    (RESULTS / "signal_tests.json").write_text(json.dumps(rows, indent=2))

    # chart: gain in ranking accuracy by horizon, one line per signal
    if not res.empty:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.edgecolor": "#B8BFBB",
                             "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.axhline(0, color="#8C958F", lw=1)
        for i, (name, g) in enumerate(res.groupby("signal", sort=False)):
            c = COLORS[i % len(COLORS)]
            ax.plot(g["horizon_months"], g["gain"], "-o", color=c, lw=2, ms=7, label=name)
            for _, r in g.iterrows():
                if r["helps"]:
                    ax.scatter(r["horizon_months"], r["gain"], s=160, facecolor="none", edgecolor=c, lw=1.5)
        ax.set_xticks(list(HORIZONS), [f"{h} months" for h in HORIZONS])
        ax.set_xlabel("How far ahead we predict")
        ax.set_ylabel("Change in ranking accuracy")
        ax.set_title("Does adding the signal improve the forecast?", loc="left", fontweight="bold", color=INK, pad=24)
        ax.text(0, 1.015, "Above 0 = helps. Circled = statistically significant.",
                transform=ax.transAxes, color=MUTED, fontsize=9.5)
        ax.grid(axis="y", color=GRID)
        ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        (RESULTS / "figures").mkdir(exist_ok=True)
        fig.savefig(RESULTS / "figures" / "signal_tests.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    main(Path(ap.parse_args().raw))
