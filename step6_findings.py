"""Step 6: findings for the presentation.

- Results by market period, the worst months and the biggest misses
- Turning points: how often the score flipped before a metro's price trend did
- The 2022 case study (Sun Belt down, Boston and Chicago up)
- Slide-ready charts in results/figures/

    python step6_findings.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS, PERIODS

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
GREEN, RED, AMBER, INK, MUTED, GRID = "#2F6B45", "#B5533C", "#D08A1E", "#1E2421", "#5E6863", "#E6E9E7"
CASE_DOWN = ["Phoenix", "Austin", "Las Vegas", "Salt Lake City"]
CASE_UP = ["Chicago", "Boston"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#B8BFBB",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def rank_corr(g):
    return g["forecast"].rank().corr(g["realized_vs_avg_pct"].rank())


def main():
    h = pd.read_csv(RESULTS / "index_history.csv", parse_dates=["date"]).sort_values(["metro", "date"])
    k = h.dropna(subset=["realized_vs_avg_pct"])
    monthly = k.groupby("date").apply(rank_corr, include_groups=False)

    # ---- by period
    rows = []
    for name, (s, e) in PERIODS.items():
        d = k[(k["date"] >= s) & (k["date"] <= e)]
        if d.empty:
            continue
        m = monthly[(monthly.index >= s) & (monthly.index <= e)]
        top = d[d["rank"] <= 5]
        rows.append({"period": name, "months": int(d["date"].nunique()),
                     "ranking_accuracy": round(float(m.mean()), 2),
                     "worst_month": round(float(m.min()), 2),
                     "top5_beat_avg": round(float((top["realized_vs_avg_pct"] > 0).mean()), 2)})
    periods = pd.DataFrame(rows)
    periods.to_csv(RESULTS / "period_summary.csv", index=False)

    # ---- biggest misses
    k = k.assign(miss=k["realized_vs_avg_pct"] - k["forecast"])
    misses = (k.reindex(k["miss"].abs().sort_values(ascending=False).index)
              [["date", "metro", "forecast", "realized_vs_avg_pct", "miss"]].head(8))
    misses["date"] = misses["date"].dt.strftime("%b %Y")
    misses.to_csv(RESULTS / "biggest_misses.csv", index=False)

    # ---- turning points: a lasting flip in the 3-month price trend vs. the average metro
    leads = []
    for metro, g in h.groupby("metro"):
        g = g.reset_index(drop=True)
        s = np.sign(g["price_vs_avg"]).values
        for i in range(3, len(g) - 2):
            if s[i] != s[i - 1] and (s[i:i + 3] == s[i]).all() and (s[i - 3:i] == s[i - 1]).all():
                up = s[i] > 0
                ok = (g["score"] >= 50) if up else (g["score"] < 50)
                j = i
                while j - 1 >= 0 and ok[j - 1] and i - (j - 1) <= 12:
                    j -= 1
                leads.append({"metro": metro, "date": g["date"][i], "direction": "up" if up else "down",
                              "months_early": (i - j) if ok[i] else -1})
    L = pd.DataFrame(leads)
    L.to_csv(RESULTS / "turning_points.csv", index=False)
    turns = {"turns": int(len(L)),
             "flagged_before": round(float((L["months_early"] >= 1).mean()), 2),
             "missed": round(float((L["months_early"] < 0).mean()), 2),
             "median_months_early_up": float(L.loc[L["direction"] == "up", "months_early"].clip(lower=0).median()),
             "median_months_early_down": float(L.loc[L["direction"] == "down", "months_early"].clip(lower=0).median())}

    # ---- 2022 case study data
    cs = h[h["metro"].isin(CASE_DOWN + CASE_UP) & (h["date"] >= "2021-07-01") & (h["date"] <= "2023-06-01")]
    cs[["metro", "date", "score", "stage", "price_vs_avg", "zhvi"]].to_csv(RESULTS / "case_study_2022.csv", index=False)

    summary = {"periods": periods.to_dict(orient="records"), "turning_points": turns,
               "worst_months": {d.strftime("%b %Y"): round(float(v), 2) for d, v in monthly.nsmallest(3).items()}}
    (RESULTS / "findings.json").write_text(json.dumps(summary, indent=2))

    # ================= slide charts =================
    track = json.loads((RESULTS / "track_record.json").read_text())

    # 1. Case study: small multiples, score line + when the price trend flipped
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
    for ax, metro in zip(axes.flat, CASE_DOWN[:3] + CASE_UP + CASE_DOWN[3:]):
        d = (cs[cs["metro"] == metro].set_index("date")
             .reindex(pd.date_range("2021-07-01", "2023-06-01", freq="MS")))  # gaps stay gaps
        up = metro in CASE_UP
        ax.axhspan(50, 112, color=GREEN, alpha=0.05, lw=0)
        ax.axhspan(0, 50, color=RED, alpha=0.05, lw=0)
        ax.axhline(50, color="#8C958F", lw=1, ls=":")
        ax.plot(d.index, d["score"], color=INK, lw=2)
        if up:
            flip = L[(L["metro"] == metro) & (L["direction"] == "up") & (L["date"] >= "2022-06-01")
                     & (L["date"] <= "2023-03-01")]
            mark, label, col = (flip["date"].iloc[0] if len(flip) else None), " prices start beating avg", GREEN
        else:
            mark, label, col = d["zhvi"].idxmax(), " home values peak", RED
        if mark is not None:
            ax.axvline(mark, color=col, lw=1.5, ls="--")
            ax.text(mark, 104, label, color=col, fontsize=9, va="bottom")
        ax.set_title(metro, loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color=GRID)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    fig.suptitle("2022: the score dropped as Sun Belt prices peaked, and rose before Chicago and Boston rebounded",
                 x=0.06, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.06, 0.915, "Housing Momentum Score (0-100), above 50 = expected to beat the average metro. "
            " Dashed line: when home values peaked (Sun Belt) or started beating the average (Chicago, Boston).",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0.03, 0, 1, 0.9))
    fig.savefig(FIG / "case_study_2022.png", dpi=200)
    plt.close(fig)

    # 2. Score bands
    bands = pd.read_csv(RESULTS / "score_bands.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    vals = bands["avg_next_3m_vs_avg_pct"]
    bars = ax.bar(bands["band"], vals, color=[RED if v < 0 else GREEN for v in vals], width=0.62)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in vals], padding=3, color=INK)
    ax.axhline(0, color="#8C958F", lw=1)
    ax.set_xlabel("Housing Momentum Score")
    ax.set_ylabel("Next 3 months vs. average metro (pts)")
    ax.set_title("Higher score, better next 3 months", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "score_bands.png", dpi=200)
    plt.close(fig)

    # 3. What predicts
    dc = {k_: v for k_, v in track["driver_rank_corr"].items() if k_ != "score"}
    items = sorted(dc.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh([a for a, _ in items], [b for _, b in items],
                   color=[GREEN if b >= 0.3 else "#9DB8A6" for _, b in items], height=0.6)
    ax.bar_label(bars, labels=[f"{b:.2f}" for _, b in items], padding=3, color=INK)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Rank correlation with the next 3 months (0 = no link)")
    ax.set_title("Market data predicts; search barely does", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "what_predicts.png", dpi=200)
    plt.close(fig)

    # 4. Search vs. no search, vs. naive
    card = pd.read_csv(RESULTS / "scorecard.csv", index_col=0)
    names = {"Naive": "Trend continues", "A": "Housing data", "B": "+ Google search",
             "C": "+ Home Depot search"}
    c = card.loc[[m for m in names if m in card.index], "rank_corr"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar([names[m] for m in c.index], c.values,
                  color=["#B8BFBB", GREEN, "#9DB8A6", "#9DB8A6"][:len(c)], width=0.6)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in c.values], padding=3, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ranking accuracy")
    ax.set_title("Housing data beats the baseline; search adds nothing", loc="left",
                 fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "search_vs_no_search.png", dpi=200)
    plt.close(fig)

    print("BY PERIOD"); print(periods.to_string(index=False))
    print("\nTURNING POINTS", turns)
    print("\nBIGGEST MISSES"); print(misses.round(2).to_string(index=False))
    print(f"\nSlide charts saved in {FIG}")


if __name__ == "__main__":
    main()
