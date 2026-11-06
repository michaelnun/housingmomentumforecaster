"""Step 4: the Housing Momentum Score (0-100) for every metro and month, plus
everything the dashboard shows: stages, drivers, reasons, alerts and the track record.

The score is the model's 3-month forecast (from step 3), turned into a 0-100 rank
across metros that month. The drivers show WHY a metro scores where it does.

    python step4_index.py
"""
import json

import numpy as np
import pandas as pd

from config import PROCESSED, RESULTS

TOP_K = 5

# Drivers shown on the dashboard. Each = a metro's percentile vs. other metros that month.
# Sign: +1 means a higher raw value is a stronger market.
DRIVERS = {
    "Price momentum": [("rel_price_chg_3m", +1), ("rel_price_chg_12m", +1)],
    "Market tightness": [("rel_median_dom_yoy", -1), ("rel_avg_sale_to_list_yoy", +1),
                         ("rel_inventory_yoy", -1), ("rel_months_of_supply_yoy", -1),
                         ("rel_price_drops_yoy", -1), ("rel_sold_above_list_yoy", +1)],
    "Buyer demand": [("rel_search_index", +1)],
    "Out-of-town demand": [("rel_outoftown_index", +1)],
    "Local jobs": [("rel_jobs_yoy", +1), ("rel_metro_unemp_chg_12m", -1)],  # metro (BLS) if downloaded
}
FALLBACK = {"Local jobs": [("rel_unemployment_chg_12m", -1)]}  # state unemployment otherwise

REASON_TEXT = {
    "Price momentum": ("prices have been rising faster than in other metros",
                       "prices have been lagging other metros"),
    "Market tightness": ("homes are selling faster with less inventory",
                         "homes are sitting longer and inventory is building"),
    "Buyer demand": ("local home searches are up more than elsewhere",
                     "local home searches are down more than elsewhere"),
    "Out-of-town demand": ("more people elsewhere are searching for homes here",
                           "interest from people elsewhere is fading"),
    "Local jobs": ("unemployment is improving vs. other metros",
                   "unemployment is worsening vs. other metros"),
}

# Stage = where prices have been going (last 3 months) vs. where the model expects them to go
STAGES = {
    "Rising": "Prices have been beating other metros and are expected to keep doing so",
    "Turning up": "Prices have lagged, but the model expects them to start beating other metros",
    "Turning down": "Prices have been strong, but the model expects them to fall behind",
    "Falling": "Prices have lagged and are expected to keep lagging",
}

AUDIENCE = {
    "Real estate": {
        "Rising": "Strong market: put agents and listing marketing here.",
        "Turning up": "Early opportunity: start building listings and agent capacity.",
        "Turning down": "Expect slower sales soon; prepare sellers for price cuts.",
        "Falling": "Hold spending; watch for the outlook to turn.",
    },
    "Home improvement retail": {
        "Rising": "Active market: keep move-in stock (paint, appliances) deep.",
        "Turning up": "Activity picking up: plan staff and stock ahead of it.",
        "Turning down": "Fewer moves ahead: shift toward repair and remodel products.",
        "Falling": "Keep inventory lean; lean on maintenance and pro customers.",
    },
    "Investors": {
        "Rising": "Strong momentum, but entry prices are already up.",
        "Turning up": "Potential early entry: the outlook is improving before prices show it.",
        "Turning down": "Warning sign: consider trimming exposure.",
        "Falling": "Wait for the outlook to improve before buying.",
    },
}


def build():
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    meta = json.loads((PROCESSED / "meta.json").read_text()) if (PROCESSED / "meta.json").exists() else {}
    card = pd.read_csv(RESULTS / "scorecard.csv", index_col=0)
    best = card.loc[[m for m in ("A", "B", "C") if m in card.index], "rank_corr"].idxmax()
    preds = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"])[["metro", "date", best]]

    out = panel[["metro", "date", "zhvi", "target"]].copy()
    out["price_chg_3m_pct"] = (100 * panel["price_chg_3m"]).round(2)
    out["price_vs_avg"] = (100 * panel["rel_price_chg_3m"]).round(2)

    used = []
    for name, parts in DRIVERS.items():
        cols = [(c, s) for c, s in parts if c in panel and panel[c].notna().any()]
        if not cols and name in FALLBACK:
            cols = [(c, s) for c, s in FALLBACK[name] if c in panel and panel[c].notna().any()]
        if not cols:
            continue
        raw = pd.concat([s * panel[c] / panel[c].std() for c, s in cols], axis=1).mean(axis=1)
        out[name] = (raw.groupby(panel["date"]).rank(pct=True) * 100).round(0)
        used.append(name)

    out = out.merge(preds.rename(columns={best: "forecast"}), on=["metro", "date"], how="inner")
    out["forecast"] = (100 * out["forecast"]).round(2)  # pts vs. average metro, next 3 months
    out["score"] = (out.groupby("date")["forecast"].rank(pct=True) * 100).round(0)

    momentum_up, outlook_up = out["price_vs_avg"] >= 0, out["forecast"] >= 0
    out["stage"] = np.select([momentum_up & outlook_up, ~momentum_up & outlook_up,
                              momentum_up & ~outlook_up], ["Rising", "Turning up", "Turning down"],
                             "Falling")

    out = out.sort_values(["metro", "date"])
    g = out.groupby("metro")
    out["score_chg"] = out["score"] - g["score"].shift(1)
    out["prev_stage"] = g["stage"].shift(1)
    out["rank"] = out.groupby("date")["score"].rank(ascending=False, method="first").astype(int)

    def reason(r):
        lead = (f"Expected to {'beat' if r['forecast'] >= 0 else 'trail'} the average metro by "
                f"{abs(r['forecast']):.1f} pts over the next 3 months")
        parts = sorted(used, key=lambda c: -abs(r[c] - 50))[:2]
        why = [REASON_TEXT[c][0 if r[c] >= 50 else 1] for c in parts if abs(r[c] - 50) >= 10]
        return lead + (": " + "; ".join(why) if why else "") + "."
    out["reason"] = out.apply(reason, axis=1)
    out["realized_vs_avg_pct"] = (100 * out["target"]).round(2)

    cols = (["metro", "date", "rank", "score", "score_chg", "stage", "prev_stage", "forecast",
             "price_vs_avg"] + used + ["price_chg_3m_pct", "realized_vs_avg_pct", "zhvi", "reason"])
    out = out[cols].sort_values(["date", "rank"])
    out.to_csv(RESULTS / "index_history.csv", index=False)

    # ---- Track record (walk-forward: each month's score used only data known then)
    hist = out.dropna(subset=["realized_vs_avg_pct"])
    hist = hist.assign(band=pd.cut(hist["score"], [0, 20, 40, 60, 80, 100],
                                   labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
                                   include_lowest=True))
    (hist.groupby("band", observed=True)["realized_vs_avg_pct"].mean().round(2).reset_index()
         .rename(columns={"realized_vs_avg_pct": "avg_next_3m_vs_avg_pct"})
         .to_csv(RESULTS / "score_bands.csv", index=False))
    (hist.groupby("stage")["realized_vs_avg_pct"].agg(["mean", "count"]).round(2).reset_index()
         .to_csv(RESULTS / "stage_performance.csv", index=False))

    top = hist[hist["rank"] <= TOP_K]
    by_month = top.groupby("date")["realized_vs_avg_pct"].agg(hit=lambda s: (s > 0).mean(),
                                                              excess="mean")
    driver_corr = {c: round(float(hist.groupby("date").apply(
        lambda x: x[c].corr(x["realized_vs_avg_pct"], method="spearman"), include_groups=False).mean()), 3)
        for c in used + ["score"]}
    track = {
        "model": best,
        "months_tested": int(by_month.shape[0]),
        "top5_hit_rate": round(float(by_month["hit"].mean()), 3),
        "top5_excess_3m_pct": round(float(by_month["excess"].mean()), 2),
        "top5_excess_per_year_pct": round(float(by_month["excess"].mean() * 4), 2),
        "driver_rank_corr": driver_corr,
        "components": used,
        "demo": bool(meta.get("demo")),
        "as_of": f"{out['date'].max():%b %Y}",
        "first_test_month": f"{by_month.index.min():%b %Y}",
    }
    (RESULTS / "track_record.json").write_text(json.dumps(track, indent=2))
    (RESULTS / "copy.json").write_text(json.dumps({"stages": STAGES, "audience": AUDIENCE}, indent=2))
    print(f"Housing Momentum Score built from model {best}, {track['as_of']}")
    print(f"Top-{TOP_K} hit rate {track['top5_hit_rate']:.0%}, "
          f"excess {track['top5_excess_per_year_pct']:+.1f} pts/yr; driver strength {driver_corr}")


if __name__ == "__main__":
    build()
