"""Step 4: the Housing Momentum Score (0-100) for every metro and month, plus
everything the dashboard shows: stages, drivers, reasons, alerts and the track record.

The score is the model's 3-month forecast (from step 3), turned into a 0-100 rank
across metros that month. The drivers show WHY a metro scores where it does.

    python step4_index.py
"""
import json

import numpy as np
import pandas as pd

from config import PROCESSED, RAW, RESULTS, SCORE_MODEL

TOP_K = 5

# Drivers shown on the dashboard. Each = a metro's percentile vs. other metros that month.
# Sign: +1 means a higher raw value is a stronger market.
DRIVERS = {
    "Price momentum": [("rel_price_chg_3m", +1), ("rel_price_chg_12m", +1)],
    "Market tightness": [("rel_median_dom_yoy", -1), ("rel_avg_sale_to_list_yoy", +1),
                         ("rel_inventory_yoy", -1), ("rel_months_of_supply_yoy", -1),
                         ("rel_price_drops_yoy", -1), ("rel_sold_above_list_yoy", +1)],
    "Buyer demand": [("rel_search_index", +1)],
    "Nationwide interest": [("rel_outoftown_index", +1)],
    "Online listing views": [("rel_pageviews_chg3", +1), ("rel_pageviews_yoy", +1)],
}
# "Local jobs" (BLS) was dropped as a driver: it had no link to what prices did next (-0.02).
FALLBACK = {}

REASON_TEXT = {
    "Price momentum": ("prices have been rising faster than in other metros",
                       "prices have been lagging other metros"),
    "Market tightness": ("homes are selling faster with less inventory",
                         "homes are sitting longer and inventory is building"),
    "Buyer demand": ("local home searches are up more than elsewhere",
                     "local home searches are down more than elsewhere"),
    "Nationwide interest": ("more people across the U.S. are searching for homes here",
                           "nationwide interest in homes here is fading"),
    "Online listing views": ("more people are viewing home listings here online",
                             "fewer people are viewing home listings here online"),
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


# Year-ahead (12-month outlook) guidance, by outlook band
AUDIENCE_12 = {
    "Real estate": {
        "Strong": "Good market to grow in over the next year: worth adding agents or marketing.",
        "Average": "Expect a typical year relative to other big metros.",
        "Weak": "Plan for a slower year here; avoid big new commitments.",
    },
    "Home improvement retail": {
        "Strong": "Plan for more move-in demand over the next year (staffing, stock, new locations).",
        "Average": "Expect typical demand relative to other big metros.",
        "Weak": "Expect fewer moves over the year; lean on repair, remodel and pro customers.",
    },
    "Investors": {
        "Strong": "Year-ahead outlook favors this market vs. other big metros.",
        "Average": "No clear edge vs. other big metros over the year.",
        "Weak": "Year-ahead outlook is weak: be cautious adding exposure here.",
    },
}


def band(score):
    return np.select([score >= 60, score < 40], ["Strong", "Weak"], "Average")


def combined_note(s3, s12):
    """One plain line comparing the 3-month score with the 12-month outlook."""
    if pd.isna(s3) or pd.isna(s12):
        return ""
    now = "strong" if s3 >= 60 else "soft" if s3 < 40 else "about average"
    year = "strong" if s12 >= 60 else "weak" if s12 < 40 else "about average"
    if now == "strong" and year == "strong":
        return "Strong now and over the year ahead."
    if now == "soft" and year == "weak":
        return "Weak now and over the year ahead."
    if now == "about average" and year == "about average":
        return "Close to the typical big metro, now and over the year ahead."
    if now == "strong" and year == "weak":
        return "Strong for now, but expected to cool over the year ahead."
    if now == "soft" and year == "strong":
        return "Soft right now, but the year-ahead outlook is strong."
    return f"{now.capitalize()} over the next 3 months; {year} over the year ahead."


def market_now(out):
    """Plain market numbers for each metro and month (no forecasting): what agents quote."""
    m = pd.read_csv(RAW / "market_clean.csv", parse_dates=["date"]).sort_values(["metro", "date"])
    num = [c for c in m.columns if c not in ("metro", "date")]
    m[num] = m.groupby("metro")[num].ffill(limit=2)
    g = m.groupby("metro")
    add = pd.DataFrame({"metro": m["metro"], "date": m["date"]})
    if "median_dom" in m:
        add["days_to_pending"] = m["median_dom"].round(0)
        add["days_to_pending_last_year"] = g["median_dom"].shift(12).round(0)
    if "avg_sale_to_list" in m:
        add["sale_to_list_pct"] = (100 * m["avg_sale_to_list"]).round(1)
    if "price_drops" in m:
        add["price_cut_share_pct"] = (100 * m["price_drops"]).round(0)
    if "sold_above_list" in m:
        add["sold_above_list_pct"] = (100 * m["sold_above_list"]).round(0)
    if "inventory" in m:
        add["for_sale_vs_last_year_pct"] = (100 * (m["inventory"] / g["inventory"].shift(12) - 1)).round(0)
    out = out.merge(add, on=["metro", "date"], how="left")
    rent_f = RAW / "rent_clean.csv"
    if rent_f.exists():
        r = pd.read_csv(rent_f, parse_dates=["date"])
        out = out.merge(r.rename(columns={"rent": "typical_rent"}), on=["metro", "date"], how="left")
        out["typical_rent"] = out["typical_rent"].round(-1)
    zz = out.sort_values(["metro", "date"])
    out["home_value_yoy_pct"] = (100 * (zz["zhvi"] / zz.groupby("metro")["zhvi"].shift(12) - 1)).round(1)
    # Rough buyer's/seller's market label from the two numbers buyers feel most
    if {"sale_to_list_pct", "price_cut_share_pct"} <= set(out):
        stl, cut = out["sale_to_list_pct"], out["price_cut_share_pct"]
        above = out.get("sold_above_list_pct", pd.Series(np.nan, index=out.index))
        dtp = out.get("days_to_pending", pd.Series(np.nan, index=out.index))
        slow = (cut >= 30) | (dtp >= 60)
        # Main rule uses sale price vs. asking. If Zillow stopped publishing that for a metro,
        # fall back to the share of homes sold above asking.
        seller = (stl >= 100) | (stl.isna() & (above >= 40))
        buyer = ((stl < 98) | (stl.isna() & (above <= 15))) & slow
        out["market_type"] = np.select([seller, buyer], ["Seller's market", "Buyer's market"], "Balanced")
        out.loc[(stl.isna() & above.isna()) | cut.isna(), "market_type"] = None
    return out


def calibration(df, score_col, real_col, horizon):
    d = df.dropna(subset=[score_col, real_col])
    d = d.assign(band=pd.cut(d[score_col], [0, 20, 40, 60, 80, 101], right=False,  # 80 -> 80-100
                             labels=["0-20", "20-40", "40-60", "60-80", "80-100"]))
    t = d.groupby("band", observed=True)[real_col].agg(
        beat_avg_share=lambda x: (x > 0).mean(), avg_vs_avg_pct="mean", n="size").reset_index()
    return t.assign(horizon_months=horizon)


def log_predictions(out):
    """Freeze this month's published scores. Rows are only ever added, never rewritten, so the
    live track record can't be quietly improved later by re-running the model."""
    f = RESULTS / "prediction_log.csv"
    newest = out["date"].max()
    cols = [c for c in ["metro", "score", "stage", "forecast", "outlook_12m", "forecast_12m"] if c in out]
    new = out.loc[out["date"] == newest, cols].assign(
        data_month=f"{newest:%Y-%m}", published=f"{pd.Timestamp.now():%Y-%m-%d}")
    log = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=new.columns)
    if f"{newest:%Y-%m}" in set(log["data_month"].astype(str)):
        return log
    log = pd.concat([log, new[["data_month", "published"] + cols]], ignore_index=True)
    log.to_csv(f, index=False)
    print(f"Logged {len(new)} published scores for {newest:%b %Y} in prediction_log.csv")
    return log


def live_record(log, out):
    """How the frozen, published scores did once the outcomes came in."""
    if log is None or log.empty:
        return {}
    lg = log.assign(date=pd.to_datetime(log["data_month"] + "-01"))
    res = {"first_published": str(log["published"].min()), "months_logged": int(lg["date"].nunique())}
    known = lg.merge(out[["metro", "date", "realized_vs_avg_pct", "realized_12m_pct"]
                         if "realized_12m_pct" in out else ["metro", "date", "realized_vs_avg_pct"]],
                     on=["metro", "date"], how="left")
    for col, real, k in [("score", "realized_vs_avg_pct", 3), ("outlook_12m", "realized_12m_pct", 12)]:
        if col not in known or real not in known:
            continue
        d = known.dropna(subset=[col, real])
        res[f"first_results_{k}m"] = f"{lg['date'].min() + pd.DateOffset(months=k):%b %Y}"
        if d.empty:
            res[f"months_scored_{k}m"] = 0
            continue
        g = d.groupby("date")
        res[f"months_scored_{k}m"] = int(g.ngroups)
        res[f"rank_corr_{k}m"] = round(float(g.apply(lambda x: x[col].corr(x[real], method="spearman"),
                                                     include_groups=False).mean()), 3)
        res[f"top5_beat_{k}m"] = round(float(g.apply(lambda x: (x.nlargest(TOP_K, col)[real] > 0).mean(),
                                                     include_groups=False).mean()), 3)
    return res


def build():
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    meta = json.loads((PROCESSED / "meta.json").read_text()) if (PROCESSED / "meta.json").exists() else {}
    card = pd.read_csv(RESULTS / "scorecard.csv", index_col=0)
    preds = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"])
    best = (SCORE_MODEL if SCORE_MODEL in preds and preds[SCORE_MODEL].notna().any()
            else "A")
    # One model for every metro. A metro missing an extra signal (e.g. page views) is scored by
    # the same model with that signal set to typical, and flagged, instead of switching models.
    if f"{best}_live" in preds:
        preds["data_gaps"] = np.where(preds[best].isna() & preds[f"{best}_live"].notna(),
                                      "page views missing (treated as typical)", "")
        preds[best] = preds[f"{best}_live"]
    else:
        preds["data_gaps"] = ""
    preds = preds[["metro", "date", best, "data_gaps"]].dropna(subset=[best])

    out = panel[["metro", "date", "zhvi", "target"]].copy()
    out["price_chg_3m_pct"] = (100 * panel["price_chg_3m"]).round(2)
    out["price_vs_avg"] = (100 * panel["rel_price_chg_3m"]).round(2)
    out["price_vs_avg_12m"] = (100 * panel["rel_price_chg_12m"]).round(2)
    if "rent_yoy" in panel:  # for investors: rent growth over the past year
        out["rent_yoy_pct"] = (100 * panel["rent_yoy"]).round(1)

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

    out = out.merge(preds.rename(columns={best: "fc3"}), on=["metro", "date"], how="inner")
    out["score"] = (out.groupby("date")["fc3"].rank(pct=True) * 100).round(0)  # rank unrounded values
    out["forecast"] = (100 * out["fc3"]).round(2)  # % vs. average metro, next 3 months

    # Stage = direction of the last 3 months x direction of the forecast. Near zero a direction
    # is just noise, so inside a small dead zone (+-0.1%) a metro keeps last month's direction.
    out = out.sort_values(["metro", "date"])

    def sticky_sign(x, dead=0.1):
        res, prev = [], None
        for v in x:
            if pd.isna(v):
                res.append(prev)
                continue
            if v > dead or prev is None and v >= 0:
                prev = True
            elif v < -dead or prev is None:
                prev = False
            res.append(prev)
        return pd.Series(res, index=x.index)
    momentum_up = out.groupby("metro", group_keys=False)["price_vs_avg"].apply(sticky_sign).astype(bool)
    outlook_up = out.groupby("metro", group_keys=False)["forecast"].apply(sticky_sign).astype(bool)
    out["stage"] = np.select([momentum_up & outlook_up, ~momentum_up & outlook_up,
                              momentum_up & ~outlook_up], ["Rising", "Turning up", "Turning down"],
                             "Falling")

    # ---- 12-month outlook (from step 3), same 0-100 idea, 12 months ahead
    f12 = RESULTS / "predictions_12m.csv"
    if f12.exists():
        p12 = pd.read_csv(f12, parse_dates=["date"])
        prod = next(c[:-5] for c in p12.columns if c.endswith("_live"))
        gaps12 = np.where(p12[prod].isna() & p12[f"{prod}_live"].notna(),
                          "some year-ahead signals missing (treated as typical)", "")
        p12 = p12.assign(fc12=p12[f"{prod}_live"], gaps12=gaps12,
                         realized_12m_pct=(100 * p12["target_12"]).round(2))[
            ["metro", "date", "fc12", "gaps12", "realized_12m_pct"]].dropna(subset=["fc12"])
        out = out.merge(p12, on=["metro", "date"], how="left")
        out["outlook_12m"] = (out.groupby("date")["fc12"].rank(pct=True) * 100).round(0)
        out["forecast_12m"] = (100 * out["fc12"]).round(2)
        out["data_gaps"] = [", ".join(x for x in (a, b) if isinstance(x, str) and x)
                            for a, b in zip(out["data_gaps"], out["gaps12"])]
        out["outlook_band"] = np.where(out["outlook_12m"].notna(), band(out["outlook_12m"].fillna(50)), None)
        out["note"] = [combined_note(a, b) for a, b in zip(out["score"], out["outlook_12m"])]

    out = out.sort_values(["metro", "date"])
    # change vs. the previous calendar month (not just the previous row, in case a month is missing)
    prev = out[["metro", "date", "score", "stage"]].assign(date=lambda d: d["date"] + pd.DateOffset(months=1))
    out = out.merge(prev.rename(columns={"score": "prev_score", "stage": "prev_stage"}),
                    on=["metro", "date"], how="left")
    out["score_chg"] = out["score"] - out["prev_score"]
    out["rank"] = out.groupby("date")["score"].rank(ascending=False, method="first").astype(int)

    # Reasons favor drivers that actually predict the next 3 months (measured over the test period),
    # so a strong-but-useless signal is never given as the main reason.
    known = out.dropna(subset=["target"])
    weight = {c: max(0.02, float(known.groupby("date").apply(
        lambda x: x[c].corr(x["target"], method="spearman"), include_groups=False).mean()))
        for c in used}

    def reason(r):
        f = r["forecast"]
        size = ("about the same as" if abs(f) < 0.05 else
                f"about {abs(f):.1f}% {'more' if f > 0 else 'less'} than")
        lead = f"Home values expected to grow {size} the typical big metro over the next 3 months"
        up = f >= 0
        agree = [c for c in used if (r[c] >= 50) == up and abs(r[c] - 50) >= 10]
        against = [c for c in used if (r[c] >= 50) != up and abs(r[c] - 50) >= 30]
        why = [REASON_TEXT[c][0 if up else 1]
               for c in sorted(agree, key=lambda c: -abs(r[c] - 50) * weight[c])[:2]]
        text = lead + (": " + "; ".join(why) if why else "")
        if abs(f) < 0.05:  # no clear direction, so no "because"
            return lead + "."
        if against:
            c = max(against, key=lambda c: abs(r[c] - 50) * weight[c])
            text += f", although {REASON_TEXT[c][0 if r[c] >= 50 else 1]}"
        return text + "."
    out["reason"] = out.apply(reason, axis=1)
    out["realized_vs_avg_pct"] = (100 * out["target"]).round(2)

    out = market_now(out)
    extra = ["outlook_12m", "outlook_band", "forecast_12m", "realized_12m_pct", "note",
             "home_value_yoy_pct", "rent_yoy_pct", "typical_rent", "days_to_pending",
             "days_to_pending_last_year", "sale_to_list_pct", "sold_above_list_pct",
             "price_cut_share_pct", "for_sale_vs_last_year_pct", "market_type", "data_gaps"]
    cols = (["metro", "date", "rank", "score", "score_chg", "stage", "prev_stage", "forecast",
             "price_vs_avg", "price_vs_avg_12m"] + used + ["price_chg_3m_pct", "realized_vs_avg_pct", "zhvi", "reason"]
            + [c for c in extra if c in out])
    out = out[cols].sort_values(["date", "rank"])
    out.to_csv(RESULTS / "index_history.csv", index=False)
    log = log_predictions(out)

    # ---- Track record (walk-forward: each month's score used only data known then)
    hist = out.dropna(subset=["realized_vs_avg_pct"])
    hist = hist.assign(band=pd.cut(hist["score"], [0, 20, 40, 60, 80, 101], right=False,
                                   labels=["0-20", "20-40", "40-60", "60-80", "80-100"]))
    (hist.groupby("band", observed=True)["realized_vs_avg_pct"].mean().round(2).reset_index()
         .rename(columns={"realized_vs_avg_pct": "avg_next_3m_vs_avg_pct"})
         .to_csv(RESULTS / "score_bands.csv", index=False))
    (hist.groupby("stage")["realized_vs_avg_pct"].agg(["mean", "count"]).round(2).reset_index()
         .to_csv(RESULTS / "stage_performance.csv", index=False))

    cal = [calibration(out, "score", "realized_vs_avg_pct", 3)]
    if "outlook_12m" in out:
        cal.append(calibration(out, "outlook_12m", "realized_12m_pct", 12))
    pd.concat(cal).round(3).to_csv(RESULTS / "calibration.csv", index=False)

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
        "updated": f"{pd.Timestamp.now():%b %d, %Y}",
        "first_test_month": f"{by_month.index.min():%b %Y}",
        "metros": int(out.loc[out["date"] == out["date"].max(), "metro"].nunique()),
        "rank_corr": round(float(driver_corr["score"]), 3),
    }
    t12, c12 = RESULTS / "tests_12m.json", RESULTS / "scorecard_12m.csv"
    if t12.exists() and c12.exists():
        t = json.loads(t12.read_text())
        c = pd.read_csv(c12, index_col=0)
        mdl = t["model"]
        track["outlook"] = {**t, "rank_corr": round(float(c.loc[mdl, "rank_corr"]), 3),
                            "top5_beat_avg": round(float(c.loc[mdl, "top5_beat_avg"]), 3),
                            "bottom5_trail_avg": round(float(c.loc[mdl, "bottom5_trail_avg"]), 3),
                            "top5_excess_pct": round(float(c.loc[mdl, "top5_excess_pct"]), 2),
                            "trend_rank_corr": round(float(c.loc["Naive12", "rank_corr"]), 3),
                            "housing_only_rank_corr": round(float(c.loc["A12", "rank_corr"]), 3)}
    track["live"] = live_record(log, out)
    (RESULTS / "track_record.json").write_text(json.dumps(track, indent=2))
    (RESULTS / "copy.json").write_text(json.dumps({"stages": STAGES, "audience": AUDIENCE,
                                                   "audience_12m": AUDIENCE_12}, indent=2))
    print(f"Housing Momentum Score built from model {best}, {track['as_of']}")
    print(f"Top-{TOP_K} hit rate {track['top5_hit_rate']:.0%}, "
          f"excess {track['top5_excess_per_year_pct']:+.1f} pts/yr; driver strength {driver_corr}")


if __name__ == "__main__":
    build()
