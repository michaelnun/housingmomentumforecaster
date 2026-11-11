"""Step 2: merge everything into one row per metro-month, with no future data in any feature.

    python step2_build.py
    python step2_build.py --raw data/raw_demo   # run on simulated data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW, PROCESSED, HORIZON, BUYER_TERMS, HOME_IMPROVEMENT_TERMS, SELLER_TERMS


def slug(term):
    return term.replace(" ", "_")


def by_metro(df, col, fn):
    return df.groupby("metro")[col].transform(fn)


def nationwide(raw, key):
    """A nationwide search series, minus any city flagged in step 1 as mostly zeros (too
    little search volume to trust). Those cities are then treated as missing that signal."""
    f = raw / f"{key}_clean.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["date"])[["date", "metro", key]]
    q = raw / "search_quality.csv"
    if q.exists():
        qq = pd.read_csv(q)
        bad = set(qq.loc[(qq["search"] == key) & (qq["zero_share"] > 0.3), "metro"])
        if bad:
            print(f"Not using '{key}' search for (too many zero months): {', '.join(sorted(bad))}")
            d.loc[d["metro"].isin(bad), key] = np.nan
    return d


def build(raw: Path):
    z = pd.read_csv(raw / "zillow_clean.csv", parse_dates=["date"])
    market = raw / "market_clean.csv"
    rf = pd.read_csv(market if market.exists() else raw / "redfin_clean.csv", parse_dates=["date"])
    # Fill short gaps in each metro's market series (Zillow has a few missing months) with the
    # last known value. Only looks backward, so no month ever uses a later month's numbers.
    rf = rf.sort_values(["metro", "date"])
    num = [c for c in rf.columns if c not in ("metro", "date")]
    rf[num] = rf.groupby("metro")[num].ffill(limit=6)
    fr = pd.read_csv(raw / "fred_clean.csv", parse_dates=["date"])
    tr = pd.read_csv(raw / "trends_clean.csv", parse_dates=["date"])
    df = (z.merge(rf, on=["metro", "date"], how="left")
           .merge(fr, on=["metro", "date"], how="left")
           .merge(tr, on=["metro", "date"], how="left"))
    ot = nationwide(raw, "outoftown")  # merged on its own, so a metro missing local search
    if ot is not None:                 # data still keeps its nationwide search
        df = df.merge(ot, on=["metro", "date"], how="left")
    df = df.sort_values(["metro", "date"]).reset_index(drop=True)
    # Some sources publish a month after Zillow's home values (sale-to-list, unemployment...).
    # Carry their last value forward up to 2 months so the newest month still gets a score.
    late = [c for c in num + ["mortgage_rate", "unemployment"] if c in df]
    df[late] = df.groupby("metro")[late].ffill(limit=2)
    g = df.groupby("metro")
    lp = np.log(df["zhvi"])

    # ---- Target: price change over the NEXT 3 months, minus the average metro's change
    df["fwd_ret"] = g["zhvi"].shift(-HORIZON).pipe(np.log) - lp
    df["target"] = df["fwd_ret"] - df.groupby("date")["fwd_ret"].transform("mean")
    # ---- Year-ahead target for the 12-month outlook
    fwd12 = g["zhvi"].shift(-12).pipe(np.log) - lp
    df["target_12"] = fwd12 - fwd12.groupby(df["date"]).transform("mean")

    # ---- Traditional features (Model A): all use data up to this month only
    for k in (1, 3, 12):
        df[f"price_chg_{k}m"] = lp - np.log(g["zhvi"].shift(k))
    for c in ("median_dom", "avg_sale_to_list", "months_of_supply", "price_drops",
              "sold_above_list"):
        if c in df:
            df[f"{c}_yoy"] = df[c] - g[c].shift(12)
    if "inventory" in df:
        df["inventory_yoy"] = np.log(df["inventory"]) - np.log(g["inventory"].shift(12))
    df["mortgage_chg_3m"] = df["mortgage_rate"] - g["mortgage_rate"].shift(3)
    df["unemployment_chg_12m"] = df["unemployment"] - g["unemployment"].shift(12)

    # ---- Search features. Trends values are 0-100 and only comparable within a metro,
    # so use year-over-year change (also removes seasonality: spring vs spring).
    def search_block(terms, prefix):
        cols = []
        if not terms:
            return cols
        for t in terms:
            if t not in df:
                continue
            x = np.log1p(df[t])
            yoy = x - g[t].shift(12).pipe(np.log1p)
            df[f"{prefix}_{slug(t)}_yoy"] = yoy.groupby(df["metro"]).transform(
                lambda s: s.rolling(3, min_periods=1).mean())  # smooth sampling noise
            cols.append(f"{prefix}_{slug(t)}_yoy")
        # one combined index: average of the terms' year-over-year changes (all in the same
        # log units, so no rescaling with full-sample statistics is needed)
        df[f"{prefix}_index"] = df[cols].mean(axis=1)
        df[f"{prefix}_index_chg3"] = df[f"{prefix}_index"] - g[f"{prefix}_index"].shift(3)
        return cols + [f"{prefix}_index", f"{prefix}_index_chg3"]

    buyer_cols = search_block(BUYER_TERMS, "search")
    ot_cols = search_block(["outoftown"], "outoftown") if "outoftown" in df else []
    hi_cols = search_block(HOME_IMPROVEMENT_TERMS, "homeimp")

    trad_cols = [c for c in df.columns if c.startswith("price_chg_") or c.endswith("_yoy")
                 and not c.startswith(("search_", "homeimp_", "outoftown_"))]
    trad_cols += ["median_dom", "avg_sale_to_list", "months_of_supply",
                  "mortgage_rate", "mortgage_chg_3m", "unemployment", "unemployment_chg_12m"]
    trad_cols = [c for c in dict.fromkeys(trad_cols) if c in df]

    # ---- Optional extra signals, tested separately in step 7 (never drop rows for them)
    extra = {}
    emp_file = raw / "employment_clean.csv"
    if emp_file.exists():
        emp = pd.read_csv(emp_file, parse_dates=["date"])
        df = df.merge(emp, on=["metro", "date"], how="left")
        ec = [c for c in ("jobs", "metro_unemp") if c in df]
        df[ec] = df.groupby("metro")[ec].ffill(limit=2)
        g = df.groupby("metro")
        cols = []
        if "jobs" in df and df["jobs"].notna().any():
            df["jobs_yoy"] = np.log(df["jobs"]) - np.log(g["jobs"].shift(12))
            df["jobs_chg3"] = np.log(df["jobs"]) - np.log(g["jobs"].shift(3))
            cols += ["jobs_yoy", "jobs_chg3"]
        if "metro_unemp" in df and df["metro_unemp"].notna().any():
            df["metro_unemp_chg_12m"] = df["metro_unemp"] - g["metro_unemp"].shift(12)
            cols += ["metro_unemp", "metro_unemp_chg_12m"]
        extra["jobs"] = cols
    hi_file = raw / "home_improvement_clean.csv"
    if hi_file.exists():
        hi = pd.read_csv(hi_file, parse_dates=["date"]).rename(columns={"hi_sales_yoy": "hi_sales_pct"})
        df = df.merge(hi, on=["metro", "date"], how="left")
        g = df.groupby("metro")
        df["hi_sales_avg3"] = g["hi_sales_pct"].transform(lambda s: s.rolling(3, min_periods=2).mean())
        df["hi_sales_chg3"] = df["hi_sales_avg3"] - g["hi_sales_avg3"].shift(3)
        extra["spending"] = ["hi_sales_avg3", "hi_sales_chg3"]

    rent_file = raw / "rent_clean.csv"  # Zillow rent index
    if rent_file.exists():
        df = df.merge(pd.read_csv(rent_file, parse_dates=["date"]), on=["metro", "date"], how="left")
        df["rent"] = df.groupby("metro")["rent"].ffill(limit=2)
        g = df.groupby("metro")
        lr = np.log(df["rent"])
        df["rent_yoy"] = lr - np.log(g["rent"].shift(12))
        df["rent_chg3"] = lr - np.log(g["rent"].shift(3))
        df["rent_vs_price"] = df["rent_yoy"] - df["price_chg_12m"]  # rents outpacing prices?
        extra["rent"] = ["rent_yoy", "rent_chg3", "rent_vs_price"]

    pv_file = raw / "pageviews_clean.csv"  # Realtor.com listing views vs. the U.S.
    if pv_file.exists():
        df = df.merge(pd.read_csv(pv_file, parse_dates=["date"]), on=["metro", "date"], how="left")
        g = df.groupby("metro")
        # Realtor.com can publish a month after Zillow: carry the last value forward up to 2 months
        df["pageviews"] = g["pageviews"].ffill(limit=2)
        lv = np.log(df["pageviews"])
        df["pageviews_level"] = lv
        df["pageviews_chg3"] = lv - np.log(g["pageviews"].shift(3))
        df["pageviews_yoy"] = lv - np.log(g["pageviews"].shift(12))
        extra["pageviews"] = ["pageviews_level", "pageviews_chg3", "pageviews_yoy"]

    for key, fname, terms in (("seller", "seller_clean.csv", SELLER_TERMS),
                              ("moving", "movingto_clean.csv", ["movingto"])):
        f = raw / fname
        if not f.exists():
            continue
        d = nationwide(raw, "movingto") if key == "moving" else pd.read_csv(f, parse_dates=["date"])
        df = df.merge(d, on=["metro", "date"], how="left")
        g = df.groupby("metro")
        extra[key] = search_block([t for t in terms if t in df], key)

    for c in sum(extra.values(), []):
        df[f"rel_{c}"] = df[c] - df.groupby("date")[c].transform("mean")

    # The target is "vs the average metro", so compare each metro's features to the
    # average metro that month too (national series like mortgage rate drop out here).
    for c in trad_cols + buyer_cols + ot_cols + hi_cols:
        df[f"rel_{c}"] = df[c] - df.groupby("date")[c].transform("mean")

    feature_sets = {
        "A": [f"rel_{c}" for c in trad_cols if not c.startswith(("mortgage",))]
             + ["mortgage_rate", "mortgage_chg_3m"],
        "B": [], "C": [],
    }
    feature_sets["B"] = feature_sets["A"] + [f"rel_{c}" for c in buyer_cols + ot_cols]
    feature_sets["C"] = feature_sets["B"] + [f"rel_{c}" for c in hi_cols]
    extra_sets = {k: [f"rel_{c}" for c in v] for k, v in extra.items() if v}

    keep = ["metro", "date", "zhvi", "fwd_ret", "target", "target_12", "price_chg_3m"] + sorted(
        set(sum(feature_sets.values(), [])) | {"search_index", "homeimp_index"}
        | ({"outoftown_index"} if ot_cols else set()) | set(sum(extra_sets.values(), []))
        | ({"rent_yoy"} if "rent" in extra_sets else set()))
    panel = df[keep].copy()
    # Need a full year of history for year-over-year features. Only the housing data (model A)
    # is required: a metro missing search or page-view data still gets a score, and each
    # model is tested on the rows where its own data exists.
    panel = panel.dropna(subset=feature_sets["A"], how="any")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "panel.csv", index=False)
    pd.Series({k: ",".join(v) for k, v in {**feature_sets, **extra_sets}.items()}).to_csv(
        PROCESSED / "feature_sets.csv", header=["features"])
    import json
    (PROCESSED / "meta.json").write_text(json.dumps({"demo": (raw / "DEMO").exists(),
                                                     "out_of_town": bool(ot_cols)}))
    print(f"Panel: {len(panel):,} rows, {panel['metro'].nunique()} metros, "
          f"{panel['date'].min():%b %Y} to {panel['date'].max():%b %Y}")
    print({k: len(v) for k, v in feature_sets.items()}, "features per model;",
          "extra signals:", {k: int(panel[v].notna().all(axis=1).sum()) for k, v in extra_sets.items()})
    print("Next: python step3_model.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    build(Path(ap.parse_args().raw))
