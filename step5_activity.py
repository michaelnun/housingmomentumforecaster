"""Step 5: does search predict market ACTIVITY (new listings) rather than prices?

Target: change in a metro's new listings over the next 3 months (year-over-year, to remove
seasons), vs. the average metro. Same walk-forward test as step 3:
  A = activity history + housing data,  B = A + buyer and out-of-town search.

    python step5_activity.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED, RAW, RESULTS, TEST_START, HORIZON
from step3_model import ridge, diebold_mariano

TOP_K = 5


def run_one(raw, metric, label):
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    market = pd.read_csv(raw / "market_clean.csv", parse_dates=["date"])
    if metric not in market or market[metric].notna().sum() == 0:
        print(f"No {metric} data; skipping.")
        return None
    nl = market[["metro", "date", metric]].rename(columns={metric: "v"}).sort_values(["metro", "date"])
    g = nl.groupby("metro")["v"]
    nl["nl_yoy"] = np.log(nl["v"].clip(lower=1)) - np.log(g.shift(12).clip(lower=1))
    nl["nl_yoy_chg3"] = nl["nl_yoy"] - nl.groupby("metro")["nl_yoy"].shift(3)
    nl["fwd"] = nl.groupby("metro")["nl_yoy"].shift(-HORIZON) - nl["nl_yoy"]
    df = panel.merge(nl[["metro", "date", "nl_yoy", "nl_yoy_chg3", "fwd"]],
                     on=["metro", "date"], how="inner")
    for c in ("nl_yoy", "nl_yoy_chg3", "fwd"):
        df[f"rel_{c}"] = df[c] - df.groupby("date")[c].transform("mean")
    df["target_act"] = df["rel_fwd"]

    fs = pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"]
    feats = {k: v.split(",") for k, v in fs.items()}
    search = [c for c in feats["B"] if c not in feats["A"]]
    A = ["rel_nl_yoy", "rel_nl_yoy_chg3"] + feats["A"]
    sets = {"Naive": None, "A": A, "B": A + search}
    df = df.dropna(subset=A + search)

    known = df.dropna(subset=["target_act"])
    rows = []
    for m in sorted(known.loc[known["date"] >= TEST_START, "date"].unique()):
        train = known[known["date"] <= pd.Timestamp(m) - pd.DateOffset(months=HORIZON)]
        test = known[known["date"] == m]
        if len(train) < 200:
            continue
        r = test[["metro", "date"]].copy()
        r["target"] = test["target_act"].values
        r["Naive"] = -test["rel_nl_yoy_chg3"].values  # recent swings tend to reverse
        for k in ("A", "B"):
            r[k] = ridge().fit(train[sets[k]], train["target_act"]).predict(test[sets[k]])
        rows.append(r)
    p = pd.concat(rows)

    def card(model):
        def m(x):
            top_p = set(x.nlargest(TOP_K, model)["metro"]); top_t = set(x.nlargest(TOP_K, "target")["metro"])
            return pd.Series({"rank_corr": stats.spearmanr(x[model], x["target"]).correlation,
                              "top5_overlap": len(top_p & top_t) / TOP_K,
                              "direction": (np.sign(x[model]) == np.sign(x["target"])).mean()})
        return p.groupby("date").apply(m, include_groups=False).mean().round(3)
    sc = pd.DataFrame({k: card(k) for k in ("Naive", "A", "B")}).T

    dm = diebold_mariano(p, "A", "B")

    # Lead time: does today's search change line up with new-listing changes k months later?
    lead = []
    for k in range(0, 13):
        fut = df.groupby("metro")["nl_yoy"].shift(-k) - df["nl_yoy"]
        fut = fut - fut.groupby(df["date"]).transform("mean")
        ok = fut.notna() & df["rel_search_index"].notna()
        sub = df[ok].assign(f=fut[ok])
        lead.append({"months_ahead": k,
                     "corr_buyer_search": round(float(sub.groupby("date").apply(
                         lambda x: x["rel_search_index"].corr(x["f"], method="spearman"),
                         include_groups=False).mean()), 3),
                     "corr_outoftown": round(float(sub.groupby("date").apply(
                         lambda x: x["rel_outoftown_index"].corr(x["f"], method="spearman"),
                         include_groups=False).mean()), 3) if "rel_outoftown_index" in sub else None})
    lead = pd.DataFrame(lead)
    print(f"\nACTIVITY TEST: {label} (next-3-month change vs. average metro)")
    print(sc.to_string())
    print("B vs A:", dm["verdict"], f"(p={dm['p_value_one_sided']:.3f})")
    return {"label": label, "scorecard": sc.to_dict(orient="index"),
            "B_vs_A": {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                       for k, v in dm.items()},
            "test_months": int(p["date"].nunique()), "lead_time": lead.to_dict(orient="records")}


def main(raw=RAW):
    out = {}
    for metric, label in (("sales_count", "Homes sold"), ("new_listings", "New listings")):
        r = run_one(raw, metric, label)
        if r:
            out[metric] = r
    (RESULTS / "activity_test.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    main(Path(ap.parse_args().raw))
