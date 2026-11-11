"""Step 8: robustness check on Zillow's UNSMOOTHED home values.

Our main target uses Zillow's smoothed series (roughly a 3-month average), which makes
short-term changes partly predictable from the past. Here the target is the change in the
raw series instead, and we re-ask: does the model still beat "trend continues", and does
nationwide search interest still help at 6 and 12 months?

    python step8_robustness.py --raw data/raw
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED, RAW, RESULTS
from step3_model import nw_pvalue
from step7_signal_tests import walk

HORIZONS = (3, 6, 12)


def monthly(m, col):
    """Ranking accuracy (rank correlation with what happened) for each month."""
    return m.groupby("date").apply(lambda x: x[col].rank().corr(x["y"].rank()), include_groups=False)


def main(raw):
    f = raw / "zillow_raw_clean.csv"
    if not f.exists():
        print("No unsmoothed Zillow series yet (run step 1); skipping.")
        return
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    fs = pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"]
    feats = {k: v.split(",") for k, v in fs.items()}
    z = pd.read_csv(f, parse_dates=["date"]).sort_values(["metro", "date"])
    lz = np.log(z["zhvi"])
    g = z.groupby("metro")["zhvi"]
    for h in HORIZONS:
        chg = lz - np.log(g.shift(h))  # past h months, vs. the average metro: "trend continues"
        z[f"rel_raw_chg_{h}m"] = chg - chg.groupby(z["date"]).transform("mean")
        fwd = np.log(g.shift(-h)) - lz
        z[f"y_{h}"] = fwd - fwd.groupby(z["date"]).transform("mean")
    zc = [f"rel_raw_chg_{h}m" for h in HORIZONS] + [f"y_{h}" for h in HORIZONS]
    df = panel.merge(z[["metro", "date"] + zc], on=["metro", "date"], how="inner")

    base = feats["A"] + ["rel_raw_chg_3m"]
    nat = [c for c in feats["B"] if "outoftown" in c]
    # other signals that helped on the smoothed data get the same stricter check
    others = {k: feats[k] for k in ("pageviews", "rent") if k in feats}
    df = df.dropna(subset=base)

    def compare(sub, cols_a, cols_b, t, h):
        """Rank accuracy without/with the extra columns on the same rows, and the p-value that
        the gain is real (monthly ranking accuracy, Newey-West for overlapping months)."""
        a_, b_ = walk(sub, cols_a, t, h), walk(sub, cols_b, t, h)
        if a_ is None or b_ is None:
            return None
        m_ = a_.merge(b_, on=["date", "metro", "y"], suffixes=("_a", "_b"))
        ra, rb = monthly(m_, "p_a"), monthly(m_, "p_b")
        return round(float(ra.mean()), 3), round(float(rb.mean()), 3), round(nw_pvalue(rb - ra, h - 1), 3), m_

    rows = []
    for h in HORIZONS:
        t = f"y_{h}"
        a = walk(df, base, t, h)
        if a is None:
            continue
        a = a.merge(df[["date", "metro", f"rel_raw_chg_{h}m"]], on=["date", "metro"])
        ra, rn = monthly(a, "p"), monthly(a.rename(columns={f"rel_raw_chg_{h}m": "naive"}), "naive")
        r = {"horizon_months": h, "months_tested": int(a["date"].nunique()),
             "naive": round(float(rn.mean()), 3), "housing_model": round(float(ra.mean()), 3),
             "p_model_vs_naive": round(nw_pvalue(ra - rn, h - 1), 3)}
        if nat:
            res = compare(df.dropna(subset=nat), base, base + nat, t, h)
            if res:
                r["nat_without"], r["with_nationwide_search"], r["p_nationwide_helps"] = res[:3]
        for k, cols in others.items():
            res = compare(df.dropna(subset=cols), base, base + cols, t, h)
            if res:
                r[f"{k}_without"], r[f"{k}_with"], r[f"p_{k}_helps"] = res[:3]
        if h == 12 and nat and others:  # the full 12-month outlook model, on raw values
            full = base + nat + sum(others.values(), [])
            sub = df.dropna(subset=full)
            o = walk(sub, full, t, h)
            if o is not None:
                o = o.merge(sub[["date", "metro", "rel_raw_chg_12m"]], on=["date", "metro"])
                ro = monthly(o, "p")
                rn2 = monthly(o.rename(columns={"rel_raw_chg_12m": "naive"}), "naive")
                r["outlook_model"] = round(float(ro.mean()), 3)
                r["outlook_naive"] = round(float(rn2.mean()), 3)
                r["p_outlook_vs_naive"] = round(nw_pvalue(ro - rn2, h - 1), 3)
        rows.append(r)
        print(r)
    pd.DataFrame(rows).to_csv(RESULTS / "robustness_unsmoothed.csv", index=False)
    (RESULTS / "robustness_unsmoothed.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    main(Path(ap.parse_args().raw))
