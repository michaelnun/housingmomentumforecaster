"""Step 9: how solid is each key finding?

Each finding is re-checked several ways, so one lucky slice of data can't carry it:
  - all metros, and Zillow's raw (unsmoothed) values as well as the main smoothed ones
  - the 22 original metros vs. the 23 added later. Every modeling choice (page views in the
    3-month score, search in the 12-month outlook) was made looking only at the original 22,
    so the 23 new metros are an untouched test of those choices.
  - early vs. recent test years
  - dropping each metro in turn (does any single metro carry the result?)

Writes results/stability.csv (every check) and results/stability.json (a verdict per finding).

    python step9_stability.py
"""
import json

import numpy as np
import pandas as pd

from config import ORIGINAL_METROS, PROCESSED, RAW, RESULTS
from step3_model import nw_pvalue
from step7_signal_tests import walk

MIN_METROS = 5  # a month needs at least this many metros to rank


def monthly_gain(d, a, b):
    """Per month: ranking accuracy of b minus a (rank correlation with what happened)."""
    def one(x):
        if len(x) < MIN_METROS:
            return np.nan
        ry = x["y"].rank()
        return x[b].rank().corr(ry) - x[a].rank().corr(ry)
    return d.groupby("date").apply(one, include_groups=False).dropna()


def check(d, a, b, h):
    g = monthly_gain(d, a, b)
    if len(g) < 12:
        return None
    return {"months": int(len(g)), "gain": round(float(g.mean()), 4),
            "better_share": round(float((g > 0).mean()), 3), "p": round(nw_pvalue(g, h - 1), 4)}


def slices(d, a, b, h):
    """Every check for one comparison (challenger b vs. baseline a)."""
    rows = []
    months = sorted(d["date"].unique())
    half = months[len(months) // 2] if months else None
    new = [m for m in d["metro"].unique() if m not in ORIGINAL_METROS]
    parts = {"All metros": d,
             "Original 22 metros": d[d["metro"].isin(ORIGINAL_METROS)],
             "New metros (untouched test)": d[d["metro"].isin(new)] if len(new) >= 10 else None,
             "Early years": d[d["date"] < half],
             "Recent years": d[d["date"] >= half]}
    for name, part in parts.items():
        r = check(part, a, b, h) if part is not None and len(part) else None
        if r:
            rows.append({"check": name, **r})
    drops = [check(d[d["metro"] != m], a, b, h) for m in sorted(d["metro"].unique())]
    drops = [x for x in drops if x]
    if drops:
        rows.append({"check": "Drop any one metro", "months": drops[0]["months"],
                     "gain": round(min(x["gain"] for x in drops), 4),  # worst case
                     "better_share": round(min(x["better_share"] for x in drops), 3),
                     "p": round(max(x["p"] for x in drops), 4),        # worst case
                     "share_significant": round(float(np.mean([x["p"] < 0.05 for x in drops])), 3)})
    return rows


def verdict(rows):
    """Solid: significant overall on every version tested, better in every slice, and still
    significant whichever metro is dropped. Holds with caveats: significant overall but weaker
    somewhere. Not supported: not significant overall."""
    overall = [r for r in rows if r["check"] == "All metros"]
    if not overall or any(r["p"] >= 0.05 or r["gain"] <= 0 for r in overall):
        return "Not supported"
    weak = [r for r in rows if r["gain"] <= 0]
    drops = [r for r in rows if r["check"] == "Drop any one metro"]
    if not weak and all(r.get("share_significant", 0) == 1 for r in drops):
        return "Solid"
    return "Holds, with caveats"


def raw_targets(panel):
    z = pd.read_csv(RAW / "zillow_raw_clean.csv", parse_dates=["date"]).sort_values(["metro", "date"])
    lz, g = np.log(z["zhvi"]), z.groupby("metro")["zhvi"]
    for h in (3, 12):
        past = lz - np.log(g.shift(h))
        z[f"rel_raw_chg_{h}m"] = past - past.groupby(z["date"]).transform("mean")
        fwd = np.log(g.shift(-h)) - lz
        z[f"y_raw_{h}"] = fwd - fwd.groupby(z["date"]).transform("mean")
    keep = ["metro", "date", "rel_raw_chg_3m", "rel_raw_chg_12m", "y_raw_3", "y_raw_12"]
    return panel.merge(z[keep], on=["metro", "date"], how="inner")


def pair(df, cols_a, cols_b, target, h, need):
    sub = df.dropna(subset=need)
    a, b = walk(sub, cols_a, target, h), walk(sub, cols_b, target, h)
    if a is None or b is None:
        return None
    return a.merge(b, on=["date", "metro", "y"], suffixes=("_a", "_b")).rename(
        columns={"p_a": "a", "p_b": "b"})


def main():
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    fs = pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"]
    F = {k: v.split(",") for k, v in fs.items()}
    A = F["A"]
    nat = [c for c in F.get("B", []) if "outoftown" in c]
    pv, rent = F.get("pageviews", []), F.get("rent", [])
    p3 = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"]).rename(columns={"target": "y"})
    f12 = RESULTS / "predictions_12m.csv"
    p12 = pd.read_csv(f12, parse_dates=["date"]).rename(columns={"target_12": "y"}) if f12.exists() else None
    has_raw = (RAW / "zillow_raw_clean.csv").exists()
    df = raw_targets(panel) if has_raw else None

    claims = {}

    # 1. The 3-month score beats "the recent trend continues"
    rows = [dict(r, version="Main values") for r in slices(p3.dropna(subset=["y", "Naive", "A"]), "Naive", "A", 3)]
    if has_raw:
        w = walk(df.dropna(subset=A + ["rel_raw_chg_3m"]), A + ["rel_raw_chg_3m"], "y_raw_3", 3)
        if w is not None:
            w = w.merge(df[["date", "metro", "rel_raw_chg_3m"]], on=["date", "metro"]).rename(
                columns={"p": "b", "rel_raw_chg_3m": "a"})
            rows += [dict(r, version="Raw values") for r in slices(w, "a", "b", 3)]
    claims["The 3-month score beats 'the recent trend continues'"] = rows

    # 2. Realtor.com page views improve the 3-month score (judged on raw values, where it was found)
    if pv:
        rows = []
        if "P" in p3:
            rows += [dict(r, version="Main values (info only)") for r in
                     slices(p3.dropna(subset=["y", "A", "P"]), "A", "P", 3)]
        if has_raw:
            base = A + ["rel_raw_chg_3m"]
            m = pair(df, base, base + pv, "y_raw_3", 3, base + pv)
            if m is not None:
                rows += [dict(r, version="Raw values") for r in slices(m, "a", "b", 3)]
        claims["Realtor.com page views improve the 3-month score"] = rows

    if p12 is not None:
        # 3. The 12-month outlook beats "the recent trend continues"
        prod = next(c[:-5] for c in p12.columns if c.endswith("_live"))
        rows = [dict(r, version="Main values") for r in
                slices(p12.dropna(subset=["y", "Naive12", prod]), "Naive12", prod, 12)]
        if has_raw and (pv or rent):
            full = A + ["rel_raw_chg_3m"] + (nat if prod == "Y12" else []) + pv + rent
            sub = df.dropna(subset=full)
            w = walk(sub, full, "y_raw_12", 12)
            if w is not None:
                w = w.merge(sub[["date", "metro", "rel_raw_chg_12m"]], on=["date", "metro"]).rename(
                    columns={"p": "b", "rel_raw_chg_12m": "a"})
                rows += [dict(r, version="Raw values") for r in slices(w, "a", "b", 12)]
        claims["The 12-month outlook beats 'the recent trend continues'"] = rows

        # 4. Nationwide Google search improves the 12-month outlook
        if "Y12" in p12 and "Y12_nosearch" in p12:
            rows = [dict(r, version="Main values") for r in
                    slices(p12.dropna(subset=["y", "Y12", "Y12_nosearch"]), "Y12_nosearch", "Y12", 12)]
            if has_raw:
                base = A + ["rel_raw_chg_3m"] + pv + rent
                m = pair(df, base, base + nat, "y_raw_12", 12, base + nat)
                if m is not None:
                    rows += [dict(r, version="Raw values") for r in slices(m, "a", "b", 12)]
            claims["Nationwide Google search improves the 12-month outlook"] = rows

    table, verdicts = [], {}
    for claim, rows in claims.items():
        judged = [r for r in rows if "info only" not in r["version"]]
        verdicts[claim] = verdict(judged)
        for r in rows:
            table.append({"finding": claim, **r})
        print(f"{verdicts[claim]:22s} {claim}")
    pd.DataFrame(table).to_csv(RESULTS / "stability.csv", index=False)
    (RESULTS / "stability.json").write_text(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
