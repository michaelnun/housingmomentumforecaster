"""Step 10: making Google search useful.

1. Early-read test. Search is available right away; Zillow publishes each month's numbers
   3 to 6 weeks later. Does search for the month after the latest housing data improve the
   3-month score? The pass/fail rule was written before this test was run:
   docs/preregistration_nowcast.md. Writes results/nowcast.csv and results/nowcast.json.

2. Out-of-town interest for the dashboard: how much people across the U.S. search
   "[city] homes for sale", compared with a year earlier. Shown as context only; it is not
   used in any score. Writes results/search_interest.csv.

    python step10_search.py
"""
import json

import numpy as np
import pandas as pd

from config import METROS, ORIGINAL_METROS, OUTLOOK_USE_SEARCH, PROCESSED, RAW, RESULTS, SCORE_MODEL
from step9_stability import check, pair, raw_targets, slices

SUN_BELT = list(METROS)[:21]  # config lists the Sun Belt and Mountain West metros first


def next_month(df, cols, tag):
    """The values of `cols` one month later, as new columns on the same row."""
    s = df[["metro", "date"] + cols].copy()
    s["date"] = s["date"] - pd.DateOffset(months=1)
    return df.merge(s.rename(columns={c: f"{c}_{tag}" for c in cols}), on=["metro", "date"], how="left")


def rows_for(m, h, test, version):
    return [dict(r, test=test, version=version) for r in slices(m, "a", "b", h)]


def early_read(panel, F):
    A = F["A"]
    pv = F.get("pageviews", []) if SCORE_MODEL == "P" else []
    base = A + pv                                   # the 3-month score model
    local = [c for c in F["B"] if c not in A and not c.startswith("rel_outoftown")]
    nat = [c for c in F["B"] if c.startswith("rel_outoftown")]
    search = local + nat
    df = next_month(panel, search, "next")
    df = next_month(df, base, "wait")               # waiting for Zillow's next month (info only)
    nxt = lambda cols: [f"{c}_next" for c in cols]
    out = []

    # Decision tests (see docs/preregistration_nowcast.md)
    m_main = pair(df, base, base + nxt(search), "target", 3, base + nxt(search))
    if m_main is None:
        print("Not enough data for the early-read test.")
        return None
    out += rows_for(m_main, 3, "Early read: search one month ahead", "Main values")
    raw = raw_targets(df) if (RAW / "zillow_raw_clean.csv").exists() else None
    if raw is not None:
        b = base + ["rel_raw_chg_3m"]
        m_raw = pair(raw, b, b + nxt(search), "y_raw_3", 3, b + nxt(search))
        if m_raw is not None:
            out += rows_for(m_raw, 3, "Early read: search one month ahead", "Raw values")

    # Info only
    for name, cols in (("Local buyer searches only", local), ("Nationwide searches only", nat)):
        if cols:
            m = pair(df, base, base + nxt(cols), "target", 3, base + nxt(search))
            r = check(m, "a", "b", 3) if m is not None else None
            if r:
                out.append(dict(r, check="All metros", test=name, version="Main values"))
    for name, grp in (("Sun Belt and Mountain West", SUN_BELT),
                      ("Coastal, Midwest and Northeast", [x for x in METROS if x not in SUN_BELT])):
        r = check(m_main[m_main["metro"].isin(grp)], "a", "b", 3)
        if r:
            out.append(dict(r, check=name, test="Early read: search one month ahead", version="Main values"))
    rent = F.get("rent", [])
    b12 = A + pv + rent + (nat if OUTLOOK_USE_SEARCH else [])
    m12 = pair(df, b12, b12 + nxt(search), "target_12", 12, b12 + nxt(search)) if "target_12" in df else None
    r12 = check(m12, "a", "b", 12) if m12 is not None else None
    if r12:
        out.append(dict(r12, check="All metros", test="12-month outlook", version="Main values"))
    wait = pair(df, base, [f"{c}_wait" for c in base], "target", 3, base + [f"{c}_wait" for c in base] + nxt(search))
    early_same = pair(df, base, base + nxt(search), "target", 3, base + [f"{c}_wait" for c in base] + nxt(search))
    gap = None
    gw = check(wait, "a", "b", 3) if wait is not None else None
    ge = check(early_same, "a", "b", 3) if early_same is not None else None
    if gw and ge:
        out.append(dict(gw, check="All metros", test="Waiting for Zillow's next month", version="Main values"))
        gap = {"waiting_gain": gw["gain"], "early_read_gain": ge["gain"],
               "share_recovered": round(ge["gain"] / gw["gain"], 3) if gw["gain"] > 0 else None}

    table = pd.DataFrame(out)
    table.to_csv(RESULTS / "nowcast.csv", index=False)

    def get(check_name, version):
        r = table[(table["test"] == "Early read: search one month ahead") & (table["check"] == check_name)
                  & (table["version"] == version)]
        if not len(r):
            return None
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.iloc[0].to_dict().items()}

    allm, new, rawm = get("All metros", "Main values"), get("New metros (untouched test)", "Main values"), \
        get("All metros", "Raw values")
    rules = {"Main values, all 45 metros: better with p < 0.05":
             bool(allm and allm["gain"] > 0 and allm["p"] < 0.05),
             "Main values, 23 new metros: better (above zero)": bool(new and new["gain"] > 0),
             "Raw values, all 45 metros: better (above zero)": bool(rawm and rawm["gain"] > 0)}
    res = {"passed": all(rules.values()), "rules": rules, "rule_file": "docs/preregistration_nowcast.md",
           "all_metros": allm, "new_metros": new, "raw_values": rawm, "waiting_for_zillow": gap,
           "months_tested": int(allm["months"]) if allm else 0,
           "first_month": f"{m_main['date'].min():%b %Y}", "last_month": f"{m_main['date'].max():%b %Y}"}
    (RESULTS / "nowcast.json").write_text(json.dumps(res, indent=2, default=float))
    print("Early read", "PASSED" if res["passed"] else "FAILED", json.dumps(rules, indent=1))
    return res


def where_search_helped(panel, F):
    """Nationwide search added to housing data alone, 12 months ahead (where it showed up in
    step 7), broken down by region, metro group and period. Info only; nothing is decided here."""
    A, nat = F["A"], [c for c in F["B"] if c.startswith("rel_outoftown")]
    if not nat or "target_12" not in panel:
        return None
    m = pair(panel, A, A + nat, "target_12", 12, A + nat)
    if m is None:
        return None
    rows = {r["check"]: r for r in slices(m, "a", "b", 12)}
    other = [x for x in METROS if x not in SUN_BELT]
    for name, grp in (("Sun Belt and Mountain West", SUN_BELT), ("Coastal, Midwest and Northeast", other)):
        r = check(m[m["metro"].isin(grp)], "a", "b", 12)
        if r:
            rows[name] = {"check": name, **r}
    order = ["All metros", "Sun Belt and Mountain West", "Coastal, Midwest and Northeast", "Original 22 metros",
             "New metros (untouched test)", "Early years", "Recent years"]
    t = pd.DataFrame([rows[k] for k in order if k in rows])
    t["metros"] = [len(SUN_BELT) if c == "Sun Belt and Mountain West" else len(other) if c.startswith("Coastal")
                   else len(ORIGINAL_METROS) if c.startswith("Original") else len(METROS) - len(ORIGINAL_METROS)
                   if c.startswith("New") else len(METROS) for c in t["check"]]
    t.to_csv(RESULTS / "search_by_region.csv", index=False)
    print(t[["check", "gain", "p"]].to_string(index=False))
    return t


def interest_table():
    """Nationwide '[city] homes for sale' searches. Each city's series is on its own scale
    (100 = its own peak), so cities are compared on changes, not levels. Each city is also
    divided by the typical (median) city in the same month: that cancels anything that moves
    every city at once, such as the jump in spring 2026 that appeared in the nationwide data
    for nearly every city but not in local searches (most likely a change on Google's side)."""
    f = RAW / "outoftown_clean.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["date"])
    q = RAW / "search_quality.csv"
    if q.exists():
        bad = pd.read_csv(q).query("search == 'outoftown' and zero_share > 0.3")["metro"]
        d = d[~d["metro"].isin(bad)]
    d = d[d["date"] < pd.Timestamp.today().normalize().replace(day=1)]  # drop the unfinished month
    d = d.sort_values(["metro", "date"])
    roll = lambda s: s.rolling(3, min_periods=3).mean()
    d["index_3m"] = d.groupby("metro")["outoftown"].transform(roll)                 # 3-month average
    typ = d.groupby("date")["outoftown"].median().sort_index()
    d["typical_3m"] = d["date"].map(roll(typ))
    d["relative"] = d["index_3m"] / d["typical_3m"]
    d["vs_year_ago_pct"] = 100 * (d["relative"] / d.groupby("metro")["relative"].shift(12) - 1)
    d["rank_rising"] = d.groupby("date")["vs_year_ago_pct"].rank(ascending=False, method="min")
    d["metros_ranked"] = d.groupby("date")["vs_year_ago_pct"].transform("count")
    out = d[["date", "metro", "outoftown", "index_3m", "typical_3m", "relative", "vs_year_ago_pct",
             "rank_rising", "metros_ranked", "term"]].dropna(subset=["index_3m", "typical_3m"])
    out = out.round({"index_3m": 1, "typical_3m": 1, "relative": 3, "vs_year_ago_pct": 1})
    out.to_csv(RESULTS / "search_interest.csv", index=False)
    last = out[out["date"] == out["date"].max()].sort_values("rank_rising")
    print(f"Out-of-town interest through {out['date'].max():%b %Y}; rising most vs. other cities:",
          ", ".join(last["metro"].head(5)), "| falling most:", ", ".join(last["metro"].tail(3)))
    return out


def main():
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    F = {k: v.split(",") for k, v in pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"].items()}
    if "B" in F:
        early_read(panel, F)
        where_search_helped(panel, F)
    interest_table()


if __name__ == "__main__":
    main()
