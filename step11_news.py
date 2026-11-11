"""Step 11: does news coverage of a city's housing market help predict it? (web content)

The pass/fail rule was written down before any news data was downloaded:
docs/preregistration_news.md. This script applies it exactly.

  - 3-month score: the current score model (housing data + Realtor.com listing page views)
    vs. the same model plus news tone and volume.
  - 12-month outlook: the current outlook model (+ rent growth) vs. the same plus news.
  Each passes only if: main values, all metros, better with p < 0.05; better on the 23 metros
  added in Nov 2026; better on Zillow's raw values.

Writes results/news_test.csv and results/news_test.json, plus results/news_metro.csv and
results/news_headlines.csv for the dashboard (each metro's news tone over time and its latest
headlines, shown as context).

    python step11_news.py
"""
import json

import numpy as np
import pandas as pd

from config import METROS, OUTLOOK_USE_SEARCH, PROCESSED, RAW, RESULTS, SCORE_MODEL
from step9_stability import check, pair, raw_targets, slices

SUN_BELT = list(METROS)[:21]  # config lists the Sun Belt and Mountain West metros first
RULE = "docs/preregistration_news.md"


def tagged(m, h, test, version):
    return [dict(r, test=test, version=version) for r in slices(m, "a", "b", h)] if m is not None else []


def one_line(m, h, test, version, check_name="All metros"):
    r = check(m, "a", "b", h) if m is not None else None
    return [dict(r, check=check_name, test=test, version=version)] if r else []


def run_test(panel, F):
    A = F["A"]
    pv = F.get("pageviews", []) if SCORE_MODEL == "P" else []
    rent = F.get("rent", [])
    nat = [c for c in F.get("B", []) if c.startswith("rel_outoftown")]
    news = F["news"]
    tone = [c for c in news if "tone" in c]
    vol = [c for c in news if "vol" in c]
    raw = raw_targets(panel) if (RAW / "zillow_raw_clean.csv").exists() else None
    if raw is not None:  # 6 months ahead on main values, for the info-only check
        z = pd.read_csv(RAW / "zillow_clean.csv", parse_dates=["date"]).sort_values(["metro", "date"])
        fwd = np.log(z.groupby("metro")["zhvi"].shift(-6)) - np.log(z["zhvi"])
        z["y_6"] = fwd - fwd.groupby(z["date"]).transform("mean")
        panel = panel.merge(z[["metro", "date", "y_6"]], on=["metro", "date"], how="left")
    out, models = [], {}

    # Decision tests
    b3 = A + pv
    b12 = A + pv + rent + (nat if OUTLOOK_USE_SEARCH else [])
    for h, base, target, name in ((3, b3, "target", "3-month score"), (12, b12, "target_12", "12-month outlook")):
        if target not in panel:
            continue
        m = pair(panel, base, base + news, target, h, base + news)
        models[name] = m
        out += tagged(m, h, name, "Main values")
        if raw is not None:
            rb = A + ["rel_raw_chg_3m"] + (pv if h == 3 else pv + rent)
            mr = pair(raw, rb, rb + news, f"y_raw_{h}", h, rb + news)
            out += [r for r in tagged(mr, h, name, "Raw values") if r["check"] == "All metros"]

    # Info only
    m3 = models.get("3-month score")
    for label, cols in (("Tone only", tone), ("Volume only", vol)):
        if cols:
            out += one_line(pair(panel, b3, b3 + cols, "target", 3, b3 + news), 3, f"3-month score: {label.lower()}",
                            "Main values")
    if "y_6" in panel:
        out += one_line(pair(panel, b3, b3 + news, "y_6", 6, b3 + news), 6, "6 months ahead", "Main values")
    if m3 is not None:
        for label, grp in (("Sun Belt and Mountain West", SUN_BELT),
                           ("Coastal, Midwest and Northeast", [x for x in METROS if x not in SUN_BELT])):
            out += one_line(m3[m3["metro"].isin(grp)], 3, "3-month score", "Main values", label)

    table = pd.DataFrame(out)
    table.to_csv(RESULTS / "news_test.csv", index=False)

    def get(test, check_name, version):
        r = table[(table["test"] == test) & (table["check"] == check_name) & (table["version"] == version)]
        if not len(r):
            return None
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.iloc[0].to_dict().items()}

    res = {"rule_file": RULE, "metros_with_news": int(panel.dropna(subset=news)["metro"].nunique())}
    for name, key in (("3-month score", "3m"), ("12-month outlook", "12m")):
        allm, new, rawm = (get(name, "All metros", "Main values"), get(name, "New metros (untouched test)", "Main values"),
                           get(name, "All metros", "Raw values"))
        rules = {"Main values, all metros: better with p < 0.05": bool(allm and allm["gain"] > 0 and allm["p"] < 0.05),
                 "Main values, 23 new metros: better (above zero)": bool(new and new["gain"] > 0),
                 "Raw values, all metros: better (above zero)": bool(rawm and rawm["gain"] > 0)}
        m = models.get(name)
        res[key] = {"passed": all(rules.values()) and allm is not None, "rules": rules, "all_metros": allm,
                    "new_metros": new, "raw_values": rawm,
                    "first_month": f"{m['date'].min():%b %Y}" if m is not None else None,
                    "last_month": f"{m['date'].max():%b %Y}" if m is not None else None}
        print(f"News, {name}: {'PASSED' if res[key]['passed'] else 'FAILED'}", rules)
    (RESULTS / "news_test.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def dashboard_tables():
    """Each metro's news tone and volume over time, and its latest headlines (context only)."""
    f = RAW / "news_clean.csv"
    if not f.exists():
        return
    d = pd.read_csv(f, parse_dates=["date"])
    d = d[d["date"] < pd.Timestamp.today().normalize().replace(day=1)].sort_values(["metro", "date"])  # whole months
    qf = RAW / "news_quality.csv"
    if qf.exists():
        d = d[d["metro"].isin(pd.read_csv(qf).query("usable")["metro"])]
    g = d.groupby("metro")
    d["tone_x"] = d["tone"] * d["articles"]
    n3 = g["articles"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    d["tone_3m"] = g["tone_x"].transform(lambda s: s.rolling(3, min_periods=3).sum()) / n3.where(n3 > 0)
    d["articles_3m"] = g["articles"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    d["typical_tone_3m"] = d.groupby("date")["tone_3m"].transform("median")
    d[["metro", "date", "articles", "articles_3m", "tone", "tone_3m", "typical_tone_3m"]].round(3).to_csv(
        RESULTS / "news_metro.csv", index=False)
    hf = RAW / "news_headlines.csv"
    if hf.exists():
        h = pd.read_csv(hf).dropna(subset=["title", "url"])
        h["seen"] = pd.to_datetime(h["seen"], format="%Y%m%dT%H%M%SZ", errors="coerce")
        h.sort_values(["metro", "seen"], ascending=[True, False]).groupby("metro").head(6).to_csv(
            RESULTS / "news_headlines.csv", index=False)


def main():
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    F = {k: v.split(",") for k, v in pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"].items()}
    if "news" in F:
        run_test(panel, F)
    else:
        print("No news data in the panel yet (run step 1 and step 2 first).")
    dashboard_tables()


if __name__ == "__main__":
    main()
