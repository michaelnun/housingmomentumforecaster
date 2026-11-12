"""Step 11: does news coverage of a city's housing market help predict it? (web content)

The pass/fail rule was written down before any news data was downloaded:
docs/preregistration_news.md. This script applies it exactly.

  - 3-month score: the current score model (housing data + Realtor.com listing page views)
    vs. the same model plus news tone and volume.
  - 12-month outlook: the current outlook model (+ rent growth) vs. the same plus news.
  Each passes only if: main values, all metros, better with p < 0.05; better on the 23 metros
  added in Nov 2026; better on Zillow's raw values.

Writes results/news_test.csv and results/news_test.json, plus, for the dashboard: each metro's news
tone over time (results/news_metro.csv), its latest headlines with our own sentiment score
(results/news_headlines.csv), and a per-metro headline sentiment summary checked against GDELT's tone
(results/news_sentiment.csv, results/news_sentiment.json).

Our own headline sentiment uses VADER, a standard dictionary-based sentiment tool, two ways: as is,
and with a short list of housing words added (a general tool reads "prices fall" as neutral or even
positive and "bidding wars" as negative). It is context only: GDELT only returns full headlines for
the last few months, so it can't be tested historically yet; the archive builds that history.

    python step11_news.py
"""
import json

import numpy as np
import pandas as pd

from config import METROS, OUTLOOK_USE_SEARCH, PROCESSED, RAW, RESULTS, SCORE_MODEL
from step9_stability import check, pair, raw_targets, slices

SUN_BELT = list(METROS)[:21]  # config lists the Sun Belt and Mountain West metros first
RULE = "docs/preregistration_news.md"
# Housing words for VADER (scale -4 to +4): positive = market strengthening, negative = weakening
HOUSING_TERMS = {
    **{w: 1.5 for w in ("rise", "rises", "rising", "rose", "climb", "climbs", "climbing", "climbed", "jump", "jumps",
                        "gain", "gains", "rebound", "rebounds", "recovery", "strong", "stronger", "hot", "appreciation")},
    **{w: 2.0 for w in ("surge", "surges", "surging", "soar", "soars", "soaring", "boom", "booming", "hottest")},
    **{w: 1.0 for w in ("bidding", "competitive")},
    **{w: -1.5 for w in ("fall", "falls", "falling", "fell", "drop", "drops", "dropping", "dropped", "decline",
                         "declines", "declining", "slide", "slides", "sliding", "slowdown", "stall", "stalls",
                         "stalled", "sluggish", "glut", "bubble", "correction")},
    **{w: -2.0 for w in ("slump", "slumps", "plunge", "plunges", "foreclosure", "foreclosures", "underwater")},
    **{w: -1.0 for w in ("cool", "cools", "cooling", "cooled", "slow", "slows", "slowing", "cut", "cuts")},
    "plummet": -2.5, "plummets": -2.5, "crash": -3.0,
    "war": 0.0, "wars": 0.0,  # "bidding wars" means a hot market
}


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
    qf = RAW / "news_quality.csv"
    q = pd.read_csv(qf) if qf.exists() else pd.DataFrame()
    problems = sorted(q.loc[q["status"].isin(["download failed", "not downloaded yet"]), "metro"]) \
        if "status" in q else []
    # Only a run in which every metro's news downloaded counts (addendum in the rule file)
    res["complete"] = "status" in q and not problems
    res["download_problems"] = problems
    res["too_little_news"] = sorted(q.loc[q["status"] == "too little news", "metro"]) if "status" in q else []
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
    headline_sentiment(d)


HOUSING_WORDS = (r"\b(home|homes|house|houses|housing|real estate|mortgages?|rents?|renters|rental|realtors?|"
                 r"listings?|propert(y|ies)|apartments?|condos?|buyers|sellers|homebuyers|homeowners)\b")
CITY_PATTERNS = {"Washington DC": r"washington|d\.c\.", "St. Louis": r"st\.? louis|saint louis"}


def on_topic(h):
    """Keep headlines that mention the city or housing; GDELT also returns national stories that
    only mention the city in passing (earnings calls, politics)."""
    title = h["title"].astype(str).str.lower()
    housing = title.str.contains(HOUSING_WORDS, regex=True)
    city = pd.Series([bool(pd.Series([t]).str.contains(CITY_PATTERNS.get(m, m.lower()), regex=True).iloc[0])
                      for t, m in zip(title, h["metro"])], index=h.index)
    return housing | city


def scorers():
    """VADER as is, and VADER with housing words added. None if VADER isn't installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("vaderSentiment not installed; skipping headline sentiment (pip install vaderSentiment)")
        return None, None
    plain, housing = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
    housing.lexicon.update(HOUSING_TERMS)
    return plain, housing


def headline_sentiment(news_monthly):
    """Score recent headlines; summarize per metro; check against GDELT's tone across metros."""
    f = next((x for x in (RAW / "news_headlines_archive.csv", RAW / "news_headlines.csv") if x.exists()), None)
    if f is None:
        return
    h = pd.read_csv(f).dropna(subset=["title", "url"]).drop_duplicates(["metro", "url"])
    h["seen"] = pd.to_datetime(h["seen"].astype(str), format="%Y%m%dT%H%M%SZ", errors="coerce")
    h = h.dropna(subset=["seen"])
    n_all = len(h)
    h = h[on_topic(h)]
    off_topic = n_all - len(h)
    plain, housing = scorers()
    if plain is not None and len(h):
        h["sent_plain"] = [plain.polarity_scores(str(t))["compound"] for t in h["title"]]
        h["sent_housing"] = [housing.polarity_scores(str(t))["compound"] for t in h["title"]]
        h["label"] = pd.cut(h["sent_housing"], [-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"])
    h.sort_values(["metro", "seen"], ascending=[True, False]).groupby("metro").head(6).round(3).to_csv(
        RESULTS / "news_headlines.csv", index=False)
    if plain is None or not len(h):
        return
    recent = h[h["seen"] >= h["seen"].max() - pd.Timedelta(days=60)]
    s = recent.groupby("metro").agg(headlines=("title", "size"), sent_housing=("sent_housing", "mean"),
                                    sent_plain=("sent_plain", "mean"),
                                    share_positive=("label", lambda x: (x == "positive").mean()),
                                    share_negative=("label", lambda x: (x == "negative").mean())).reset_index()
    # GDELT's tone over the same recent months, to compare with our own scores
    since = (recent["seen"].min().to_period("M")).to_timestamp()
    g = news_monthly[news_monthly["date"] >= since].assign(tx=lambda x: x["tone"] * x["articles"])
    g = g.groupby("metro").apply(lambda x: x["tx"].sum() / x["articles"].sum() if x["articles"].sum() else np.nan,
                                 include_groups=False).rename("gdelt_tone").reset_index()
    s = s.merge(g, on="metro", how="left")
    s.round(3).to_csv(RESULTS / "news_sentiment.csv", index=False)
    ok = s[s["headlines"] >= 5].dropna(subset=["gdelt_tone"])
    def corr(c):
        v = ok[c].rank().corr(ok["gdelt_tone"].rank()) if len(ok) >= 10 else np.nan
        return None if pd.isna(v) else round(float(v), 3)
    out = {"headlines_scored": int(len(recent)), "metros": int(s["metro"].nunique()),
           "from": f"{recent['seen'].min():%b %d, %Y}", "to": f"{recent['seen'].max():%b %d, %Y}",
           "headlines_in_archive": int(n_all), "dropped_as_off_topic": int(off_topic),
           "agreement_with_gdelt_plain": corr("sent_plain"), "agreement_with_gdelt_housing": corr("sent_housing"),
           "share_label_changed_by_housing_words": round(float(
               (np.sign(recent["sent_plain"].round(2)) != np.sign(recent["sent_housing"].round(2))).mean()), 3)}
    (RESULTS / "news_sentiment.json").write_text(json.dumps(out, indent=2))
    print("Headline sentiment:", out)


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
