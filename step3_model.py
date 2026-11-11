"""Step 3: walk-forward test of Model A vs B vs C (vs P, with page views), this month's ranked
metro list, and the 12-month outlook model.

Walk-forward = for every test month, train only on months whose 3-month outcome was
already known at that time, then predict. Exactly what an investor could have done live.

    python step3_model.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED, RESULTS, HORIZON, TEST_START, PERIODS, SCORE_MODEL, OUTLOOK_USE_SEARCH

warnings.filterwarnings("ignore")
TOP_K = 5
COLORS = {"Naive": "#9aa0a6", "A": "#4c6ef5", "B": "#f08c00", "C": "#2b8a3e", "P": "#ae3ec9"}
LABELS = {"Naive": "Naive (trend continues)", "A": "A: housing data",
          "B": "B: + buyer search", "C": "C: + home-improvement search",
          "P": "P: + Realtor.com page views"}


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def fit_predict(train, test, cols, target, base=None):
    """Fit on training rows that have every column; predict the test rows.
    Returns (strict, live). strict: NaN for a metro missing any column (used for fair tests).
    live: if `base` is given, a metro missing only the extra (non-base) columns is predicted
    with those set to 0, i.e. 'typical, like the average metro' (every extra column is already
    relative to the average metro). One model for every metro, so scores stay comparable."""
    tr = train.dropna(subset=cols)
    strict = np.full(len(test), np.nan)
    live = np.full(len(test), np.nan)
    if len(tr) < 200:
        return strict, live
    model = ridge().fit(tr[cols], tr[target])
    ok = test[cols].notna().all(axis=1).values
    if ok.any():
        strict[ok] = model.predict(test.loc[ok, cols])
    if base is not None:
        okb = test[base].notna().all(axis=1).values
        if okb.any():
            live[okb] = model.predict(test.loc[okb, cols].fillna(0))
    return strict, live


def walk_forward(panel, feats, use_gbm=False, live=None):
    known = panel.dropna(subset=["target"])  # outcomes we can train on
    # predict every month, including the latest ones whose outcome isn't known yet
    months = sorted(panel.loc[panel["date"] >= TEST_START, "date"].unique())
    out, gbm, gbm_fit_month = [], None, None
    for m in months:
        cutoff = pd.Timestamp(m) - pd.DateOffset(months=HORIZON)  # outcome known by m
        train = known[known["date"] <= cutoff]
        test = panel[panel["date"] == m]
        if len(train) < 200 or train["date"].nunique() < 12:  # at least a year of history
            continue
        row = test[["metro", "date", "target"]].copy()
        row["Naive"] = test["rel_price_chg_3m"].values  # last 3 months repeat
        for name, cols in feats.items():
            base = (live or {}).get(name)
            strict, lv = fit_predict(train, test, cols, "target", base)
            row[name] = strict
            if base is not None:
                row[f"{name}_live"] = lv
        if use_gbm:  # nonlinear check on Model C, refit twice a year to save time
            if gbm_fit_month is None or (pd.Timestamp(m) - gbm_fit_month).days > 180:
                gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                    max_iter=300, random_state=0)
                gbm.fit(train[feats["C"]], train["target"])
                gbm_fit_month = pd.Timestamp(m)
            row["C_gbm"] = gbm.predict(test[feats["C"]])
        out.append(row)
    return pd.concat(out, ignore_index=True)


def month_metrics(g, model):
    g = g.dropna(subset=[model])  # each model is scored on the metros it has data for
    if len(g) < 2 * TOP_K:
        return pd.Series(dtype=float)
    p, y = g[model], g["target"]
    pred_top = set(g.nlargest(TOP_K, model)["metro"])
    true_top = set(g.nlargest(TOP_K, "target")["metro"])
    return pd.Series({
        "rank_corr": stats.spearmanr(p, y).correlation,
        "top5_hit": len(pred_top & true_top) / TOP_K,
        "mae": (p - y).abs().mean(),
        "sq_err": ((p - y) ** 2).mean(),
        "direction": (np.sign(p) == np.sign(y)).mean(),
    })


def scorecard(preds, models):
    rows = []
    for mdl in models:
        mm = preds.groupby("date").apply(month_metrics, model=mdl)
        rows.append({"model": mdl,
                     "rank_corr": mm["rank_corr"].mean(),
                     "top5_hit_rate": mm["top5_hit"].mean(),
                     "direction_acc": mm["direction"].mean(),
                     "MAE_pct": 100 * mm["mae"].mean(),
                     "RMSE_pct": 100 * np.sqrt(mm["sq_err"].mean())})
    return pd.DataFrame(rows).set_index("model")


def diebold_mariano(preds, a, b):
    """Is b's lower error real or luck? Monthly avg squared-error gap, with
    Newey-West standard errors because 3-month-ahead targets overlap."""
    preds = preds.dropna(subset=[a, b])  # same rows for both models
    e = preds.assign(d=(preds["target"] - preds[a]) ** 2 - (preds["target"] - preds[b]) ** 2)
    d = e.groupby("date")["d"].mean().values
    T, lag = len(d), HORIZON - 1
    dc = d - d.mean()
    var = dc @ dc / T + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
                                for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / T)
    return {"comparison": f"{b} vs {a}", "DM_stat": stat,
            "p_value_one_sided": 1 - stats.norm.cdf(stat),
            "verdict": f"{b} is better (significant)" if stat > 1.645 else "no clear difference"}


def lead_time(panel, col="search_index", max_lag=12):
    """How many months ahead does search move before prices?"""
    p = panel.copy()
    p["sig"] = p[col] - p.groupby("metro")[col].transform("mean")
    p["past_ret"] = p["target"]
    out = []
    for k in range(0, max_lag + 1):
        fut = p.groupby("metro")["target"].shift(-k)
        ok = p["sig"].notna() & fut.notna()
        out.append({"months_ahead": k, "corr": np.corrcoef(p.loc[ok, "sig"], fut[ok])[0, 1]})
    return pd.DataFrame(out)


def charts(card, periods, lt_buyer, lt_hi, ranked):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    models = [m for m in ["Naive", "A", "B", "C"] if m in card.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ["rank_corr", "top5_hit_rate"],
                                 ["Ranking accuracy (rank correlation)",
                                  f"Top-{TOP_K} hit rate"]):
        vals = card.loc[models, metric]
        bars = ax.bar([LABELS[m] for m in models], vals, color=[COLORS[m] for m in models])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        if metric == "top5_hit_rate":
            n = ranked["metro"].nunique()
            ax.axhline(TOP_K / n, ls="--", color="#666", lw=1)
            ax.text(-0.4, TOP_K / n + 0.01, "random pick", fontsize=9, color="#666")
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.2
    x = np.arange(len(periods.index.get_level_values(0).unique()))
    pers = list(periods.index.get_level_values(0).unique())
    for i, m in enumerate(models):
        vals = [periods.loc[(p, m), "rank_corr"] if (p, m) in periods.index else np.nan
                for p in pers]
        ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(x, [p.replace(" (", "\n(") for p in pers])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title("Ranking accuracy by market period", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "by_period.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lt_buyer["months_ahead"], lt_buyer["corr"], "-o", color=COLORS["B"],
            label="Buyer search (homes for sale, mortgage...)")
    ax.plot(lt_hi["months_ahead"], lt_hi["corr"], "-o", color=COLORS["C"],
            label="Home-improvement search (Home Depot...)")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("Months between search change and price move")
    ax.set_ylabel("Correlation")
    ax.set_title("How far ahead does search lead prices?", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "lead_time.png", dpi=180)
    plt.close(fig)

    r = ranked.sort_values("predicted_vs_avg_pct")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(r) + 1))
    cols = np.where(r["predicted_vs_avg_pct"] >= 0, COLORS["C"], "#c92a2a")
    ax.barh(r["metro"], r["predicted_vs_avg_pct"], color=cols)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Predicted 3-month price change vs average metro (percentage points)")
    ax.set_title(f"Metro ranking, as of {ranked['as_of'].iloc[0]}", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "ranked_list.png", dpi=180)
    plt.close(fig)


# ---------------- 12-month outlook
def walk_12(panel, feats12, base, product):
    """Same month-by-month test as above, 12 months ahead: train only on months whose
    12-month outcome was already known (at least 12 months before the test month)."""
    known = panel.dropna(subset=["target_12"])
    out = []
    for m in sorted(panel.loc[panel["date"] >= TEST_START, "date"].unique()):
        train = known[known["date"] <= pd.Timestamp(m) - pd.DateOffset(months=12)]
        test = panel[panel["date"] == m]
        if len(train) < 200 or train["date"].nunique() < 12:
            continue
        row = test[["metro", "date", "target_12"]].copy()
        row["Naive12"] = test["rel_price_chg_12m"].values  # last 12 months repeat
        for name, cols in feats12.items():
            strict, lv = fit_predict(train, test, cols, "target_12", base)
            row[name] = strict
            if name == product:
                row[f"{name}_live"] = lv
        out.append(row)
    return pd.concat(out, ignore_index=True) if out else None


def monthly_rank_corr(e, model, target):
    return e.groupby("date").apply(lambda x: x[model].rank().corr(x[target].rank()),
                                   include_groups=False)


def nw_pvalue(d, lag):
    """One-sided p-value that the average of d is above 0, Newey-West for overlapping months."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    dc = d - d.mean()
    var = dc @ dc / T + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T for k in range(1, lag + 1))
    return float(1 - stats.norm.cdf(d.mean() / np.sqrt(var / T))) if var > 0 else 1.0


def outlook_12(panel, fs):
    if "target_12" not in panel:
        return None
    A = fs["A"].split(",")
    nat = [c for c in fs.get("B", "").split(",") if "outoftown" in c]
    pv = fs["pageviews"].split(",") if "pageviews" in fs else []
    rent = fs["rent"].split(",") if "rent" in fs else []
    feats12 = {"A12": A}
    if pv or rent:
        feats12["Y12_nosearch"] = A + pv + rent
    if nat:
        feats12["Y12"] = A + nat + pv + rent
    print("12-month outlook: walk-forward testing...")
    product = ("Y12" if "Y12" in feats12 and OUTLOOK_USE_SEARCH
               else "Y12_nosearch" if "Y12_nosearch" in feats12 else "A12")
    preds = walk_12(panel, feats12, A, product)
    if preds is None:
        return None
    preds.to_csv(RESULTS / "predictions_12m.csv", index=False)
    models = ["Naive12"] + list(feats12)
    e = preds.dropna(subset=["target_12"] + models)
    labels = {"Naive12": "Trend continues (last 12 months repeat)", "A12": "Housing data only",
              "Y12_nosearch": "+ page views + rent", "Y12": "+ page views + rent + nationwide search"}
    rows, rc = [], {}
    for m in models:
        rc[m] = monthly_rank_corr(e, m, "target_12")
        g = e.groupby("date")
        rows.append({"model": m, "label": labels[m], "rank_corr": rc[m].mean(),
                     "top5_beat_avg": g.apply(lambda x: (x.nlargest(TOP_K, m)["target_12"] > 0).mean(),
                                              include_groups=False).mean(),
                     "bottom5_trail_avg": g.apply(lambda x: (x.nsmallest(TOP_K, m)["target_12"] < 0).mean(),
                                                  include_groups=False).mean(),
                     "top5_excess_pct": 100 * g.apply(lambda x: x.nlargest(TOP_K, m)["target_12"].mean(),
                                                      include_groups=False).mean()})
    card = pd.DataFrame(rows).set_index("model")
    card.round(3).to_csv(RESULTS / "scorecard_12m.csv")
    tests = {"months_tested": int(e["date"].nunique()),
             "first_month": f"{e['date'].min():%b %Y}", "last_month": f"{e['date'].max():%b %Y}",
             "model": product}
    best = tests["model"]
    tests["p_vs_trend"] = round(nw_pvalue(rc[best] - rc["Naive12"], 11), 4)
    tests["p_vs_housing_only"] = round(nw_pvalue(rc[best] - rc["A12"], 11), 4)
    if "Y12" in rc and "Y12_nosearch" in rc:
        d = rc["Y12"] - rc["Y12_nosearch"]
        tests["search_gain"] = round(float(d.mean()), 4)
        tests["search_better_share"] = round(float((d > 0).mean()), 3)
        tests["p_search_helps"] = round(nw_pvalue(d, 11), 4)
    lv = preds.dropna(subset=["target_12", f"{product}_live"])  # every metro, gaps filled as typical
    tests["all_metros_rank_corr"] = round(float(monthly_rank_corr(lv, f"{product}_live", "target_12").mean()), 3)
    tests["share_with_data_gaps"] = round(float(lv[product].isna().mean()), 3)
    import json
    (RESULTS / "tests_12m.json").write_text(json.dumps(tests, indent=2))
    print(card.round(3).to_string())
    print(tests)
    return card


def main():
    panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
    fs = pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"]
    feats = {k: v.split(",") for k, v in fs.items() if k in ("A", "B", "C")}  # extras: step 7
    if "pageviews" in fs:  # model P: housing data + Realtor.com listing page views
        feats["P"] = feats["A"] + fs["pageviews"].split(",")

    print("Walk-forward testing (refits every month)...")
    live = {SCORE_MODEL: feats["A"]} if SCORE_MODEL in feats and SCORE_MODEL != "A" else None
    preds = walk_forward(panel, feats, use_gbm=True, live=live)
    preds.to_csv(RESULTS / "predictions.csv", index=False)
    models = ["Naive", "A", "B", "C", "C_gbm"] + (["P"] if "P" in feats else [])
    # score only known outcomes, and compare every model on the same rows
    # score only known outcomes; each model on the metros it has data for (pairwise tests use
    # the same rows for both models)
    all_preds, preds = preds, preds.dropna(subset=["target"])

    card = scorecard(preds, models)
    card.round(3).to_csv(RESULTS / "scorecard.csv")

    per = []
    for name, (s, e) in PERIODS.items():
        sub = preds[(preds["date"] >= s) & (preds["date"] <= e)]
        if sub["date"].nunique() >= 3:
            per.append(scorecard(sub, ["Naive", "A", "B", "C"] + (["P"] if "P" in feats else []))
                       .assign(period=name))
    periods = pd.concat(per).reset_index().set_index(["period", "model"])
    periods.round(3).to_csv(RESULTS / "scorecard_by_period.csv")

    dm = pd.DataFrame([diebold_mariano(preds, "Naive", "A"),
                       diebold_mariano(preds, "A", "B"),
                       diebold_mariano(preds, "B", "C")]
                      + ([diebold_mariano(preds.dropna(subset=["P"]), "A", "P")] if "P" in feats else []))
    dm.round(3).to_csv(RESULTS / "significance_tests.csv", index=False)

    lt_b, lt_h = lead_time(panel, "search_index"), lead_time(panel, "homeimp_index")
    lt = lt_b.merge(lt_h, on="months_ahead", suffixes=("_buyer", "_homeimp"))
    lt.round(3).to_csv(RESULTS / "lead_time.csv", index=False)

    # Which model goes in the product? The one set in config.py, else the best of A/B/C.
    best = SCORE_MODEL if SCORE_MODEL in feats else card.loc[["A", "B", "C"], "rank_corr"].idxmax()
    trained = panel.dropna(subset=["target"] + feats[best])
    final = ridge().fit(trained[feats[best]], trained["target"])
    # every metro with housing data; missing extra signals count as typical (0), as in step 4
    latest = panel[panel["date"] == panel["date"].max()].dropna(subset=feats["A"])
    ranked = pd.DataFrame({
        "metro": latest["metro"].values,
        "predicted_vs_avg_pct": 100 * final.predict(latest[feats[best]].fillna(0)),
    }).sort_values("predicted_vs_avg_pct", ascending=False)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["signal"] = np.where(ranked["rank"] <= TOP_K, "Overweight",
                       np.where(ranked["rank"] > len(ranked) - TOP_K, "Avoid", "Hold"))
    ranked["model_used"] = best
    ranked["as_of"] = f"{panel['date'].max():%b %Y}"
    ranked.round(2).to_csv(RESULTS / "ranked_list_latest.csv", index=False)

    coefs = pd.Series(final[-1].coef_, index=feats[best]).sort_values(key=abs, ascending=False)
    coefs.round(4).to_csv(RESULTS / "drivers.csv", header=["standardized_weight"])

    charts(card, periods, lt_b, lt_h, ranked)
    outlook_12(panel, fs)

    pd.set_option("display.width", 120)
    print("\nSCORECARD (test period", f"{preds['date'].min():%b %Y}-{preds['date'].max():%b %Y})")
    print(card.round(3).to_string())
    print("\nSIGNIFICANCE")
    print(dm.round(3).to_string(index=False))
    print("\nLEAD TIME: buyer search peaks at",
          int(lt_b.loc[lt_b["corr"].idxmax(), "months_ahead"]), "months ahead")
    print(f"\nTOP {TOP_K} METROS NOW (model {best}):")
    print(ranked.head(TOP_K)[["rank", "metro", "predicted_vs_avg_pct"]].to_string(index=False))
    print(f"\nAll outputs saved in {RESULTS}")


if __name__ == "__main__":
    main()
