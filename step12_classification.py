"""Classification check: does a metro beat the typical metro over the next 3 months (yes/no)?
Compares, month by month with the same walk-forward rules as step 3:
  - "Trend continues" (last 3 months' relative growth > 0)
  - our ridge score model (P: housing data + Realtor.com page views), read as yes if its forecast > 0
  - a logistic regression on the same inputs, trained only on months whose outcome was known
Writes results/classification.csv and results/classification.json."""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import PROCESSED, RESULTS, HORIZON, TEST_START, ORIGINAL_METROS
from step3_model import nw_pvalue

panel = pd.read_csv(PROCESSED / "panel.csv", parse_dates=["date"])
fs = pd.read_csv(PROCESSED / "feature_sets.csv", index_col=0)["features"]
cols = fs["A"].split(",") + fs["pageviews"].split(",")
preds = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"])

known = panel.dropna(subset=["target"])
rows = []
for m in sorted(preds["date"].unique()):
    cutoff = pd.Timestamp(m) - pd.DateOffset(months=HORIZON)
    train = known[known["date"] <= cutoff].dropna(subset=cols)
    test = panel[panel["date"] == m].dropna(subset=cols)
    if len(train) < 200 or train["date"].nunique() < 12 or test.empty:
        continue
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(train[cols], (train["target"] > 0).astype(int))
    rows.append(pd.DataFrame({"metro": test["metro"].values, "date": m,
                              "Logit": model.predict_proba(test[cols])[:, 1]}))
lg = pd.concat(rows)
d = preds.merge(lg, on=["metro", "date"]).dropna(subset=["target", "P", "Naive", "Logit"])
d["beat"] = (d["target"] > 0).astype(int)
MODELS = {"Trend continues": ("Naive", 0.0), "Ridge score model (ours)": ("P", 0.0), "Logistic regression": ("Logit", 0.5)}

def evaluate(g):
    out = []
    for name, (col, cut) in MODELS.items():
        yhat = (g[col] > cut).astype(int)
        tn, fp, fn, tp = confusion_matrix(g["beat"], yhat, labels=[0, 1]).ravel()
        by_m = g.groupby("date")
        rank = by_m.apply(lambda x: spearmanr(x[col], x["target"])[0]).mean()
        top5 = by_m.apply(lambda x: x.nlargest(5, col)["beat"].mean()).mean()
        acc_m = by_m.apply(lambda x: accuracy_score(x["beat"], (x[col] > cut).astype(int)))
        out.append({"model": name, "accuracy": accuracy_score(g["beat"], yhat),
                    "precision": precision_score(g["beat"], yhat), "recall": recall_score(g["beat"], yhat),
                    "auc": roc_auc_score(g["beat"], g[col]), "rank_corr": rank, "top5_hit": top5,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "_acc_m": acc_m})
    return out

res = {}
for subset, g in {"All 45 metros": d, "23 untouched metros": d[~d["metro"].isin(ORIGINAL_METROS)]}.items():
    ev = evaluate(g)
    ridge_m = [e for e in ev if e["model"].startswith("Ridge")][0]["_acc_m"]
    for e in ev:
        diff = (e["_acc_m"] - ridge_m).dropna()
        e["acc_vs_ridge_p"] = None if e["model"].startswith("Ridge") else float(nw_pvalue(diff, HORIZON))
        e.pop("_acc_m")
    res[subset] = ev
    print(f"\n{subset}: {len(g)} metro-months, {g['date'].nunique()} months, beat share {g['beat'].mean():.2f}")
    print(pd.DataFrame(ev).round(3).to_string(index=False))
flat = pd.DataFrame([{"subset": s, **e} for s, ev in res.items() for e in ev])
flat.to_csv(RESULTS / "classification.csv", index=False)
json.dump({"months": [str(d["date"].min().date()), str(d["date"].max().date())], "n": int(len(d)), "results": res,
           "features": cols}, open(RESULTS / "classification.json", "w"), indent=1, default=float)
