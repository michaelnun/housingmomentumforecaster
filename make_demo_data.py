"""Simulated data in the same format as step 1, to test the pipeline without downloads.
Built so buyer search leads prices by ~4 months and Home Depot search lags them.
The numbers mean nothing; only use them to check that the code runs.

    python make_demo_data.py && python step2_build.py --raw data/raw_demo && python step3_model.py
"""
import numpy as np
import pandas as pd

from config import ROOT, METROS, START, BUYER_TERMS, HOME_IMPROVEMENT_TERMS

rng = np.random.default_rng(7)
out = ROOT / "data" / "raw_demo"
out.mkdir(parents=True, exist_ok=True)
dates = pd.date_range(START, "2026-09-01", freq="MS")
T = len(dates)
national = np.cumsum(rng.normal(0, 0.15, T))
mortgage = 4 + np.where(dates >= "2022-03-01", 2.8, 0) + rng.normal(0, 0.1, T)
season = np.sin(2 * np.pi * (dates.month - 3) / 12)

z, rf, fr, tr, ot = [], [], [], [], []
for m in METROS:
    demand = np.zeros(T)
    for t in range(1, T):
        demand[t] = 0.9 * demand[t - 1] + rng.normal(0, 0.3)
    demand += 0.4 * national
    growth = 0.003 + 0.004 * np.roll(demand, 4) + rng.normal(0, 0.004, T)
    growth[:4] = 0.003
    price = 250_000 * np.exp(np.cumsum(growth))
    z.append(pd.DataFrame({"metro": m, "date": dates, "zhvi": price}))
    rf.append(pd.DataFrame({
        "metro": m, "date": dates,
        "median_dom": 40 - 5 * np.roll(demand, 1) + 10 * season + rng.normal(0, 3, T),
        "avg_sale_to_list": 0.98 + 0.008 * np.roll(demand, 2) + rng.normal(0, 0.004, T),
        "inventory": 5000 * np.exp(-0.1 * demand + rng.normal(0, 0.05, T)),
        "price_drops": 0.2 - 0.02 * demand + rng.normal(0, 0.02, T),
        "months_of_supply": 3 - 0.3 * demand + rng.normal(0, 0.2, T),
        "homes_sold": 2000 * np.exp(0.1 * demand + rng.normal(0, 0.05, T)),
        "sold_above_list": 0.3 + 0.03 * demand + rng.normal(0, 0.03, T),
    }))
    fr.append(pd.DataFrame({"metro": m, "date": dates, "mortgage_rate": mortgage,
                            "unemployment": 4 - 0.2 * demand + rng.normal(0, 0.2, T)}))
    trd = {"metro": m, "date": dates}
    for term in BUYER_TERMS:
        trd[term] = np.clip(50 + 8 * demand + 10 * season + rng.normal(0, 6, T), 1, 100)
    for term in HOME_IMPROVEMENT_TERMS:
        trd[term] = np.clip(50 + 5 * np.roll(demand, -2) * 0 + 5 * np.roll(demand, 3)
                            + 15 * season + rng.normal(0, 6, T), 1, 100)
    tr.append(pd.DataFrame(trd))
    ot.append(pd.DataFrame({"metro": m, "date": dates, "outoftown": np.clip(
        50 + 7 * np.roll(demand, -2) + 8 * season + rng.normal(0, 6, T), 1, 100)}))

for name, parts in [("zillow", z), ("redfin", rf), ("fred", fr), ("trends", tr), ("outoftown", ot)]:
    pd.concat(parts).to_csv(out / f"{name}_clean.csv", index=False)
(out / "DEMO").write_text("simulated data")
print(f"Simulated data written to {out}")
