**Written November 11, 2026, 3:25 pm ET, before running this test.**

# Can search give an early read before Zillow's numbers come out?

Every earlier test gave Google search and housing data the same timing. In practice search is available right away, while Zillow publishes each month's numbers 3 to 6 weeks later. So at the start of a month you have search for last month but housing data only for the month before. This test asks whether that extra month of search improves the 3-month score. It has not been run before.

**The comparison** (`step10_search.py`, results in `results/nowcast.csv` and `results/nowcast.json`):

- **Score as published:** the 3-month score model (housing data + Realtor.com listing page views) using data through month T, the latest month Zillow has published.
- **Early read:** the same model plus Google search for month T+1, one month newer than the housing data. Search here means the search features already in the project: local buyer searches ("homes for sale", "mortgage calculator", "zillow", "realtor", "open house") and nationwide "[city] homes for sale".
- **What both predict:** the same thing the score predicts: each metro's home-value change from T to T+3, compared with the typical metro.
- **How it's judged:** the same month-by-month walk-forward test and measures as step 9 (change in ranking accuracy, Newey-West p-value), from May 2020 on.

**Pass rule, fixed now. The early read passes only if all three hold:**

1. On Zillow's main values, across all 45 metros, it improves ranking accuracy (change > 0) with p < 0.05.
2. On the 23 metros added in November 2026, the change is above zero.
3. On Zillow's raw (unsmoothed) values, across all 45 metros, the change is above zero.

**What happens next:**

- **If it passes:** the dashboard gets an "early read" of the 3-month score each month, clearly labeled, based on the newer month of search, until Zillow's numbers catch up.
- **If it fails:** the project reports that search doesn't help even as an early signal, and nothing is added to the score.

Either way, the result is reported as it comes out, including on the dashboard.

**Reported but not used for the decision:** each kind of search on its own, early vs. recent years, drop-one-metro, Sun Belt vs. other metros, the 12-month outlook, and how much of the benefit of waiting for Zillow's next month the early read recovers.

**Limits, stated up front:**

- We have seen these data in other tests; only this comparison is new.
- Google Trends re-samples its data on every download, so today's historical values aren't exactly what was available at the time.
- The test uses a full month of search. A run on the 22nd would only have about three weeks of the current month, so this is a best case.

---

**Result (added November 11, 2026, after the run; the rule above was not changed): failed.**

| Check | Change in ranking accuracy | Needed | Met? |
| --- | --- | --- | --- |
| Main values, all 45 metros | −0.003 (p = 0.93) | above zero, p < 0.05 | No |
| Main values, 23 new metros | −0.004 | above zero | No |
| Raw values, all 45 metros | −0.016 | above zero | No |

74 months tested, May 2020 to June 2026. For comparison, waiting for Zillow's next month would improve the score by +0.083, so newer data does matter, but an extra month of search recovers none of it. Neither kind of search helped on its own either (local buyer searches −0.003, nationwide searches +0.000). Search is not used for an early read.
