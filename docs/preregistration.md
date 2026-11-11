**Written November 11, 2026, before downloading data for the 23 new metros.**

Up to now every modeling choice was made looking only at the original 22 metros:

- the **3-month score** uses housing data **plus Realtor.com listing page views**, because page views improved the forecast on Zillow's raw (unsmoothed) values;
- the **12-month outlook** uses housing data, page views, rent growth **and nationwide Google search**, because search improved it further.

The 23 metros added next (New York, Los Angeles, San Francisco, Washington DC, Philadelphia, Minneapolis, Detroit, Portland, Sacramento, St. Louis, Pittsburgh, Baltimore, Kansas City, Cincinnati, Cleveland, Milwaukee, Richmond, Oklahoma City, Memphis, Louisville, New Orleans, Birmingham, Buffalo) have never been used to make a choice. They are an untouched test.

**The tests** (step 9, `results/stability.csv`), each judged by the month-by-month change in ranking accuracy with Newey-West p-values:

1. The 3-month score beats "the recent trend continues" (main and raw values).
2. Page views improve the 3-month score (raw values).
3. The 12-month outlook beats "the recent trend continues" (main and raw values).
4. Nationwide Google search improves the 12-month outlook (main and raw values).

Each is also checked on early vs. recent years and with each metro dropped in turn.

**Decision rules, fixed now:**

- If page views do **not** improve the raw-value 3-month forecast on the 23 new metros (change ≤ 0), the 3-month score goes back to housing data only.
- If nationwide search does **not** improve the 12-month outlook on the 23 new metros (change ≤ 0 on main values), **or** is not significant across all 45 metros, search comes out of the 12-month outlook and the project reports that search did not help.
- Whatever the results, they are reported as they come out, including on the dashboard.

**What counts as "Solid":** significant across all metros on every version tested, better (above zero) in every check, and still significant whichever single metro is dropped. "Holds, with caveats" = significant overall but weaker somewhere. "Not supported" = not significant overall.

---

**Results (added November 11, 2026, after the 45-metro run; the rules above were not changed).**

Change in ranking accuracy on the 23 untouched metros, and across all 45:

| Test | 23 new metros | All 45 metros | Outcome |
| --- | --- | --- | --- |
| Page views improve the 3-month score (raw values) | +0.083 (p < 0.001) | +0.052 (p < 0.001) | **Passed.** Page views stay in the 3-month score. |
| Nationwide search improves the 12-month outlook (main values) | −0.060 (p = 0.96) | −0.001 (p = 0.56) | **Failed.** Search was taken out of the 12-month outlook. |
| 3-month score beats "trend continues" | +0.190 (p < 0.001) | +0.147 (p < 0.001) | Solid on every check. |
| 12-month outlook beats "trend continues" (without search) | +0.062 (p = 0.19) | +0.235 (p = 0.003) | Holds across all metros but not clearly on the new ones, so the 12-month outlook is labeled **experimental**. |

What this means for the project's question: Google searches did not improve the forecasts once housing data and Realtor.com listing views were included. Listing views (people looking at actual homes for sale) did, including on metros the model had never been tuned on.
