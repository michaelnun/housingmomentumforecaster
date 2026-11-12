**Written November 11, 2026, 4:36 pm ET, before any news data was downloaded.**

# Does the news about a city's housing market help predict it?

So far the project has tested web *usage* data: what people search for (Google Trends) and what they look at (Realtor.com listing page views). This test adds web *content*: how much the news writes about each city's housing market, and how positive or negative that coverage is. No news data has been downloaded or looked at yet.

**The data** (GDELT DOC 2.0 API, free, U.S. English-language news, January 2017 on):

- **Query for each metro:** the city's name together with at least one housing phrase: "housing market", "home prices", "home sales", "real estate market" or "housing prices". City names that other places share get their state added (Portland Oregon, Columbus Ohio, Richmond Virginia, Birmingham Alabama), and Washington DC and St. Louis are written out in full. The exact query for each metro is in `config.py` (`NEWS_QUERIES`).
- **Per metro, per month:** the number of matching articles, the share of all news GDELT tracked that month, and the average tone of the matching articles (GDELT's sentiment score, weighted by article count).
- **Features,** each compared with the typical metro that month like everything else: average tone over the last 3 months; the change in that tone vs. 3 months earlier; and the change in news volume (3-month average share) vs. a year earlier.
- **Data quality check, fixed now:** a metro averaging fewer than 5 matching articles a month is treated as having no news data. A query may be reworded only if it returns almost nothing, and only before the test is run.

**The test** (`step11_news.py`, results in `results/news_test.csv` and `results/news_test.json`): the same month-by-month walk-forward test and measures as steps 9 and 10 (change in ranking accuracy, Newey-West p-value), on every month the data allow.

- **3-month score:** the current score model (housing data + Realtor.com listing page views) vs. the same model plus the news features.
- **12-month outlook:** the current outlook model (housing data + page views + rent growth) vs. the same model plus the news features.

**Pass rule, fixed now.** For each of the two, news passes only if all three hold:

1. On Zillow's main values, across all 45 metros, ranking accuracy improves (change > 0) with p < 0.05.
2. On the 23 metros added in November 2026, the change is above zero.
3. On Zillow's raw (unsmoothed) values, across all 45 metros, the change is above zero.

**What happens next:**

- **If it passes for the 3-month score:** the news features are added to the 3-month score.
- **If it passes for the 12-month outlook:** they are added to the 12-month outlook.
- **If it fails:** the project reports that news coverage didn't help. The dashboard still shows each metro's news (tone over time and recent headlines) as context, clearly labeled, the same way it shows out-of-town search interest.

Either way, the result is reported as it comes out, including on the dashboard.

**Reported but not used for the decision:** 6 months ahead; tone and volume on their own; Sun Belt vs. other metros; early vs. recent years; dropping each metro in turn.

**Limits, stated up front:**

- A city's name plus a housing phrase also catches national stories that mention the city in passing.
- GDELT's tone score is an automated dictionary measure, not a reading of each article.
- We have already seen the housing, search and page-view data in other tests; only the news data is new.

---

**Addendum (November 11, 2026, 5:52 pm ET, written while the first news download was still running and before any news results existed).** GDELT limited how fast GitHub's computers could download, so some cities' downloads failed (for example, Memphis and Oklahoma City were logged as "no news found" after three blocked requests). A failed download is not the same as "too little news," so those cities don't fall under the data quality rule above. The downloader now records failed downloads separately and retries them on the next run. **The decision uses the first run in which every metro's news downloaded.** Any result from a run with failed downloads is shown as preliminary and doesn't count. Our own headline sentiment (VADER, scored as is and with housing words added) is shown on the dashboard as context and compared with GDELT's tone; it is not part of this test, because GDELT returns full headlines only for the last few months.

---

**Addendum 2 (November 12, 2026, 12:55 AM ET, before any Media Cloud data was downloaded): a second news source.** GDELT kept blocking downloads (both GitHub's computers and a home connection), so the same question is also tested with **Media Cloud** (a free news archive run by researchers at MIT, Northeastern and UMass). Only the archive's date coverage may be checked before running this (whether stories exist back to 2017); no relationship with prices is looked at.

- **Data:** Media Cloud's U.S. National news collection; the same city names and housing phrases as above (Lucene syntax, in `config.py`, `MC_QUERIES`). Per city: daily story counts and the share of all stories in the collection, and the headlines of the matching stories.
- **Tone comes from our own scoring:** each headline that mentions the city or a housing word is scored with VADER plus the housing word list (`sentiment.py`), and averaged by month.
- **Features,** compared with the typical metro like everything else: average headline tone over the last 3 months (needs at least 10 headlines, otherwise missing); its change vs. 3 months earlier; news volume (3-month average share) vs. a year earlier.
- **Test and pass rule:** exactly the same as above (3-month score and 12-month outlook; three conditions each; only a run in which every metro's Media Cloud data downloaded counts). If the archive covers too little history for at least 24 months of the month-by-month test, the test is not run and the project says so.
- **Two news tests now exist (GDELT and Media Cloud).** Each is reported. News goes into a score only if that source passes its own rule; if both pass, the one with the larger gain on the 23 new metros is used.
