# Housing Momentum Forecaster

A monthly dashboard that ranks 45 big U.S. housing markets by where home values are headed, compared with the typical big metro, and explains why. Built for real estate firms, home-improvement retailers and investors.

**Live dashboard:** https://housing-momentum-forecaster.streamlit.app/

![Dashboard overview](docs/overview.png)

## What it does

- **3-month score (0–100)** for each metro, every month: how home values are expected to grow over the next 3 months compared with the typical big metro. Built from housing market data plus **Realtor.com listing page views**.
- **12-month outlook (0–100, experimental):** the same for the year ahead, adding rent growth.
- **Stages:** Rising, Turning up, Turning down, Falling (where prices have been vs. where they're expected to go).
- **How often it's right** next to every score, a **"market right now"** panel (home value, rent, days to sell, price cuts, buyer's or seller's market), plain-language reasons, advice for each audience, CSV downloads and shareable links for each metro.
- **Out-of-town interest:** how much people across the U.S. search "[city] homes for sale", compared with other cities and a year ago. Shown as context only, not used in the scores (see findings).
- **In the news:** how positive or negative each city's housing news is (GDELT tone) over time, and its latest headlines.
- **Usage analytics:** the dashboard records anonymous page views (which pages, metros and industries people look at, and downloads) with Google Analytics, so we can see how the tool is used.
- **Updates itself** every month (GitHub Actions), and saves each month's published scores so the live track record can't be revised later.

![Metro detail](docs/metro.png)

## Key findings

Tested month by month from May 2020 to 2026. Every forecast used only data available at the time.

| Finding | Result |
| --- | --- |
| 3-month score vs. "trend continues" | Ranking accuracy 0.88 vs. 0.73 (p < 0.001). **Solid** on every stability check |
| Top 5 picks | Beat the typical metro over the next 3 months 99% of the time, about +6% a year faster growth |
| Scores 80–100 | Beat the typical metro 98% of the time (3 months) and 88% (12 months) |
| Turning points | Score crossed 50 before 84% of 174 lasting price-trend turns, typically 2 months early |
| **Does Google search help?** | **No.** Not at 3 or 6 months. Nationwide search looked helpful 12 months out on the first 22 metros, but on 23 metros added later it made the outlook worse (−0.06), and across all 45 it added nothing (p = 0.56). It was removed, as the [pre-registered rule](docs/preregistration.md) required |
| Where search did help | Nationwide "[city] homes for sale" searches helped a year out in Sun Belt and Mountain West metros (+0.037, p = 0.003), where people from other states were buying in 2020–23, and hurt in coastal, Midwest and Northeast metros (−0.082). The first 22 metros were mostly Sun Belt, which is why it looked useful at first. The gain faded after 2023, and listing views and rent capture it |
| Search as an early read | **No.** Search arrives weeks before Zillow's numbers, so we tested whether one extra month of search improves the 3-month score, with the [rule written down first](docs/preregistration_nowcast.md). It didn't (−0.003, p = 0.93); waiting for Zillow's next month would add +0.083 |
| **Does website activity help?** | **Yes.** Realtor.com listing page views improve the forecast on Zillow's raw values, including on the 23 untouched metros (+0.08, p < 0.001). **Solid** |
| 12-month outlook | Ranking accuracy 0.65 vs. 0.42 for "trend continues" across all metros (p = 0.003), but not clearly better on the 23 new metros, so it's labeled experimental |
| Didn't help | Local buyer and seller searches, Home Depot / Lowe's searches, metro jobs, Census store sales |

**Takeaway:** what people *search* for didn't add anything beyond housing data, not even as an early signal, but what they actually *look at* (listing views) did. Nationwide search tracked one big event, the 2020–23 move to the Sun Belt, and didn't carry over to other regions or later years.

**News coverage (web content), being tested:** does how much the news writes about a city's housing market, and how positive it is, improve the forecasts? The pass/fail rule was [written down before any news data was downloaded](docs/preregistration_news.md); the result appears on the dashboard's Track record page after the first run with news data.

**How we kept it honest:** all modeling choices were made on the original 22 metros. The rules for judging them on 23 new metros were [written down before that data was downloaded](docs/preregistration.md). Every finding is re-checked on raw (unsmoothed) values, on early vs. recent years and with each metro dropped (`step9_stability.py`; dashboard → Track record).

![2022 case study](docs/case_study_2022.png)

![Which signals help](docs/signal_tests.png)

**Limits:** Zillow smooths its home-value series, which flatters short-term accuracy; on raw values it's lower (0.63 vs. 0.54 for the trend at 3 months) but still significant. The scores compare metros with each other and don't forecast national price levels. Google Trends data is a sample, and nationwide searches for nearly every city jumped in spring 2026 and fell back by fall while local searches didn't (most likely a change in Google's data). Realtor.com changed how it counts page views in late 2021.

## Data (all free and public)

- [Zillow Research](https://www.zillow.com/research/data/): home values, rents (ZORI), days to pending, sale-to-list ratio, inventory, price cuts, new listings, homes sold
- [Realtor.com Market Hotness, via FRED](https://fred.stlouisfed.org/): listing page views per property vs. the U.S., by metro
- [FRED](https://fred.stlouisfed.org/): 30-year mortgage rate, state unemployment, Census home-improvement store sales by state
- [BLS](https://www.bls.gov/): metro payroll jobs and unemployment
- [Google Trends](https://trends.google.com/): local buyer, seller and Home Depot / Lowe's searches; nationwide "[city] homes for sale" and "moving to [city]" (averaged across monthly downloads)
- [GDELT](https://www.gdeltproject.org/) DOC 2.0 API: U.S. news articles about each city's housing market (monthly volume and tone, 2017 on) and the latest headlines

The downloaded data is stored in `data/`.

## Method

1. `step1_collect.py`: download all data
2. `step2_build.py`: one row per metro-month; every feature compared with the average metro that month
3. `step3_model.py`: month-by-month (walk-forward) ridge regression tests vs. a "trend continues" baseline, with Diebold–Mariano tests; the 12-month outlook model
4. `step4_index.py`: scores, stages, reasons, calibration, market stats, prediction log
5. `step5_activity.py`: does search predict new listings and homes sold?
6. `step6_findings.py`: results by period, misses, turning points, 2022 case study, charts
7. `step7_signal_tests.py`: does each extra signal help at 3, 6 and 12 months?
8. `step8_robustness.py`: the tests again on Zillow's unsmoothed values
9. `step9_stability.py`: untouched metros, early vs. recent years, drop-one-metro, verdict per finding
10. `step10_search.py`: search as an early read (pre-registered test), where search helped, and the out-of-town interest table for the dashboard
11. `step11_news.py`: does news coverage help? (pre-registered test), plus each metro's news tone and headlines for the dashboard
12. `dashboard.py`: Streamlit dashboard, with anonymous usage tracking (Google Analytics 4, server-side)

## Monthly updates (automatic)

`.github/workflows/monthly_update.yml` runs the whole pipeline on the 22nd of every month and saves the new data and results here; the dashboard picks them up within minutes. Anyone with write access can also run it: **Actions → Monthly data update → Run workflow**. If Google Trends blocks GitHub's servers, the run uses the search data already saved and still finishes.

## Usage analytics (Google Analytics)

The dashboard sends anonymous events to Google Analytics 4 with the Measurement Protocol: a page view for each page and metro opened (with the industry picked), file downloads, and visits that arrive through a shared metro link. No personal information is sent. Tracking is off unless the app's Streamlit secrets contain:

```toml
[ga4]
measurement_id = "G-XXXXXXXXXX"
api_secret = "your-measurement-protocol-secret"
```

(Streamlit Community Cloud: app → Settings → Secrets. Never put these in the repo.)

## Run it yourself

```bash
pip install -r requirements.txt
python step1_collect.py          # about 1 hour (Google Trends is slow)
python step2_build.py && python step3_model.py && python step4_index.py
python step5_activity.py && python step6_findings.py && python step7_signal_tests.py
python step8_robustness.py && python step9_stability.py && python step10_search.py && python step11_news.py
streamlit run dashboard.py
```

On Windows, double-click `run_all.bat`, then `run_dashboard.bat`.

## Team

Web Analytics, Fordham University Gabelli School of Business, Fall 2026.
