# Data guide: where every file comes from

Every file here is downloaded or built by `step1_collect.py` (raw data) and `step2_build.py` (the model's table). The monthly GitHub run refreshes them on the 22nd. `raw/downloaded.json` records when each file was last downloaded.

**Raw downloads** keep the source's own format. **`_clean` files** are our tidy version: one row per metro per month, 45 metros, ready to use.

## Housing market: Zillow Research (free CSVs, [zillow.com/research/data](https://www.zillow.com/research/data/))

All from `https://files.zillowstatic.com/research/public_csvs/<folder>/Metro_<folder>_<file>.csv`. Each covers ~900 U.S. metros; we keep 45.

| File | Zillow folder | What it is |
| --- | --- | --- |
| `raw/zillow_zhvi_metro.csv` | `zhvi` (smoothed, seasonally adjusted) | Typical home value (Zillow Home Value Index) |
| `raw/zillow_zhvi_metro_raw.csv` | `zhvi` (not smoothed) | Same, without Zillow's smoothing; used for the raw-prices check |
| `raw/zillow_rent.csv` | `zori` | Typical rent (Zillow Observed Rent Index) |
| `raw/zillow_median_dom.csv` | `mean_doz_pending` | Days from listing to pending sale |
| `raw/zillow_avg_sale_to_list.csv` | `mean_sale_to_list` | Sale price as a share of list price |
| `raw/zillow_inventory.csv` | `invt_fs` | Homes for sale |
| `raw/zillow_price_drops.csv` | `perc_listings_price_cut` | Share of listings with a price cut |
| `raw/zillow_sold_above_list.csv` | `pct_sold_above_list` | Share of homes sold above list price |
| `raw/zillow_new_listings.csv` | `new_listings` | New listings |
| `raw/zillow_sales_count.csv` | `sales_count_now` | Homes sold |
| `raw/zillow_clean.csv`, `raw/zillow_raw_clean.csv`, `raw/rent_clean.csv`, `raw/market_clean.csv` | | Our tidy versions of the files above, 45 metros, Jan 2015 on |

## Web usage: what people look at (Realtor.com, via FRED)

| File | Source | What it is |
| --- | --- | --- |
| `raw/pageviews_clean.csv` | Realtor.com Market Hotness on [FRED](https://fred.stlouisfed.org/), series `LDPEPRVSUSMSA<metro code>` | Listing page views per home for sale, metro vs. U.S. (1 = U.S. average), Aug 2017 on |

## Web usage: what people search for (Google Trends, via the `pytrends` package)

Searched for each metro's TV market (Google Trends' smallest area), Jan 2015 on. Google Trends returns a slightly different sample on each download, so each series is downloaded at least twice and averaged.

| File | What it is |
| --- | --- |
| `raw/trends/<Metro>.csv` | Local buyer searches ("homes for sale", "mortgage calculator", "zillow", "realtor", "open house") and home-improvement searches ("home depot", "lowes"), 0 to 100 |
| `raw/trends_seller/<Metro>.csv` | Local seller searches ("sell my house", "what is my home worth") |
| `raw/trends_archive/` | Every individual download kept before averaging |
| `raw/trends_clean.csv`, `raw/seller_clean.csv` | Our tidy versions, all metros in one file |
| `raw/outoftown_clean.csv` | Nationwide searches for "[city] homes for sale" (interest from other places) |
| `raw/movingto_clean.csv` | Nationwide searches for "moving to [city]" |
| `raw/search_quality.csv` | How many downloads were averaged for each search, and how often it was zero |

## Web content: what the news says

| File | Source | What it is |
| --- | --- | --- |
| `raw/mediacloud/<Metro>_counts.csv` | [Media Cloud](https://www.mediacloud.org/) search API, U.S. National collection (free key) | Daily count of stories naming the city and a housing phrase, and the share of all stories |
| `raw/mediacloud/<Metro>_headlines.csv` | Media Cloud | Every matching story: date, headline, link, outlet |
| `raw/mediacloud/queries.json` | Our settings | The exact search used for each metro |
| `raw/mediacloud_clean.csv` | Built from the two above | Monthly story counts and average headline sentiment (VADER plus housing words, `sentiment.py`) |
| `raw/mediacloud_quality.csv` | Built | Stories per month and whether each metro has enough news to use |
| `raw/news/<Metro>.csv`, `raw/news_*.csv` | [GDELT](https://www.gdeltproject.org/) DOC 2.0 API | The first news source tried; only 16 metros finished, so it is preliminary |

## Economy

| File | Source | What it is |
| --- | --- | --- |
| `raw/fred_clean.csv` | [FRED](https://fred.stlouisfed.org/): `MORTGAGE30US`, `<state>UR` | 30-year mortgage rate; state unemployment rate |
| `raw/employment_clean.csv` | [BLS](https://www.bls.gov/) public API: `SMU…` (payroll jobs), `LAUMT…` (unemployment) | Metro jobs and unemployment (Cleveland not available) |
| `raw/home_improvement_clean.csv` | Census Monthly State Retail Sales on FRED: `MSRS<state>444` | Sales at building-material and garden stores (Home Depot, Lowe's), change from a year ago, 2019 on |

## The model's table

| File | What it is |
| --- | --- |
| `processed/panel.csv` | Everything above joined: one row per metro per month (4,082 rows × 63 columns, March 2019 on). Columns starting `rel_` are compared with the typical metro that month. `target` = next 3 months' home value growth minus the average metro's |
| `processed/feature_sets.csv` | Which columns each model uses |

Results (scores, forecasts, every test) are in [`../results`](../results).
