"""Step 1: download raw data. Safe to re-run: anything already downloaded is skipped.

    python step1_collect.py            # everything
    python step1_collect.py --skip-trends
    python step1_collect.py --pulls 3  # average 3 Google Trends pulls (less noise, slower)
"""
import argparse
import io
import random
import time

import pandas as pd
import requests

import sys


class _Tee:
    """Print to the window AND to data/collect_log.txt, so progress can be checked."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.__stdout__
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


from config import (RAW, START, METROS, BUYER_TERMS, HOME_IMPROVEMENT_TERMS,
                    FRED_NATIONAL, BLS_AREAS)

ZILLOW_URL = ("https://files.zillowstatic.com/research/public_csvs/zhvi/"
              "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
REDFIN_URL = ("https://redfin-public-data.s3.us-west-2.amazonaws.com/"
              "redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def download(url, dest):
    if dest.exists():
        print(f"  already have {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def city_key(name):
    """'Phoenix-Mesa-Chandler, AZ metro area' -> 'phoenix, az' for matching."""
    name = str(name).lower().replace(" metro area", "")
    city, _, state = name.partition(",")
    return f"{city.split('-')[0].split('/')[0].strip()}, {state.strip()[:2]}"


def collect_zillow():
    print("Zillow home values")
    dest = RAW / "zillow_zhvi_metro.csv"
    download(ZILLOW_URL, dest)
    df = pd.read_csv(dest)
    keys = {city_key(v[0]): k for k, v in METROS.items()}
    df["metro"] = df["RegionName"].map(lambda n: keys.get(city_key(n)))
    df = df.dropna(subset=["metro"])
    date_cols = [c for c in df.columns if c[:2] in ("19", "20")]
    long = df.melt(id_vars="metro", value_vars=date_cols, var_name="date", value_name="zhvi")
    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp()
    long = long[long["date"] >= START]
    long.to_csv(RAW / "zillow_clean.csv", index=False)
    missing = set(METROS) - set(long["metro"])
    print(f"  {long['metro'].nunique()} metros matched" + (f"; MISSING {missing}" if missing else ""))


ZILLOW_BASE = "https://files.zillowstatic.com/research/public_csvs/{folder}/Metro_{folder}_{tail}"
# our column name : Zillow folder names to try (mean first, then median)
ZILLOW_MARKET = {
    "median_dom": ["mean_doz_pending", "med_doz_pending"],        # days to pending
    "avg_sale_to_list": ["mean_sale_to_list", "med_sale_to_list"],
    "inventory": ["invt_fs"],                                    # homes for sale
    "price_drops": ["perc_listings_price_cut"],                  # share with a price cut
    "sold_above_list": ["pct_sold_above_list"],
    "new_listings": ["new_listings"],
    "sales_count": ["sales_count_now"],                          # homes sold
}
ZILLOW_TAILS = ["uc_sfrcondo_sm_month.csv", "uc_sfrcondo_month.csv", "uc_sfr_sm_month.csv",
                "uc_sfrcondo_sm_week.csv"]


def zillow_long(path, value_name):
    df = pd.read_csv(path)
    keys = {city_key(v[0]): k for k, v in METROS.items()}
    df["metro"] = df["RegionName"].map(lambda n: keys.get(city_key(n)))
    df = df.dropna(subset=["metro"])
    date_cols = [c for c in df.columns if c[:2] in ("19", "20")]
    long = df.melt(id_vars="metro", value_vars=date_cols, var_name="date", value_name=value_name)
    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp()
    return long.groupby(["metro", "date"], as_index=False)[value_name].mean()


def collect_market():
    """Market conditions from Zillow (Redfin's old download link now returns 403)."""
    print("Zillow market data (days to pending, sale-to-list, inventory, price cuts)")
    frames = []
    for col, folders in ZILLOW_MARKET.items():
        dest = RAW / f"zillow_{col}.csv"
        if not dest.exists():
            for folder in folders:
                for tail in ZILLOW_TAILS:
                    url = ZILLOW_BASE.format(folder=folder, tail=tail)
                    try:
                        r = requests.get(url, timeout=120)
                    except requests.RequestException:
                        continue
                    if r.ok and r.text.startswith("RegionID"):
                        dest.write_bytes(r.content)
                        print(f"  {col}: {folder}/{tail}")
                        break
                if dest.exists():
                    break
        if dest.exists():
            frames.append(zillow_long(dest, col))
        else:
            print(f"  {col}: NOT FOUND on Zillow, skipping")
    if not frames:
        raise RuntimeError("No Zillow market data found")
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["metro", "date"], how="outer")
    df = df[df["date"] >= START]
    df.to_csv(RAW / "market_clean.csv", index=False)
    print(f"  {df['metro'].nunique()} metros, {df['date'].min():%b %Y} to {df['date'].max():%b %Y}")


def collect_redfin():
    print("Redfin market data (large file, a few minutes)")
    dest = RAW / "redfin_metro.tsv.gz"
    download(REDFIN_URL, dest)
    keys = {city_key(v[0]): k for k, v in METROS.items()}
    want = ["period_begin", "region", "property_type", "is_seasonally_adjusted",
            "median_dom", "avg_sale_to_list", "inventory", "price_drops",
            "months_of_supply", "homes_sold", "sold_above_list"]
    parts = []
    for chunk in pd.read_csv(dest, sep="\t", compression="gzip", chunksize=200_000,
                             low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = chunk[(chunk["property_type"] == "All Residential")
                      & (chunk["is_seasonally_adjusted"].astype(str).str.lower().str[0] == "f")]
        if "period_duration" in chunk.columns:  # keep monthly rows, drop rolling 90-day ones
            pdur = pd.to_numeric(chunk["period_duration"], errors="coerce")
            if (pdur == 30).any():
                chunk = chunk[pdur == 30]
        chunk = chunk[[c for c in want if c in chunk.columns]]
        chunk["metro"] = chunk["region"].map(lambda n: keys.get(city_key(n)))
        parts.append(chunk.dropna(subset=["metro"]))
    df = pd.concat(parts)
    df["date"] = pd.to_datetime(df["period_begin"]).dt.to_period("M").dt.to_timestamp()
    df = (df.drop(columns=["period_begin", "region", "property_type", "is_seasonally_adjusted"])
            .groupby(["metro", "date"]).mean(numeric_only=True).reset_index())
    df = df[df["date"] >= START]
    df.to_csv(RAW / "redfin_clean.csv", index=False)
    print(f"  {df['metro'].nunique()} metros matched")


def fred(series_id):
    r = requests.get(FRED_URL.format(series_id), timeout=60)
    r.raise_for_status()
    s = pd.read_csv(io.StringIO(r.text))
    s.columns = ["date", "value"]
    s["date"] = pd.to_datetime(s["date"])
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    return s.set_index("date")["value"].resample("MS").mean()


def collect_fred():
    print("FRED macro data")
    out = {name: fred(sid) for name, sid in FRED_NATIONAL.items()}
    nat = pd.DataFrame(out)
    states = sorted({v[2] for v in METROS.values()})
    ur = pd.DataFrame({st: fred(f"{st}UR") for st in states})
    rows = []
    for metro, (_, _, st) in METROS.items():
        d = nat.copy()
        d["unemployment"] = ur[st]
        d["metro"] = metro
        rows.append(d.reset_index().rename(columns={"index": "date"}))
    df = pd.concat(rows)
    df = df[df["date"] >= START]
    df.to_csv(RAW / "fred_clean.csv", index=False)
    print(f"  mortgage rate + unemployment for {len(states)} states")


def collect_home_improvement():
    """Census monthly sales at building-material & garden stores (Home Depot, Lowe's...),
    by state, % change from a year ago. Free on FRED as MSRS<ST>444. Starts 2019."""
    print("Home-improvement store sales by state (Census, via FRED)")
    states = sorted({v[2] for v in METROS.values()})
    got = {}
    for st in states:
        try:
            got[st] = fred(f"MSRS{st}444")
        except Exception as e:  # keep going if one state is missing
            print(f"  {st}: not available ({type(e).__name__})")
    rows = []
    for metro, (_, _, st) in METROS.items():
        if st in got:
            rows.append(pd.DataFrame({"metro": metro, "date": got[st].index,
                                      "hi_sales_yoy": got[st].values}))
    df = pd.concat(rows)
    df.to_csv(RAW / "home_improvement_clean.csv", index=False)
    print(f"  {len(got)} of {len(states)} states, {df['date'].min():%b %Y} to "
          f"{df.dropna()['date'].max():%b %Y}")


BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"


def bls(series_ids, start, end):
    r = requests.post(BLS_URL, json={"seriesid": series_ids, "startyear": str(start),
                                     "endyear": str(end)}, timeout=60)
    r.raise_for_status()
    js = r.json()
    out = {}
    for s in js.get("Results", {}).get("series", []):
        pts = [(f"{p['year']}-{p['period'][1:]}-01", p["value"]) for p in s["data"]
               if p["period"].startswith("M") and p["period"] != "M13"]
        if pts:
            v = pd.Series({pd.Timestamp(d): pd.to_numeric(x, errors="coerce") for d, x in pts})
            out[s["seriesID"]] = v.sort_index()
    return out


def collect_employment():
    """Metro payroll jobs (CES) and metro unemployment rate (LAUS) from the BLS public API.
    No key needed (limited to 25 requests a day, this uses about 8)."""
    print("Metro employment (BLS)")
    dest = RAW / "employment_clean.csv"
    if dest.exists():
        print("  already have it")
        return
    ids = {}
    for metro, (fips, areas) in BLS_AREAS.items():
        for a in areas:
            ids[f"SMU{fips}{a}0000000001"] = (metro, "jobs")
            ids[f"LAUMT{fips}{a}00000003"] = (metro, "metro_unemp")
    keys = list(ids)
    series = {}
    first = int(START[:4])
    last = pd.Timestamp.today().year
    for y0 in range(first, last + 1, 10):        # v1 API: max 10 years per request
        y1 = min(y0 + 9, last)
        for i in range(0, len(keys), 25):          # max 25 series per request
            part = bls(keys[i:i + 25], y0, y1)
            for k, v in part.items():
                series[k] = pd.concat([series.get(k, pd.Series(dtype=float)), v])
            time.sleep(2)
    rows = {}
    for k, v in series.items():
        metro, col = ids[k]
        if (metro, col) not in rows:               # first code that worked wins (Boston)
            rows[(metro, col)] = v[~v.index.duplicated()].sort_index()
    if not rows:
        print("  BLS returned nothing; keeping state unemployment only")
        return
    frames = []
    for metro in BLS_AREAS:
        d = pd.DataFrame({c: rows[(metro, c)] for c in ("jobs", "metro_unemp") if (metro, c) in rows})
        if d.empty:
            print(f"  {metro}: no BLS data")
            continue
        d.index.name = "date"
        frames.append(d.reset_index().assign(metro=metro))
    df = pd.concat(frames)
    df.to_csv(dest, index=False)
    have = df.dropna(subset=["jobs"])["metro"].nunique() if "jobs" in df else 0
    print(f"  jobs for {have} metros, {df['date'].min():%b %Y} to {df['date'].max():%b %Y}")


def trends_pull(pytrends, terms, geo, timeframe):
    for attempt in range(6):
        try:
            pytrends.build_payload(terms, geo=geo, timeframe=timeframe)
            df = pytrends.interest_over_time()
            return df.drop(columns="isPartial", errors="ignore")
        except Exception as e:  # Google rate-limits often; back off and retry
            wait = 60 * (attempt + 1) + random.uniform(0, 30)
            print(f"    rate limited ({type(e).__name__}), waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Google Trends kept failing for {geo}")


def collect_trends(pulls):
    from pytrends.request import TrendReq
    print("Google Trends (slow on purpose to avoid Google's rate limit)")
    out_dir = RAW / "trends"
    out_dir.mkdir(exist_ok=True)
    timeframe = f"{START} {pd.Timestamp.today():%Y-%m-%d}"
    pytrends = TrendReq(hl="en-US", tz=300)
    for metro, (_, geo, _) in METROS.items():
        dest = out_dir / f"{metro.replace(' ', '_')}.csv"
        if dest.exists():
            continue
        print(f"  {metro}")
        frames = []
        for terms in (BUYER_TERMS, HOME_IMPROVEMENT_TERMS):
            # Each pull is a sample, so values wobble. Averaging pulls reduces that noise.
            pulls_df = [trends_pull(pytrends, terms, geo, timeframe) for _ in range(pulls)]
            frames.append(sum(pulls_df) / pulls)
            time.sleep(random.uniform(15, 30))
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
        df.index.name = "date"
        df.to_csv(dest)
    all_ = []
    for metro in METROS:
        f = out_dir / f"{metro.replace(' ', '_')}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"])
            d["metro"] = metro
            all_.append(d)
    pd.concat(all_).to_csv(RAW / "trends_clean.csv", index=False)


def collect_out_of_town():
    """Nationwide searches for '<city> homes for sale'. Mostly people outside the metro,
    i.e. buyers thinking of moving there. Scaling differs per request, but we only use
    each city's change over time, so that does not matter."""
    from pytrends.request import TrendReq
    print("Google Trends: out-of-town interest")
    dest = RAW / "outoftown_clean.csv"
    if dest.exists():
        print("  already have it")
        return
    timeframe = f"{START} {pd.Timestamp.today():%Y-%m-%d}"
    pytrends = TrendReq(hl="en-US", tz=300)
    names = list(METROS)
    terms = {m: f"{m.lower()} homes for sale" for m in names}
    parts = []
    for i in range(0, len(names), 5):
        group = names[i:i + 5]
        df = trends_pull(pytrends, [terms[m] for m in group], "US", timeframe)
        df.columns = group
        parts.append(df)
        time.sleep(random.uniform(15, 30))
    df = pd.concat(parts, axis=1)
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    long = df.reset_index().melt(id_vars=df.index.name or "index", var_name="metro",
                                 value_name="outoftown")
    long = long.rename(columns={long.columns[0]: "date"})
    long.to_csv(dest, index=False)
    print(f"  {len(names)} metros")


if __name__ == "__main__":
    sys.stdout = sys.stderr = _Tee(RAW.parent / "collect_log.txt")
    print(f"--- run started {pd.Timestamp.now():%Y-%m-%d %H:%M} ---")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-trends", action="store_true")
    ap.add_argument("--pulls", type=int, default=1)
    a = ap.parse_args()
    collect_zillow()
    collect_fred()
    for extra in (collect_home_improvement, collect_employment):
        try:
            extra()
        except Exception as e:  # optional extras must never stop the main download
            print(f"  {extra.__name__} failed: {type(e).__name__}: {e}")
    collect_market()
    if not a.skip_trends:
        collect_trends(a.pulls)
        collect_out_of_town()
    print("Done. Next: python step2_build.py")
