"""Settings shared by every step. Edit METROS to change the study set."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for p in (RAW, PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

START = "2015-01-01"   # first month of data
HORIZON = 3            # predict price change this many months ahead
TEST_START = "2018-01-01"  # walk-forward predictions start here (gives a pre-COVID test period)
# Which model the Housing Momentum Score uses.
#   "A" = housing data only. "P" = housing data + Realtor.com listing page views.
# P was chosen in Nov 2026 (after seeing the test results): same accuracy as A on Zillow's
# smoothed prices, better on unsmoothed prices (step 8). A metro missing page views is scored by
# the same model with page views set to typical. If page-view data is missing entirely, A is used.
SCORE_MODEL = "P"
# Whether the 12-month outlook uses nationwide Google search. Set to False on Nov 11, 2026 by
# the rule in docs/preregistration.md: on the 23 metros added later, search made the outlook
# worse (-0.06), and across all 45 it added nothing (p = 0.56).
OUTLOOK_USE_SEARCH = False

# Periods for the "does it hold up in different markets?" breakdown
PERIODS = {
    "Pre-COVID (2018-Feb 2020)": ("2018-01-01", "2020-02-01"),
    "COVID boom (Mar 2020-Jun 2022)": ("2020-03-01", "2022-06-01"),
    "Rate shock (Jul 2022-Dec 2023)": ("2022-07-01", "2023-12-01"),
    "Recent (2024 on)": ("2024-01-01", "2100-01-01"),
}

# name used in charts : (Zillow/Redfin "City, ST" key, Google Trends DMA code, state)
# DMA = Nielsen media market, the smallest area Google Trends reports.
METROS = {
    # Sun Belt and Mountain West
    "Phoenix":        ("Phoenix, AZ",        "US-AZ-753", "AZ"),
    "Dallas":         ("Dallas, TX",         "US-TX-623", "TX"),
    "Houston":        ("Houston, TX",        "US-TX-618", "TX"),
    "Austin":         ("Austin, TX",         "US-TX-635", "TX"),
    "San Antonio":    ("San Antonio, TX",    "US-TX-641", "TX"),
    "Atlanta":        ("Atlanta, GA",        "US-GA-524", "GA"),
    "Tampa":          ("Tampa, FL",          "US-FL-539", "FL"),
    "Orlando":        ("Orlando, FL",        "US-FL-534", "FL"),
    "Miami":          ("Miami, FL",          "US-FL-528", "FL"),
    "Jacksonville":   ("Jacksonville, FL",   "US-FL-561", "FL"),
    "Charlotte":      ("Charlotte, NC",      "US-NC-517", "NC"),
    "Raleigh":        ("Raleigh, NC",        "US-NC-560", "NC"),
    "Nashville":      ("Nashville, TN",      "US-TN-659", "TN"),
    "Memphis":        ("Memphis, TN",        "US-TN-640", "TN"),
    "Birmingham":     ("Birmingham, AL",     "US-AL-630", "AL"),
    "New Orleans":    ("New Orleans, LA",    "US-LA-622", "LA"),
    "Oklahoma City":  ("Oklahoma City, OK",  "US-OK-650", "OK"),
    "Richmond":       ("Richmond, VA",       "US-VA-556", "VA"),
    "Denver":         ("Denver, CO",         "US-CO-751", "CO"),
    "Las Vegas":      ("Las Vegas, NV",      "US-NV-839", "NV"),
    "Salt Lake City": ("Salt Lake City, UT", "US-UT-770", "UT"),
    # West Coast
    "Los Angeles":    ("Los Angeles, CA",    "US-CA-803", "CA"),
    "San Diego":      ("San Diego, CA",      "US-CA-825", "CA"),
    "San Francisco":  ("San Francisco, CA",  "US-CA-807", "CA"),
    "Sacramento":     ("Sacramento, CA",     "US-CA-862", "CA"),
    "Seattle":        ("Seattle, WA",        "US-WA-819", "WA"),
    "Portland":       ("Portland, OR",       "US-OR-820", "OR"),
    # Northeast and Mid-Atlantic
    "New York":       ("New York, NY",       "US-NY-501", "NY"),
    "Boston":         ("Boston, MA",         "US-MA-506", "MA"),
    "Philadelphia":   ("Philadelphia, PA",   "US-PA-504", "PA"),
    "Washington DC":  ("Washington, DC",     "US-DC-511", "DC"),
    "Baltimore":      ("Baltimore, MD",      "US-MD-512", "MD"),
    "Pittsburgh":     ("Pittsburgh, PA",     "US-PA-508", "PA"),
    "Buffalo":        ("Buffalo, NY",        "US-NY-514", "NY"),
    # Midwest
    "Chicago":        ("Chicago, IL",        "US-IL-602", "IL"),
    "Detroit":        ("Detroit, MI",        "US-MI-505", "MI"),
    "Minneapolis":    ("Minneapolis, MN",    "US-MN-613", "MN"),
    "St. Louis":      ("St. Louis, MO",      "US-MO-609", "MO"),
    "Kansas City":    ("Kansas City, MO",    "US-MO-616", "MO"),
    "Milwaukee":      ("Milwaukee, WI",      "US-WI-617", "WI"),
    "Columbus":       ("Columbus, OH",       "US-OH-535", "OH"),
    "Cincinnati":     ("Cincinnati, OH",     "US-OH-515", "OH"),
    "Cleveland":      ("Cleveland, OH",      "US-OH-510", "OH"),
    "Indianapolis":   ("Indianapolis, IN",   "US-IN-527", "IN"),
    "Louisville":     ("Louisville, KY",     "US-KY-529", "KY"),
}
# How each city is written in nationwide searches, when not just its name in lowercase
# City names that other places share get their state added, so a nationwide search for
# "columbus homes for sale" doesn't mix in Columbus, Georgia.
SEARCH_NAMES = {"Washington DC": "washington dc", "St. Louis": "st louis",
                "Portland": "portland oregon", "Columbus": "columbus ohio",
                "Richmond": "richmond va", "Birmingham": "birmingham al"}
# The 22 metros the project started with. Every modeling choice (page views in the 3-month
# score, search in the 12-month outlook) was made looking only at these. The other 23 were
# added afterwards, so they are an untouched test of those choices (see step 9).
ORIGINAL_METROS = ["Phoenix", "Dallas", "Houston", "Austin", "San Antonio", "Atlanta", "Tampa",
                   "Orlando", "Miami", "Jacksonville", "Charlotte", "Raleigh", "Nashville", "Denver",
                   "Las Vegas", "Salt Lake City", "Seattle", "San Diego", "Boston", "Chicago",
                   "Columbus", "Indianapolis"]

# Google Trends terms. Pulled together per metro (max 5 per request).
BUYER_TERMS = ["homes for sale", "mortgage calculator", "zillow", "realtor", "open house"]
# Model C: home-improvement interest, the Home Depot idea, via search.
# ("home inspection" and "moving company" were dropped: zero in about 60% of months, too noisy)
HOME_IMPROVEMENT_TERMS = ["home depot", "lowes"]
# Seller-side searches in each metro: more people looking to sell can mean more supply ahead.
# ("what is my home worth" was dropped: zero in about 90% of months, too noisy)
SELLER_TERMS = ["sell my house"]
# Nationwide searches about each city, searched across the whole U.S. ({} = city name).
NATIONWIDE_SEARCHES = {
    "outoftown": "{} homes for sale",   # people anywhere looking at homes there
    "movingto": "moving to {}",         # people anywhere planning a move there
}

# National FRED series (free CSV, no key needed)
FRED_NATIONAL = {"mortgage_rate": "MORTGAGE30US"}
# State unemployment rate = "<ST>UR" on FRED, a reliable pattern for every state.

# For metro employment from the BLS and Realtor.com page views on FRED:
# (state FIPS, [CBSA code, ...]). Boston also tries its NECTA code for the BLS.
BLS_AREAS = {
    "Phoenix": ("04", ["38060"]), "Dallas": ("48", ["19100"]), "Houston": ("48", ["26420"]),
    "Austin": ("48", ["12420"]), "San Antonio": ("48", ["41700"]), "Atlanta": ("13", ["12060"]),
    "Tampa": ("12", ["45300"]), "Orlando": ("12", ["36740"]), "Miami": ("12", ["33100"]),
    "Jacksonville": ("12", ["27260"]), "Charlotte": ("37", ["16740"]), "Raleigh": ("37", ["39580"]),
    "Nashville": ("47", ["34980"]), "Memphis": ("47", ["32820"]), "Birmingham": ("01", ["13820"]),
    "New Orleans": ("22", ["35380"]), "Oklahoma City": ("40", ["36420"]),
    "Richmond": ("51", ["40060"]), "Denver": ("08", ["19740"]), "Las Vegas": ("32", ["29820"]),
    "Salt Lake City": ("49", ["41620"]), "Los Angeles": ("06", ["31080"]),
    "San Diego": ("06", ["41740"]), "San Francisco": ("06", ["41860"]),
    "Sacramento": ("06", ["40900"]), "Seattle": ("53", ["42660"]), "Portland": ("41", ["38900"]),
    "New York": ("36", ["35620"]), "Boston": ("25", ["14460", "71650"]),
    "Philadelphia": ("42", ["37980"]), "Washington DC": ("11", ["47900"]),
    "Baltimore": ("24", ["12580"]), "Pittsburgh": ("42", ["38300"]), "Buffalo": ("36", ["15380"]),
    "Chicago": ("17", ["16980"]), "Detroit": ("26", ["19820"]), "Minneapolis": ("27", ["33460"]),
    "St. Louis": ("29", ["41180"]), "Kansas City": ("29", ["28140"]),
    "Milwaukee": ("55", ["33340"]), "Columbus": ("39", ["18140"]), "Cincinnati": ("39", ["17140"]),
    "Cleveland": ("39", ["17460"]), "Indianapolis": ("18", ["26900"]),
    "Louisville": ("21", ["31140"]),
}

# News coverage of each city's housing market (web content), from the GDELT DOC 2.0 API: U.S.
# English-language news, January 2017 on. The rules for testing it were written down before any
# news data was downloaded: docs/preregistration_news.md
NEWS_START = "2017-01-01"
NEWS_PHRASES = ['"housing market"', '"home prices"', '"home sales"', '"real estate market"',
                '"housing prices"']
# How each city is written in the news query, when not just its name. Variants are tried in order;
# a later one is used only if the ones before it return almost nothing (NEWS_MIN_ARTICLES).
NEWS_NAMES = {
    "Portland": ['"portland" oregon', '"portland"'],
    "Columbus": ['"columbus" ohio', '"columbus"'],
    "Richmond": ['"richmond" virginia', '"richmond"'],
    "Birmingham": ['"birmingham" alabama', '"birmingham"'],
    "Washington DC": ['"washington dc"', '"washington d.c."', '"district of columbia"'],
    "St. Louis": ['"st. louis"', '"st louis"', '"saint louis"'],
}
NEWS_MIN_ARTICLES = 5  # a metro averaging fewer matching articles a month counts as having no news data
NEWS_QUERIES = {
    m: [f'{name} ({" OR ".join(NEWS_PHRASES)}) sourcecountry:US sourcelang:english'
        for name in NEWS_NAMES.get(m, [f'"{m.lower()}"'])]
    for m in METROS
}

# Second news source: Media Cloud (free researcher news archive; needs a free API key in the
# MEDIACLOUD_API_KEY environment variable / GitHub secret). Same cities and housing phrases as above,
# in Media Cloud's search syntax. Rules: addendum 2 in docs/preregistration_news.md
MEDIACLOUD_COLLECTIONS = [34412234]  # United States - National
MC_NAMES = {
    "Portland": '"portland" AND oregon', "Columbus": '"columbus" AND ohio',
    "Richmond": '"richmond" AND virginia', "Birmingham": '"birmingham" AND alabama',
    "Washington DC": '("washington dc" OR "washington d.c." OR "district of columbia")',
    "St. Louis": '("st. louis" OR "st louis" OR "saint louis")',
}
MC_QUERIES = {m: f'{MC_NAMES.get(m, chr(34) + m.lower() + chr(34))} AND ({" OR ".join(NEWS_PHRASES)})'
              for m in METROS}
MC_MIN_HEADLINES = 10  # months with fewer scored headlines (3-month window) get no tone value
