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
    "Denver":         ("Denver, CO",         "US-CO-751", "CO"),
    "Las Vegas":      ("Las Vegas, NV",      "US-NV-839", "NV"),
    "Salt Lake City": ("Salt Lake City, UT", "US-UT-770", "UT"),
    "Seattle":        ("Seattle, WA",        "US-WA-819", "WA"),
    "San Diego":      ("San Diego, CA",      "US-CA-825", "CA"),
    "Boston":         ("Boston, MA",         "US-MA-506", "MA"),
    "Chicago":        ("Chicago, IL",        "US-IL-602", "IL"),
    "Columbus":       ("Columbus, OH",       "US-OH-535", "OH"),
    "Indianapolis":   ("Indianapolis, IN",   "US-IN-527", "IN"),
}

# Google Trends terms. Pulled together per metro (max 5 per request).
BUYER_TERMS = ["homes for sale", "mortgage calculator", "zillow", "realtor", "open house"]
# Model C: home-improvement interest, the Home Depot idea, via search.
HOME_IMPROVEMENT_TERMS = ["home depot", "lowes", "home inspection", "moving company"]

# National FRED series (free CSV, no key needed)
FRED_NATIONAL = {"mortgage_rate": "MORTGAGE30US"}
# State unemployment rate = "<ST>UR" on FRED, a reliable pattern for every state.

# For metro employment from the BLS: (state FIPS, CBSA code). Boston also tries its NECTA code.
BLS_AREAS = {
    "Phoenix": ("04", ["38060"]), "Dallas": ("48", ["19100"]), "Houston": ("48", ["26420"]),
    "Austin": ("48", ["12420"]), "San Antonio": ("48", ["41700"]), "Atlanta": ("13", ["12060"]),
    "Tampa": ("12", ["45300"]), "Orlando": ("12", ["36740"]), "Miami": ("12", ["33100"]),
    "Jacksonville": ("12", ["27260"]), "Charlotte": ("37", ["16740"]), "Raleigh": ("37", ["39580"]),
    "Nashville": ("47", ["34980"]), "Denver": ("08", ["19740"]), "Las Vegas": ("32", ["29820"]),
    "Salt Lake City": ("49", ["41620"]), "Seattle": ("53", ["42660"]), "San Diego": ("06", ["41740"]),
    "Boston": ("25", ["14460", "71650"]), "Chicago": ("17", ["16980"]),
    "Columbus": ("39", ["18140"]), "Indianapolis": ("18", ["26900"]),
}
