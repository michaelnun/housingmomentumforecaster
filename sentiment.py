"""Headline sentiment shared by the pipeline: VADER (a standard dictionary-based sentiment tool), as is
and with housing words added, plus a filter that keeps headlines about the city or housing.

A general tool reads "prices fall" as neutral or even positive and "bidding wars" as negative, so the
housing words shift those. It still can't always tell *what* is rising ("rates climb" vs. "prices climb").
"""
import pandas as pd

# Housing words for VADER (scale -4 to +4): positive = market strengthening, negative = weakening
HOUSING_TERMS = {
    **{w: 1.5 for w in ("rise", "rises", "rising", "rose", "climb", "climbs", "climbing", "climbed", "jump", "jumps",
                        "gain", "gains", "rebound", "rebounds", "recovery", "strong", "stronger", "hot", "appreciation")},
    **{w: 2.0 for w in ("surge", "surges", "surging", "soar", "soars", "soaring", "boom", "booming", "hottest")},
    **{w: 1.0 for w in ("bidding", "competitive")},
    **{w: -1.5 for w in ("fall", "falls", "falling", "fell", "drop", "drops", "dropping", "dropped", "decline",
                         "declines", "declining", "slide", "slides", "sliding", "slowdown", "stall", "stalls",
                         "stalled", "sluggish", "glut", "bubble", "correction")},
    **{w: -2.0 for w in ("slump", "slumps", "plunge", "plunges", "foreclosure", "foreclosures", "underwater")},
    **{w: -1.0 for w in ("cool", "cools", "cooling", "cooled", "slow", "slows", "slowing", "cut", "cuts")},
    "plummet": -2.5, "plummets": -2.5, "crash": -3.0,
    "war": 0.0, "wars": 0.0,  # "bidding wars" means a hot market
}


HOUSING_WORDS = (r"\b(home|homes|house|houses|housing|real estate|mortgages?|rents?|renters|rental|realtors?|"
                 r"listings?|propert(y|ies)|apartments?|condos?|buyers|sellers|homebuyers|homeowners)\b")
CITY_PATTERNS = {"Washington DC": r"washington|d\.c\.", "St. Louis": r"st\.? louis|saint louis",
                 "New York": r"new york|nyc|manhattan|brooklyn|queens|bronx|staten island|long island|westchester"}


def on_topic(h):
    """Keep headlines that mention the city or housing; GDELT also returns national stories that
    only mention the city in passing (earnings calls, politics)."""
    title = h["title"].astype(str).str.lower()
    housing = title.str.contains(HOUSING_WORDS, regex=True)
    city = pd.Series([bool(pd.Series([t]).str.contains(CITY_PATTERNS.get(m, m.lower()), regex=True).iloc[0])
                      for t, m in zip(title, h["metro"])], index=h.index)
    return housing | city


def scorers():
    """VADER as is, and VADER with housing words added. None if VADER isn't installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("vaderSentiment not installed; skipping headline sentiment (pip install vaderSentiment)")
        return None, None
    plain, housing = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
    housing.lexicon.update(HOUSING_TERMS)
    return plain, housing
