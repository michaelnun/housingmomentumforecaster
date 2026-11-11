"""Housing Momentum Forecaster dashboard.

    streamlit run dashboard.py

Links can open a page directly, e.g.  ?view=Metro&metro=Denver
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parent / "results"
STAGE_COLORS = {"Rising": "#2F6B45", "Turning up": "#4E9A6B", "Turning down": "#D08A1E",
                "Falling": "#B5533C"}
BAND_COLORS = {"Strong": "#2F6B45", "Average": "#8C958F", "Weak": "#B5533C"}
VERDICT_COLORS = {"Solid": "#2F6B45", "Holds, with caveats": "#D08A1E", "Not supported": "#B5533C"}
MARKET_COLORS = {"Seller's market": "#B5533C", "Balanced": "#8C958F", "Buyer's market": "#2F6B45"}
STAGE_ORDER = ["Rising", "Turning up", "Turning down", "Falling"]
INK, MUTED, GRID = "#1E2421", "#5E6863", "#E6E9E7"
VIEWS = ["Overview", "Metro", "Alerts", "Track record", "About"]

st.set_page_config(page_title="Housing Momentum Forecaster", page_icon="🏠", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; max-width: 1280px;}
  h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
  .sub {color: #5E6863; margin-top: -0.2rem; margin-bottom: 0.4rem;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px; color:white;
         font-size:0.85rem; font-weight:600; white-space:nowrap;}
  .card {border:1px solid #E6E9E7; border-radius:10px; padding:14px 16px; background:white;}
  .card h4 {margin:0 0 4px 0; font-size:0.95rem;}
  .card p {margin:0; color:#3A423E; font-size:0.92rem;}
  .big {font-size:2.3rem; font-weight:700; line-height:1.1; color:#1E2421;}
  .lbl {color:#5E6863; font-size:0.85rem;}
  .note {background:#F4F6F5; border-radius:8px; padding:8px 12px; margin:6px 0 10px 0;}
  div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
def _csv(name, **kw):
    f = RESULTS / name
    return pd.read_csv(f, **kw) if f.exists() else None


def _json(name):
    f = RESULTS / name
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data(ttl=3600)
def load():
    return {
        "hist": pd.read_csv(RESULTS / "index_history.csv", parse_dates=["date"]),
        "track": _json("track_record.json"), "copy": _json("copy.json"),
        "bands": _csv("score_bands.csv"), "stage_perf": _csv("stage_performance.csv"),
        "card": _csv("scorecard.csv", index_col=0), "sig": _csv("significance_tests.csv"),
        "card12": _csv("scorecard_12m.csv", index_col=0), "cal": _csv("calibration.csv"),
        "act": _json("activity_test.json"), "fnd": _json("findings.json"),
        "misses": _csv("biggest_misses.csv"), "cs": _csv("case_study_2022.csv", parse_dates=["date"]),
        "tp": _csv("turning_points.csv", parse_dates=["date"]),
        "sigt": _csv("signal_tests.csv"), "rob": _csv("robustness_unsmoothed.csv"),
        "stab": _csv("stability.csv"), "verdicts": _json("stability.json"),
    }


if not (RESULTS / "index_history.csv").exists():
    st.error("No results yet. Run run_all.bat (or step1 to step4) first.")
    st.stop()

D = load()
hist, track, copy = D["hist"], D["track"], D["copy"]
HAS12 = "outlook_12m" in hist and hist["outlook_12m"].notna().any() and "audience_12m" in copy
months = sorted(hist["date"].unique())


# ---------------------------------------------------------------- helpers
def pill(text, colors):
    return f"<span class='pill' style='background:{colors.get(text, '#8C958F')}'>{text}</span>"


def style(fig, h=420):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def ok(v):
    if v is None:
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def vs_avg(x):
    """+1.2 -> 'about 1.2% more than the typical big metro'"""
    if not ok(x):
        return "n/a"
    if abs(x) < 0.05:
        return "about the same as the typical big metro"
    return f"about {abs(x):.1f}% {'more' if x > 0 else 'less'} than the typical big metro"


def window(month, k):
    m = pd.Timestamp(month)
    return f"{m:%b %Y} → {(m + pd.DateOffset(months=k)):%b %Y}"


def band_of(score):
    for lo, hi, lbl in [(80, 101, "80-100"), (60, 80, "60-80"), (40, 60, "40-60"), (20, 40, "20-40")]:
        if lo <= score < hi:
            return lbl
    return "0-20"


def how_often(score, horizon):
    """'In past tests, metros scoring 80-100 beat the typical big metro 96% of the time...'"""
    cal = D["cal"]
    if cal is None or not ok(score):
        return ""
    b = band_of(score)
    r = cal[(cal["horizon_months"] == horizon) & (cal["band"] == b)]
    if r.empty:
        return ""
    return (f"In past tests, metros scoring {b.replace('-', '–')} beat the typical big metro "
            f"{r.iloc[0]['beat_avg_share']:.0%} of the time over the next {horizon} months.")


def outlook_caveat():
    """Plain caveat for the 12-month outlook when the untouched-metro test didn't confirm it."""
    v = D["verdicts"].get("The 12-month outlook beats 'the recent trend continues'")
    if v and v != "Solid":
        return ("Experimental: the 12-month outlook beat the simple 'trend continues' guess across all "
                "metros, but not clearly on the 23 metros added after the model was built.")
    return ""


def open_metro():
    """Callback for 'Open a metro': switch pages before the next run draws them."""
    m = st.session_state.get("jump")
    if m and m != "Choose a metro…":
        st.session_state["view"] = "Metro"
        st.session_state["metro"] = m
        st.session_state["jump"] = "Choose a metro…"


# ---------------------------------------------------------------- navigation + sidebar
qp = st.query_params
if "view" not in st.session_state:
    st.session_state["view"] = qp.get("view") if qp.get("view") in VIEWS else "Overview"
if "metro" not in st.session_state and qp.get("metro"):
    st.session_state["metro"] = qp.get("metro")  # checked against this month's metros below

with st.sidebar:
    st.markdown("### Settings")
    audience = st.radio("I work in", list(copy["audience"]), index=0)
    month = st.select_slider("Month", options=months, value=months[-1],
                             format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),
                             help="Look back at what the scores said in any past month.")
    st.divider()
    st.markdown("**How to read it**")
    st.caption("Each metro gets two scores from 0 to 100. **3-month score:** where home values are "
               "headed over the next 3 months. **12-month outlook:** the year ahead. 100 = expected "
               "to do best of all the metros, 50 = typical.")
    st.caption("Scores compare metros with each other. A high score means faster growth than the "
               "typical big metro, not a promise that prices will rise. Not financial advice.")
    st.caption("Data: Zillow (home values, rents, market stats), Realtor.com listing views, "
               "Google Trends, FRED, Census, BLS. Refreshed monthly.")

now = hist[hist["date"] == month].sort_values("rank")
as_of = pd.Timestamp(month).strftime("%B %Y")
latest = pd.Timestamp(months[-1])
n_metros = now["metro"].nunique()

st.markdown("# Housing Momentum Forecaster")
upd = f" · updated {track['updated']}" if track.get("updated") else ""
st.markdown(f"<div class='sub'>Which of {n_metros} big U.S. housing markets are set to outperform · "
            f"data through {latest:%B %Y}{upd}</div>", unsafe_allow_html=True)
if pd.Timestamp(month) != latest:
    st.info(f"You're looking back at **{as_of}**. Move the Month slider to the end for the latest scores.")
if track.get("demo"):
    st.warning("Prototype running on **simulated data**. Every number here is a placeholder.", icon="⚠️")

view = st.segmented_control("Page", VIEWS, key="view", label_visibility="collapsed") or "Overview"
st.query_params["view"] = view
if view != "Metro" and "metro" in st.query_params:
    del st.query_params["metro"]

# ================================================================ Overview
if view == "Overview":
    has12_now = HAS12 and now["outlook_12m"].notna().any()
    horizons = ["Next 3 months"] + (["Next 12 months"] if has12_now else [])
    if HAS12 and not has12_now:
        st.caption("The 12-month outlook starts in 2021 (it needs a year of tested history first).")
    hz = st.segmented_control("Horizon", horizons, default="Next 3 months") or "Next 3 months"
    twelve = hz == "Next 12 months"
    sc, fc, px = ("outlook_12m", "forecast_12m", "price_vs_avg_12m") if twelve else \
        ("score", "forecast", "price_vs_avg")
    k = 12 if twelve else 3
    view_df = now.dropna(subset=[sc]).sort_values(sc, ascending=False)
    if view_df.empty:
        st.info("No scores for this month.")
        st.stop()
    st.caption(f"Forecast window: **{window(month, k)}**. Ranked by the "
               f"{'12-month outlook' if twelve else '3-month score'}."
               + (f" {outlook_caveat()}" if twelve and outlook_caveat() else ""))

    top = view_df.iloc[0]
    changed = now[now["prev_stage"].notna() & (now["prev_stage"] != now["stage"])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Top metro · {top[sc]:.0f}", top["metro"])
    if twelve:
        c2.metric("Strong year-ahead outlook", f"{(view_df['outlook_band'] == 'Strong').sum()} of {len(view_df)}",
                  help="Outlook of 60 or more: expected to beat the typical big metro over the year.")
    else:
        c2.metric("Turning up", f"{(now['stage'] == 'Turning up').sum()} of {len(now)}",
                  help="Prices have lagged, but expected to start beating other metros.")
    c3.metric("Stage changes this month", len(changed))
    cal = D["cal"]
    if cal is not None:
        r80 = cal[(cal["horizon_months"] == k) & (cal["band"] == "80-100")]
        if len(r80):
            c4.metric("Top scores (80+) right", f"{r80.iloc[0]['beat_avg_share']:.0%}",
                      help=f"In past tests, metros scoring 80–100 beat the typical big metro this "
                           f"often over the next {k} months.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Market map")
        st.caption(f"Left to right: how home values did over the last {k} months. Bottom to top: "
                   f"what's expected over the next {k}. Both compared with the typical big metro.")
        d = view_df.dropna(subset=[px, fc])
        xr_ = max(0.5, d[px].abs().max() * 1.2)
        yr = max(0.5, d[fc].abs().max() * 1.25)
        quads = (["Keeps leading", "Catching up", "Cooling", "Lagging"] if twelve else STAGE_ORDER)
        signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        fig = go.Figure()
        for name, (sx, sy), col in zip(quads, signs, [STAGE_COLORS[s] for s in STAGE_ORDER]):
            fig.add_shape(type="rect", x0=0, x1=sx * xr_, y0=0, y1=sy * yr, line_width=0,
                          fillcolor=col, opacity=0.07, layer="below")
            fig.add_annotation(x=sx * xr_ * 0.97, y=sy * 0.03 * yr, text=f"<b>{name}</b>", showarrow=False,
                               xanchor="right" if sx > 0 else "left",
                               yanchor="bottom" if sy > 0 else "top", font=dict(color=col, size=14))
        fig.add_vline(x=0, line_color="#B8BFBB", line_width=1)
        fig.add_hline(y=0, line_color="#B8BFBB", line_width=1)
        show_names = set(d.head(6)["metro"]) | set(d.tail(3)["metro"])
        colors = [BAND_COLORS.get(b, "#8C958F") for b in d["outlook_band"]] if twelve else \
            [STAGE_COLORS[s] for s in d["stage"]]
        fig.add_trace(go.Scatter(
            x=d[px], y=d[fc], mode="markers+text", text=d["metro"],
            texttemplate=[m if m in show_names else "" for m in d["metro"]],
            # alternate label above / below so neighbors don't print on top of each other
            textposition=["top center" if i % 2 == 0 else "bottom center" for i in range(len(d))],
            textfont=dict(size=11, color=INK),
            marker=dict(size=8 + d[sc] / 8, color=colors, line=dict(color="white", width=1)),
            customdata=d[[sc]],
            hovertemplate="<b>%{text}</b><br>Score %{customdata[0]:.0f}<br>Last " + str(k) +
                          " months %{x:+.1f}% vs. typical<br>Next " + str(k) +
                          " months (expected) %{y:+.1f}% vs. typical<extra></extra>"))
        fig.update_xaxes(range=[-xr_, xr_], title=f"Last {k} months vs. typical metro (%)")
        fig.update_yaxes(range=[-yr, yr], title=f"Next {k} months, expected vs. typical (%)")
        st.plotly_chart(style(fig, 520), width="stretch")

    with right:
        st.markdown("#### Ranking")
        tcols = ["metro", "score"] + (["outlook_12m"] if HAS12 else []) + ["stage"] + \
                (["market_type"] if "market_type" in view_df else [])
        table = view_df[tcols].copy()
        table.insert(0, "rank", range(1, len(table) + 1))
        if "market_type" in table:
            table["market_type"] = table["market_type"].fillna("n/a").str.replace(" market", "", regex=False)
        cfg = {
            "rank": st.column_config.NumberColumn("#", width=36),
            "metro": st.column_config.TextColumn("Metro"),
            "score": st.column_config.ProgressColumn("3-month", min_value=0, max_value=100, format="%d",
                                                     width="small", help="3-month score (0–100)"),
            "outlook_12m": st.column_config.ProgressColumn("12-month", min_value=0, max_value=100,
                                                           format="%d", width="small",
                                                           help="12-month outlook (0–100)"),
            "stage": st.column_config.TextColumn("Stage", width=105),
            "market_type": st.column_config.TextColumn("Market now", width=80,
                                                       help="Who has the upper hand right now: buyers, "
                                                            "sellers, or balanced"),
        }
        st.dataframe(table, column_config=cfg, hide_index=True, width="stretch", height=520)
        st.selectbox("Open a metro", ["Choose a metro…"] + sorted(now["metro"]), key="jump",
                     on_change=open_metro)

    dcols = ["metro", "score"] + (["outlook_12m"] if HAS12 else []) + ["stage", "forecast"] + \
            (["forecast_12m"] if HAS12 else []) + ["reason"] + (["note"] if "note" in view_df else [])
    dl = view_df[dcols].copy()
    dl.insert(0, "rank", range(1, len(dl) + 1))
    dl["todo3"] = dl["stage"].map(copy["audience"][audience])
    if HAS12:
        dl["todo12"] = view_df["outlook_band"].map(copy["audience_12m"][audience]).values
    dl.insert(0, "data_month", pd.Timestamp(month).strftime("%Y-%m"))
    dl = dl.rename(columns={"data_month": "Data month", "rank": "Rank", "metro": "Metro",
                            "score": "3-month score", "outlook_12m": "12-month outlook", "stage": "Stage",
                            "forecast": "Next 3 months vs. typical metro (%)",
                            "forecast_12m": "Next 12 months vs. typical metro (%)", "reason": "Why",
                            "note": "Now vs. year ahead", "todo3": f"Next 3 months ({audience})",
                            "todo12": f"Year ahead ({audience})"})
    st.download_button("Download this ranking (CSV, opens in Excel)", dl.to_csv(index=False),
                       file_name=f"housing_momentum_{pd.Timestamp(month):%Y_%m}.csv", mime="text/csv")

# ================================================================ Metro
elif view == "Metro":
    metros = sorted(now["metro"])
    wanted = st.session_state.get("metro")
    if wanted not in metros:
        if wanted:
            st.warning(f"{wanted} has no score for {as_of} (some of its data is missing). "
                       f"Showing {now.iloc[0]['metro']} instead.")
        st.session_state["metro"] = now.iloc[0]["metro"]
    pick = st.selectbox("Metro", metros, key="metro")
    st.query_params["metro"] = pick
    r = now[now["metro"] == pick].iloc[0]
    series = hist[(hist["metro"] == pick) & (hist["date"] <= month)].sort_values("date")
    st.markdown(f"## {pick}")
    st.caption("Tip: this page's web address opens straight to this metro, so you can send it to a "
               "client or colleague.")

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown(f"<div class='lbl'>NEXT 3 MONTHS · {window(month, 3)}</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='big'>{r['score']:.0f}</span> <span class='lbl'>/ 100 · rank "
                        f"{int(r['rank'])} of {len(now)}</span>", unsafe_allow_html=True)
            st.markdown(f"{pill(r['stage'], STAGE_COLORS)} &nbsp; {copy['stages'][r['stage']]}",
                        unsafe_allow_html=True)
            st.markdown(f"Home values expected to grow **{vs_avg(r['forecast'])}**.")
            st.caption(how_often(r["score"], 3))
    with b:
        with st.container(border=True):
            if HAS12 and ok(r.get("outlook_12m")):
                rk = int((now["outlook_12m"] > r["outlook_12m"]).sum() + 1)
                st.markdown(f"<div class='lbl'>NEXT 12 MONTHS · {window(month, 12)}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<span class='big'>{r['outlook_12m']:.0f}</span> <span class='lbl'>/ 100 · rank "
                            f"{rk} of {now['outlook_12m'].notna().sum()}</span>", unsafe_allow_html=True)
                st.markdown(f"{pill(r['outlook_band'], BAND_COLORS)} &nbsp; year-ahead outlook",
                            unsafe_allow_html=True)
                st.markdown(f"Home values expected to grow **{vs_avg(r['forecast_12m'])}**.")
                st.caption(how_often(r["outlook_12m"], 12))
                if outlook_caveat():
                    st.caption(outlook_caveat())
            else:
                st.markdown("<div class='lbl'>NEXT 12 MONTHS</div>", unsafe_allow_html=True)
                st.write("No 12-month outlook for this metro yet (missing data).")
    if ok(r.get("note")) and r.get("note"):
        st.markdown(f"<div class='note'>{r['note']}</div>", unsafe_allow_html=True)
    if ok(r.get("data_gaps")) and r.get("data_gaps"):
        st.caption(f"Data note: {r['data_gaps']}. Scores for this metro are a little less certain.")

    st.markdown(f"#### What it means for {audience.lower()}")
    st.markdown(f"- **Next 3 months:** {copy['audience'][audience][r['stage']]}")
    if HAS12 and ok(r.get("outlook_band")):
        st.markdown(f"- **Year ahead:** {copy['audience_12m'][audience][r['outlook_band']]}")
    with st.expander("For other industries"):
        for aud, lines in copy["audience"].items():
            if aud != audience:
                extra = (f" Year ahead: {copy['audience_12m'][aud][r['outlook_band']]}"
                         if HAS12 and ok(r.get("outlook_band")) else "")
                st.markdown(f"- **{aud}:** {lines[r['stage']]}{extra}")

    st.markdown("#### Market right now")
    st.caption(f"Plain numbers for {pd.Timestamp(month):%B %Y}, no forecasting. A few of the newest numbers "
               "are carried forward from the month before while Zillow catches up.")
    m1, m2, m3 = st.columns(3)
    if ok(r.get("zhvi")):
        m1.metric("Typical home value", f"${r['zhvi']:,.0f}",
                  f"{r['home_value_yoy_pct']:+.1f}% vs. a year ago" if ok(r.get("home_value_yoy_pct")) else None)
    if ok(r.get("typical_rent")):
        m2.metric("Typical rent", f"${r['typical_rent']:,.0f}/mo",
                  f"{r['rent_yoy_pct']:+.1f}% vs. a year ago" if ok(r.get("rent_yoy_pct")) else None)
    if ok(r.get("market_type")):
        m3.markdown(f"<div class='lbl'>Who has the upper hand</div>{pill(r['market_type'], MARKET_COLORS)}",
                    unsafe_allow_html=True)
        m3.caption("Rough guide: seller's market = homes sell at or above asking on average (or 40%+ "
                   "sell above asking); buyer's market = under 98% of asking, with many price cuts or "
                   "slow sales.")
    m4, m5, m6, m7 = st.columns(4)
    if ok(r.get("days_to_pending")):
        last = r.get("days_to_pending_last_year")
        m4.metric("Days to go under contract", f"{r['days_to_pending']:.0f}",
                  f"{r['days_to_pending'] - last:+.0f} vs. a year ago" if ok(last) else None,
                  delta_color="inverse")
    if ok(r.get("sale_to_list_pct")):
        m5.metric("Sale price vs. asking", f"{r['sale_to_list_pct']:.1f}%")
    if ok(r.get("price_cut_share_pct")):
        m6.metric("Listings with a price cut", f"{r['price_cut_share_pct']:.0f}%")
    if ok(r.get("for_sale_vs_last_year_pct")):
        m7.metric("Homes for sale vs. a year ago", f"{r['for_sale_vs_last_year_pct']:+.0f}%")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Why")
        st.write(r["reason"])
        comps = [c for c in track.get("components", []) if c in r and ok(r[c])]
        fig = go.Figure(go.Bar(
            x=[r[c] for c in comps], y=comps, orientation="h",
            marker_color=["#2F6B45" if r[c] >= 50 else "#C9CFCB" for c in comps],
            text=[f"{r[c]:.0f}" for c in comps], textposition="outside"))
        fig.add_vline(x=50, line_dash="dot", line_color="#8C958F")
        fig.update_xaxes(range=[0, 112], title="vs. other metros (50 = typical)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 250), width="stretch")
    with right:
        st.markdown("#### Scores over the last 5 years")
        recent = series[series["date"] >= pd.Timestamp(month) - pd.DateOffset(years=5)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["date"], y=recent["score"], mode="lines", name="3-month score",
                                 line=dict(color="#2F6B45", width=2.5),
                                 hovertemplate="%{x|%b %Y}: %{y:.0f}<extra>3-month</extra>"))
        if HAS12:
            fig.add_trace(go.Scatter(x=recent["date"], y=recent["outlook_12m"], mode="lines",
                                     name="12-month outlook", line=dict(color="#D08A1E", width=2, dash="dot"),
                                     hovertemplate="%{x|%b %Y}: %{y:.0f}<extra>12-month</extra>"))
        fig.add_hline(y=50, line_dash="dot", line_color="#8C958F")
        fig.update_yaxes(range=[0, 100], title="Score")
        style(fig, 280).update_layout(showlegend=True, legend=dict(orientation="h", y=1.15, x=0))
        st.plotly_chart(fig, width="stretch")

    hcols = [c for c in ["date", "rank", "score", "stage", "forecast", "outlook_12m", "forecast_12m",
                         "realized_vs_avg_pct", "realized_12m_pct", "zhvi", "home_value_yoy_pct",
                         "typical_rent", "rent_yoy_pct", "days_to_pending", "sale_to_list_pct",
                         "price_cut_share_pct", "market_type", "reason"] if c in series]
    hd = series[hcols].copy()
    hd["date"] = hd["date"].dt.strftime("%Y-%m")
    hd = hd.rename(columns={
        "date": "Month", "rank": "3-month rank", "score": "3-month score", "stage": "Stage",
        "forecast": "Expected next 3 months vs. typical metro (%)", "outlook_12m": "12-month outlook",
        "forecast_12m": "Expected next 12 months vs. typical metro (%)",
        "realized_vs_avg_pct": "Actual next 3 months vs. typical (%)",
        "realized_12m_pct": "Actual next 12 months vs. typical (%)", "zhvi": "Typical home value ($)",
        "home_value_yoy_pct": "Home value vs. a year ago (%)", "typical_rent": "Typical rent ($)",
        "rent_yoy_pct": "Rent vs. a year ago (%)", "days_to_pending": "Days to go under contract",
        "sale_to_list_pct": "Sale price vs. asking (%)", "price_cut_share_pct": "Listings with a price cut (%)",
        "market_type": "Market", "reason": "Why"})
    st.download_button(f"Download {pick}'s full history (CSV)", hd.to_csv(index=False),
                       file_name=f"housing_momentum_{pick.lower().replace(' ', '_').replace('.', '')}.csv",
                       mime="text/csv")

# ================================================================ Alerts
elif view == "Alerts":
    st.markdown(f"#### 3-month stage changes in {as_of}")
    changed = now[now["prev_stage"].notna() & (now["prev_stage"] != now["stage"])]
    if changed.empty:
        st.info("No metro changed stage this month.")
    for _, x in changed.iterrows():
        st.markdown(f"<div class='card' style='margin-bottom:8px'><h4>{x['metro']}: "
                    f"{pill(x['prev_stage'], STAGE_COLORS)} → {pill(x['stage'], STAGE_COLORS)}</h4>"
                    f"<p>{copy['audience'][audience][x['stage']]}</p></div>", unsafe_allow_html=True)
    if HAS12:
        prev_m = [m for m in months if m < month]
        if prev_m:
            prev = hist[hist["date"] == prev_m[-1]].set_index("metro")["outlook_band"]
            flips = now[now["outlook_band"].notna()].copy()
            flips["prev_band"] = flips["metro"].map(prev)
            flips = flips[flips["prev_band"].notna() & (flips["prev_band"] != flips["outlook_band"])]
            st.markdown("#### Year-ahead outlook changes")
            if flips.empty:
                st.info("No metro's year-ahead outlook changed band this month.")
            for _, x in flips.iterrows():
                st.markdown(f"<div class='card' style='margin-bottom:8px'><h4>{x['metro']}: "
                            f"{pill(x['prev_band'], BAND_COLORS)} → {pill(x['outlook_band'], BAND_COLORS)}</h4>"
                            f"<p>{copy['audience_12m'][audience][x['outlook_band']]}</p></div>",
                            unsafe_allow_html=True)
    st.markdown("#### Biggest 3-month score moves")
    mv = now.dropna(subset=["score_chg"]).sort_values("score_chg")
    mv = pd.concat([mv.head(5), mv.tail(5)]).drop_duplicates("metro").sort_values("score_chg")
    fig = go.Figure(go.Bar(x=mv["score_chg"], y=mv["metro"], orientation="h",
                           marker_color=["#2F6B45" if v > 0 else "#B5533C" for v in mv["score_chg"]],
                           text=[f"{v:+.0f}" for v in mv["score_chg"]], textposition="outside"))
    fig.update_xaxes(title="Change in 3-month score vs. last month")
    st.plotly_chart(style(fig, 360), width="stretch")

# ================================================================ Track record
elif view == "Track record":
    st.markdown("#### How often is it right?")
    st.caption("Every score was made month by month using only data available at the time, then checked "
               "against what home values actually did. 'Beat' = grew faster than the typical big metro.")
    cal = D["cal"]
    if cal is not None:
        t = cal.pivot(index="band", columns="horizon_months", values="beat_avg_share")
        t = t.reindex(["80-100", "60-80", "40-60", "20-40", "0-20"])
        t.index = [f"Score {b.replace('-', '–')}" for b in t.index]
        t.columns = [f"Beat the typical metro, next {c} months" for c in t.columns]
        st.dataframe(t.style.format("{:.0%}"), width="stretch")

    stab, verdicts = D["stab"], D["verdicts"]
    if stab is not None and verdicts:
        st.markdown("#### How solid is each finding?")
        st.caption("Each finding is re-checked several ways so one lucky slice of data can't carry it: "
                   "Zillow's raw values as well as its main ones; the 23 metros added after every modeling "
                   "choice was made (an untouched test); early vs. recent years; and dropping each metro in "
                   "turn. **Solid** = significant overall and better in every check. Numbers are the change in "
                   "ranking accuracy; ✓ = statistically significant (p < 0.05).")
        for finding, v in verdicts.items():
            d = stab[stab["finding"] == finding].copy()
            d["cell"] = [f"{g:+.3f}{' ✓' if p < 0.05 else ''}" for g, p in zip(d["gain"], d["p"])]
            t = d.pivot(index="check", columns="version", values="cell")
            order = ["All metros", "Original 22 metros", "New metros (untouched test)", "Early years",
                     "Recent years", "Drop any one metro"]
            t = t.reindex([o for o in order if o in t.index])
            t = t.rename(index={"Drop any one metro": "Drop any one metro (worst case)"})
            with st.expander(f"{finding}: {v}", expanded=False):
                st.markdown(pill(v, VERDICT_COLORS), unsafe_allow_html=True)
                st.dataframe(t, width="stretch")
                dr = d[d["check"] == "Drop any one metro"]
                if len(dr) and "share_significant" in dr:
                    st.caption("Still significant after dropping any one metro: " + ", ".join(
                        f"{r.version.lower()} {r.share_significant:.0%} of the time" for r in dr.itertuples()))
        pre = Path(__file__).resolve().parent / "docs" / "preregistration.md"
        if pre.exists():
            with st.expander("The tests were written down before the 45-metro results came in"):
                st.markdown(pre.read_text())

    live = track.get("live", {})
    if live:
        st.markdown("#### Live record: scores as published")
        st.caption("Each month's published scores are saved and never changed, so this record can't be "
                   "improved by re-running the model later.")
        if live.get("months_scored_3m"):
            l1, l2, l3 = st.columns(3)
            l1.metric("Months with results", live["months_scored_3m"])
            l2.metric("Ranking accuracy (3-month)", f"{live['rank_corr_3m']:.2f}")
            l3.metric("Top 5 beat the typical metro", f"{live['top5_beat_3m']:.0%}")
        else:
            st.markdown(f"Logging since {live.get('first_published', 'now')} "
                        f"({live.get('months_logged', 0)} month(s) so far). The first 3-month results come in "
                        f"once Zillow publishes {live.get('first_results_3m', 'three months later')} data.")

    st.markdown("#### 3-month score")
    k1, k2, k3, k4 = st.columns(4)
    if "rank_corr" in track:
        k1.metric("Ranking accuracy", f"{track['rank_corr']:.2f}",
                  help="How well the order of metros matched what happened (1 = perfect, 0 = random).")
    k2.metric("Top 5 beat the typical metro", f"{track['top5_hit_rate']:.0%}")
    k3.metric("Top-5 extra growth", f"{track['top5_excess_per_year_pct']:+.1f}%/yr",
              help="How much faster home values in the top 5 grew than the typical big metro, annualized. "
                   "Not an investment return.")
    k4.metric("Months tested", track["months_tested"])
    bands, sp = D["bands"], D["stage_perf"]
    l, rr = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=bands["band"], y=bands["avg_next_3m_vs_avg_pct"],
                               marker_color=["#B5533C" if v < 0 else "#2F6B45" for v in bands["avg_next_3m_vs_avg_pct"]],
                               text=[f"{v:+.2f}%" for v in bands["avg_next_3m_vs_avg_pct"]], textposition="outside"))
        fig.add_hline(y=0, line_color="#8C958F")
        fig.update_xaxes(title="3-month score", type="category")
        fig.update_yaxes(title="Next 3 months vs. typical metro (%)")
        st.markdown("**Higher score, better next 3 months**")
        st.plotly_chart(style(fig, 320), width="stretch")
    with rr:
        sp = sp.set_index("stage").reindex(STAGE_ORDER).reset_index()
        fig = go.Figure(go.Bar(x=sp["stage"], y=sp["mean"], marker_color=[STAGE_COLORS[s] for s in sp["stage"]],
                               text=[f"{v:+.2f}%" for v in sp["mean"]], textposition="outside"))
        fig.add_hline(y=0, line_color="#8C958F")
        fig.update_yaxes(title="Next 3 months vs. typical metro (%)")
        st.markdown("**What happened next, by stage**")
        st.plotly_chart(style(fig, 320), width="stretch")

    o = track.get("outlook")
    if o:
        st.markdown("#### 12-month outlook")
        st.caption(f"Tested {o['first_month']} to {o['last_month']} ({o['months_tested']} months). A year out "
                   "is harder to call, so accuracy is lower than the 3-month score. On Zillow's main "
                   "(smoothed) values it's well above the simple 'trend continues' guess; the stricter test "
                   "on raw values is in the table further down.")
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Ranking accuracy", f"{o['rank_corr']:.2f}",
                  f"vs. {o['trend_rank_corr']:.2f} for 'trend continues'", delta_color="off")
        j2.metric("Top 5 beat the typical metro", f"{o['top5_beat_avg']:.0%}")
        j3.metric("Bottom 5 trailed it", f"{o['bottom5_trail_avg']:.0%}")
        j4.metric("Top-5 extra growth", f"{o['top5_excess_pct']:+.1f}%/yr")
        if outlook_caveat():
            st.caption(outlook_caveat())
        if "p_search_helps" in o:
            uses = o.get("model") == "Y12"
            sigw = "a statistically significant gain" if o["p_search_helps"] < 0.05 else "no real gain"
            st.markdown(f"**Does Google search help the year-ahead outlook?** Adding nationwide search interest "
                        f"(people across the U.S. searching '[city] homes for sale') on top of housing data, "
                        f"page views and rent changed ranking accuracy by {o['search_gain']:+.3f}, better in "
                        f"{o['search_better_share']:.0%} of months: {sigw} (p = {o['p_search_helps']:.2f}). "
                        + ("So the outlook uses it." if uses else
                           "So the outlook doesn't use it. Listing page views already capture the same "
                           "online interest more directly."))
        c12 = D["card12"]
        if c12 is not None:
            show = c12.set_index("label")[["rank_corr", "top5_beat_avg", "bottom5_trail_avg"]]
            show.columns = ["Ranking accuracy", "Top 5 beat typical", "Bottom 5 trailed"]
            show.index.name = "12-month model"
            st.dataframe(show.style.format({"Ranking accuracy": "{:.2f}", "Top 5 beat typical": "{:.0%}",
                                            "Bottom 5 trailed": "{:.0%}"}), width="stretch")

    card, sig = D["card"], D["sig"]
    if card is not None:
        st.markdown("#### Which data drives the 3-month score?")
        st.markdown("Adding Google search to the housing data did **not** improve the 3-month score. "
                    "Realtor.com listing page views didn't change it on Zillow's main values but improved it "
                    "on the stricter raw-value test below, so the 3-month score uses housing data plus page "
                    "views. (That choice was made after seeing these test results, so treat the gain as "
                    "promising rather than proven.)")
        names = {"Naive": "Trend continues (baseline)", "A": "Housing data only", "B": "+ Google search",
                 "C": "+ Home Depot / Lowe's searches", "C_gbm": "Gradient boosting (check)",
                 "P": "+ Realtor.com page views (used)"}
        show = card.rename(index=names)[["rank_corr", "top5_hit_rate", "direction_acc", "MAE_pct"]]
        show.columns = ["Ranking accuracy", "Top 5 matched actual top 5", "Direction right", "Avg error (%)"]
        st.dataframe(show.style.format("{:.2f}"), width="stretch")
        if sig is not None:
            st.caption(" · ".join(f"{x.comparison}: {x.verdict} (p={x.p_value_one_sided:.3f})"
                                  for x in sig.itertuples()))

    sigt = D["sigt"]
    if sigt is not None and len(sigt):
        st.markdown("#### Which extra data helps, and how far ahead?")
        st.caption("Each signal added to the housing data and tested month by month, 3, 6 and 12 months "
                   "ahead. Change in ranking accuracy; ✓ = statistically significant (p < 0.05). With this "
                   "many tests one or two ✓ could be luck, so a signal that helps at several horizons is "
                   "more convincing.")
        piv = sigt.assign(cell=[f"{g:+.3f}{' ✓' if h else ''}" for g, h in zip(sigt["gain"], sigt["helps"])])
        piv = piv.pivot(index="signal", columns="horizon_months", values="cell")
        piv = piv.reindex(list(dict.fromkeys(sigt["signal"])))
        piv.columns = [f"{c} months" for c in piv.columns]
        st.dataframe(piv, width="stretch")

    rob = D["rob"]
    if rob is not None and len(rob):
        st.markdown("#### Stricter test: Zillow's unsmoothed home values")
        st.caption("Zillow smooths its main home-value series, which makes short-term trends look more "
                   "predictable. These re-run the key tests on the raw series. Ranking accuracy; p-values "
                   "test whether a gain is real.")
        rv = rob.rename(columns={
            "horizon_months": "Months ahead", "months_tested": "Months tested", "naive": "Trend continues",
            "housing_model": "Housing data", "with_nationwide_search": "+ nationwide search",
            "p_model_vs_naive": "p (beats trend)", "nat_without": "Without search",
            "p_nationwide_helps": "p (search helps)", "outlook_model": "12-month outlook model",
            "outlook_naive": "Trend continues (same rows)", "p_outlook_vs_naive": "p (outlook beats trend)",
            "pageviews_without": "Without page views", "pageviews_with": "+ page views",
            "p_pageviews_helps": "p (page views help)", "rent_without": "Without rent",
            "rent_with": "+ rent growth", "p_rent_helps": "p (rent helps)"})
        st.dataframe(rv, hide_index=True, width="stretch")

    cs, tp, fnd = D["cs"], D["tp"], D["fnd"]
    if cs is not None:
        st.markdown("#### 2022 turning point")
        st.caption("3-month score through the 2022 rate shock. Dashed line: when home values peaked (Sun Belt) "
                   "or started beating the typical metro (Chicago, Boston).")
        from plotly.subplots import make_subplots
        order = [m for m in ["Phoenix", "Austin", "Las Vegas", "Chicago", "Boston", "Salt Lake City"]
                 if m in set(cs["metro"])]
        fig = make_subplots(rows=2, cols=3, subplot_titles=order, shared_yaxes=True,
                            vertical_spacing=0.16, horizontal_spacing=0.05)
        for i, m in enumerate(order):
            row, col = i // 3 + 1, i % 3 + 1
            d = cs[cs["metro"] == m]
            fig.add_hrect(y0=50, y1=105, fillcolor="#2F6B45", opacity=0.05, line_width=0, row=row, col=col)
            fig.add_trace(go.Scatter(x=d["date"], y=d["score"], mode="lines", line=dict(color=INK, width=2),
                                     hovertemplate="%{x|%b %Y}: %{y:.0f}<extra></extra>"), row=row, col=col)
            if m in ("Chicago", "Boston") and tp is not None:
                f = tp[(tp["metro"] == m) & (tp["direction"] == "up") & (tp["date"] >= "2022-06-01")]
                mark, colr = (f["date"].iloc[0] if len(f) else None), "#2F6B45"
            else:
                mark, colr = d.loc[d["zhvi"].idxmax(), "date"], "#B5533C"
            if mark is not None:
                fig.add_vline(x=mark, line_dash="dash", line_color=colr, row=row, col=col)
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(style(fig, 460), width="stretch")
        tps = fnd.get("turning_points", {})
        if tps:
            st.markdown(f"Across all metros, the 3-month score had already crossed 50 in the right direction "
                        f"before **{tps['flagged_before']:.0%}** of {tps['turns']} lasting turns in a metro's "
                        f"price trend (typically {tps['median_months_early_up']:.0f} months early for upturns, "
                        f"{tps['median_months_early_down']:.0f} for downturns). It missed {tps['missed']:.0%}.")
    if fnd.get("periods"):
        st.markdown("#### When it works, and when it misses")
        pr = pd.DataFrame(fnd["periods"]).rename(columns={
            "period": "Period", "months": "Months", "ranking_accuracy": "Ranking accuracy",
            "worst_month": "Worst month", "top5_beat_avg": "Top 5 beat typical"})
        st.dataframe(pr, hide_index=True, width="stretch",
                     column_config={"Top 5 beat typical": st.column_config.NumberColumn(format="percent")})
        misses = D["misses"]
        if misses is not None:
            st.caption("Biggest misses came in fast-moving markets, when prices swung harder than any recent "
                       "trend suggested:")
            mm = misses.rename(columns={"date": "Month", "metro": "Metro", "forecast": "Expected (%)",
                                        "realized_vs_avg_pct": "Actual (%)", "miss": "Miss (%)"})
            st.dataframe(mm.head(5).round(2), hide_index=True, width="stretch")

# ================================================================ About
else:
    o = track.get("outlook", {})
    st.markdown("#### What this is")
    st.markdown(f"""
A monthly ranking of {n_metros} big U.S. housing markets by where home values are headed, built for
**real estate firms, home-improvement retailers and investors**.

- **3-month score (0–100):** how home values are expected to do over the next 3 months compared with
  the typical big metro. 100 = best outlook of all the metros, 50 = typical.
- **12-month outlook (0–100, experimental):** the same idea for the year ahead.
- **Stage:** compares where prices have been with where they're expected to go: Rising, Turning up,
  Turning down or Falling. *Turning up* and *Turning down* are the early warnings.
- **Market right now:** plain numbers (home value, rent, days to sell, price cuts), no forecasting.
""")
    st.markdown("#### What it is not")
    st.markdown("""
- **Not a price forecast.** A high score means faster growth than other big metros, which can still mean
  prices fall if the whole country falls.
- **Not financial advice.** It's one input for decisions, based on public data.
- **Not neighborhood-level.** Each score covers a whole metro area; neighborhoods within it differ.
""")
    st.markdown("#### How it works")
    st.markdown("""
1. **Data**, every month: Zillow (home values, rents, how fast homes sell, inventory, price cuts),
   Realtor.com listing page views, Google Trends searches, mortgage rates (FRED), unemployment and jobs
   (FRED, BLS) and home-improvement store sales (Census).
2. **Compare each metro with the typical one**, so national swings like mortgage rates drop out.
3. **Model:** a simple, explainable regression. The 3-month score uses housing data plus Realtor.com
   listing page views. The 12-month outlook adds rent growth, which only helps further out.
   Google searches were tested at every horizon and kept out: once listing views are in, they add nothing.
4. **Tested honestly:** for every month since 2020, the model learned only from data available at the
   time, then predicted. The Track record page shows how often it was right.
""")
    st.markdown("#### Limits")
    st.markdown(f"""
- Zillow publishes with a lag of about a month, and a few market numbers lag one more; the newest month
  carries those forward.
- Zillow smooths its home-value series, which flatters short-term accuracy. The stricter test on raw
  values gives clearly lower numbers (Track record page).
- Search data comes from Google Trends samples, which shift a little every download; city names like
  Portland or Columbus also pick up searches for other places with the same name.
- The choices to add page views to the 3-month score and search to the 12-month outlook were made
  after seeing results for the original 22 metros. The 23 metros added later were an untouched test,
  with the rules written down beforehand (Track record page). Page views passed. Search failed and was
  taken out. The 12-month outlook's edge over the simple trend didn't clearly hold on the new metros,
  so it's marked experimental.
- The 12-month outlook has fewer test years ({o.get('months_tested', 'about 56')} months, from 2021),
  so treat it as a guide.
- Realtor.com changed how it counts page views in late 2021.
""")
    st.caption("Web Analytics, Fordham University Gabelli School of Business, Fall 2026. Code and data: "
               "github.com/michaelnun/housingmomentumforecaster")
