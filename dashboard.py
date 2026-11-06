"""Housing Momentum Forecaster dashboard.

    streamlit run dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parent / "results"
STAGE_COLORS = {"Rising": "#2F6B45", "Turning up": "#4E9A6B", "Turning down": "#D08A1E",
                "Falling": "#B5533C"}
STAGE_ORDER = ["Rising", "Turning up", "Turning down", "Falling"]
INK, MUTED, GRID = "#1E2421", "#5E6863", "#E6E9E7"

st.set_page_config(page_title="Housing Momentum Forecaster", page_icon="🏠", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
  .sub {color: #5E6863; margin-top: -0.2rem; margin-bottom: 1rem;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px; color:white;
         font-size:0.85rem; font-weight:600;}
  .card {border:1px solid #E6E9E7; border-radius:10px; padding:14px 16px; background:white;}
  .card h4 {margin:0 0 4px 0; font-size:0.95rem;}
  .card p {margin:0; color:#3A423E; font-size:0.92rem;}
  div[data-testid="stMetricValue"] {font-size: 1.7rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    hist = pd.read_csv(RESULTS / "index_history.csv", parse_dates=["date"])
    track = json.loads((RESULTS / "track_record.json").read_text())
    copy = json.loads((RESULTS / "copy.json").read_text())
    bands = pd.read_csv(RESULTS / "score_bands.csv")
    stage_perf = pd.read_csv(RESULTS / "stage_performance.csv")
    card = pd.read_csv(RESULTS / "scorecard.csv", index_col=0) if (RESULTS / "scorecard.csv").exists() else None
    sig = pd.read_csv(RESULTS / "significance_tests.csv") if (RESULTS / "significance_tests.csv").exists() else None
    act_file = RESULTS / "activity_test.json"
    act = json.loads(act_file.read_text()) if act_file.exists() else {}
    fnd_file = RESULTS / "findings.json"
    fnd = json.loads(fnd_file.read_text()) if fnd_file.exists() else {}
    misses = pd.read_csv(RESULTS / "biggest_misses.csv") if (RESULTS / "biggest_misses.csv").exists() else None
    cs_file = RESULTS / "case_study_2022.csv"
    cs = pd.read_csv(cs_file, parse_dates=["date"]) if cs_file.exists() else None
    sig_file = RESULTS / "signal_tests.csv"
    sigt = pd.read_csv(sig_file) if sig_file.exists() else None
    return hist, track, copy, bands, stage_perf, card, sig, act, fnd, misses, cs, sigt


def pill(stage):
    return f"<span class='pill' style='background:{STAGE_COLORS[stage]}'>{stage}</span>"


def style(fig, h=420):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


if not (RESULTS / "index_history.csv").exists():
    st.error("No results yet. Run run_all.bat (or step1 to step4) first.")
    st.stop()

hist, track, copy, bands, stage_perf, card, sig, act, fnd, misses, cs, sigt = load()
months = sorted(hist["date"].unique())

# ---------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    audience = st.radio("I work in", list(copy["audience"]), index=0)
    month = st.select_slider("Month", options=months, value=months[-1],
                             format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
    st.divider()
    st.markdown("**How the score works**")
    st.caption("0–100 per metro, each month. It ranks metros by the model's forecast of "
               "home-price change over the next 3 months vs. the average metro. 100 = strongest "
               "outlook. The drivers on each metro page show why.")
    st.caption("Stages compare where prices have been (last 3 months) with where the model "
               "expects them to go.")
    st.caption("Data: Zillow Research (home values, market stats), FRED, Google Trends.")

now = hist[hist["date"] == month].sort_values("rank")
changed = now[now["prev_stage"].notna() & (now["prev_stage"] != now["stage"])]
as_of = pd.Timestamp(month).strftime("%B %Y")

st.markdown("# Housing Momentum Forecaster")
st.markdown(f"<div class='sub'>Which housing markets are set to outperform over the next 3 months · "
            f"22 U.S. metros · {as_of}</div>", unsafe_allow_html=True)
if track.get("demo"):
    st.warning("Prototype running on **simulated data**. Every number here is placeholder "
               "until the real download finishes.", icon="⚠️")

tab_over, tab_metro, tab_alerts, tab_track = st.tabs(
    ["Overview", "Metro detail", "Alerts", "Track record"])

# ---------------- Overview
with tab_over:
    heat = now[now["stage"] == "Turning up"]
    top = now.iloc[0]
    movers = now.dropna(subset=["score_chg"])
    up = movers.loc[movers["score_chg"].idxmax()] if len(movers) else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Top metro · score {top['score']:.0f}", top["metro"])
    c2.metric("Turning up", f"{len(heat)} of {len(now)}",
              help="Prices have lagged, but the model expects them to start beating other metros.")
    if up is not None:
        c3.metric("Biggest riser", up["metro"], f"{up['score_chg']:+.0f} pts")
    c4.metric("Stage changes", len(changed), help="Metros that moved to a new stage this month.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Market map")
        st.caption("Left to right: how prices did over the last 3 months. Bottom to top: what "
                   "the model expects over the next 3. Both vs. the average metro.")
        fig = go.Figure()
        xr_ = max(0.5, now["price_vs_avg"].abs().max() * 1.2)
        yr = max(0.5, now["forecast"].abs().max() * 1.25)
        xr = [-xr_, xr_]
        for x0, x1, y0, y1, name in [(0, xr_, 0, yr, "Rising"), (-xr_, 0, 0, yr, "Turning up"),
                                     (0, xr_, -yr, 0, "Turning down"), (-xr_, 0, -yr, 0, "Falling")]:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line_width=0,
                          fillcolor=STAGE_COLORS[name], opacity=0.07, layer="below")
            right_side = x0 == 0
            top_half = y1 > 0
            fig.add_annotation(x=x1 * 0.97 if right_side else x0 * 0.97,
                               y=0.03 * yr if top_half else -0.03 * yr,
                               yanchor="bottom" if top_half else "top",
                               text=f"<b>{name}</b>", showarrow=False,
                               xanchor="right" if right_side else "left",
                               font=dict(color=STAGE_COLORS[name], size=14))
        fig.add_vline(x=0, line_color="#B8BFBB", line_width=1)
        fig.add_hline(y=0, line_color="#B8BFBB", line_width=1)
        # label the strongest, weakest and stage-changers; hover shows the rest
        show_names = set(now.head(6)["metro"]) | set(now.tail(3)["metro"]) | set(changed["metro"])
        for s in STAGE_ORDER:
            d = now[now["stage"] == s]
            fig.add_trace(go.Scatter(
                x=d["price_vs_avg"], y=d["forecast"], mode="markers+text", text=d["metro"],
                texttemplate=[m if m in show_names else "" for m in d["metro"]],
                textposition="top center", textfont=dict(size=11, color=INK),
                marker=dict(size=8 + d["score"] / 8, color=STAGE_COLORS[s],
                            line=dict(color="white", width=1)),
                customdata=d[["score", "rank"]],
                hovertemplate="<b>%{text}</b><br>Score %{customdata[0]:.0f} (rank %{customdata[1]})"
                              "<br>Last 3 months %{x:+.2f} pts<br>Next 3 months (forecast) %{y:+.2f} pts"
                              "<extra></extra>"))
        fig.update_xaxes(range=xr, title="Last 3 months: price change vs. average metro (pts)")
        fig.update_yaxes(range=[-yr, yr], title="Next 3 months: forecast vs. average (pts)")
        st.plotly_chart(style(fig, 520), width='stretch')

    with right:
        st.markdown("#### Ranking")
        st.caption(f"What it means for **{audience.lower()}** is on each metro's detail page.")
        table = now[["rank", "metro", "score", "score_chg", "stage"]
                    + (["forecast"] if "forecast" in now else [])].copy()
        cfg = {
            "rank": st.column_config.NumberColumn("#", width=36),
            "metro": st.column_config.TextColumn("Metro"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d",
                                                     width="small"),
            "score_chg": st.column_config.NumberColumn("Δ", format="%+d", width=40,
                                                     help="Change vs. last month"),
            "stage": st.column_config.TextColumn("Stage", width=105),
            "forecast": st.column_config.NumberColumn("Forecast", format="%+.1f", width=75,
                                                      help="Model forecast: next 3 months price change vs. the average metro, in pts"),
        }
        st.dataframe(table, column_config=cfg, hide_index=True, width='stretch', height=520)

# ---------------- Metro detail
with tab_metro:
    pick = st.selectbox("Metro", now["metro"].tolist(), index=0)
    r = now[now["metro"] == pick].iloc[0]
    series = hist[hist["metro"] == pick].sort_values("date")
    a, b = st.columns([1, 1.6])
    with a:
        st.markdown(f"### {pick}")
        st.markdown(f"{pill(r['stage'])} &nbsp; {copy['stages'][r['stage']]}", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Score", f"{r['score']:.0f}", None if pd.isna(r["score_chg"]) else f"{r['score_chg']:+.0f}")
        m2.metric("Rank", f"{int(r['rank'])} of {len(now)}")
        if "forecast" in r and pd.notna(r["forecast"]):
            m3.metric("Forecast", f"{r['forecast']:+.1f} pts", help="Next 3 months vs. the average metro")
        st.markdown("**Why**")
        st.write(r["reason"])
        st.markdown("**What it means**")
        for aud, lines in copy["audience"].items():
            bold = "**" if aud == audience else ""
            st.markdown(f"- {bold}{aud}:{bold} {lines[r['stage']]}")
    with b:
        comps = [c for c in track["components"]]
        fig = go.Figure(go.Bar(
            x=[r[c] for c in comps], y=comps, orientation="h",
            marker_color=["#2F6B45" if r[c] >= 50 else "#C9CFCB" for c in comps],
            text=[f"{r[c]:.0f}" for c in comps], textposition="outside"))
        fig.add_vline(x=50, line_dash="dot", line_color="#8C958F")
        fig.update_xaxes(range=[0, 108], title="Percentile vs. other metros (50 = typical)")
        fig.update_yaxes(autorange="reversed")
        st.markdown("**What's driving the score**")
        st.plotly_chart(style(fig, 230), width='stretch')

        recent = series[series["date"] >= pd.Timestamp(month) - pd.DateOffset(years=5)]
        recent = recent[recent["date"] <= month]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["date"], y=recent["score"], mode="lines",
                                 line=dict(color="#2F6B45", width=2.5), name="Score",
                                 hovertemplate="%{x|%b %Y}: %{y:.0f}<extra></extra>"))
        fig.add_hline(y=50, line_dash="dot", line_color="#8C958F")
        fig.update_yaxes(range=[0, 100], title="Score")
        st.markdown("**Score, last 5 years**")
        st.plotly_chart(style(fig, 250), width='stretch')

# ---------------- Alerts
with tab_alerts:
    st.markdown(f"#### Stage changes in {as_of}")
    if changed.empty:
        st.info("No metro changed stage this month.")
    else:
        for _, x in changed.iterrows():
            st.markdown(f"<div class='card' style='margin-bottom:8px'><h4>{x['metro']}: "
                        f"{pill(x['prev_stage'])} → {pill(x['stage'])}</h4>"
                        f"<p>{copy['audience'][audience][x['stage']]}</p></div>",
                        unsafe_allow_html=True)
    st.markdown("#### Biggest score moves")
    mv = now.dropna(subset=["score_chg"]).sort_values("score_chg")
    mv = pd.concat([mv.head(5), mv.tail(5)]).drop_duplicates("metro").sort_values("score_chg")
    fig = go.Figure(go.Bar(x=mv["score_chg"], y=mv["metro"], orientation="h",
                           marker_color=["#2F6B45" if v > 0 else "#B5533C" for v in mv["score_chg"]],
                           text=[f"{v:+.0f}" for v in mv["score_chg"]], textposition="outside"))
    fig.update_xaxes(title="Change in score vs. last month")
    st.plotly_chart(style(fig, 360), width='stretch')

# ---------------- Track record
with tab_track:
    st.markdown("#### Does a high score come before price gains?")
    st.caption("Honest caveat: Zillow smooths its home-value series, so price trends persist "
               "and are partly predictable from past prices alone. The fair test is how much "
               "the model beats the naive 'trend continues' baseline (bottom table).")
    st.caption(f"Tested month by month from {track['first_test_month']}: each month's score "
               "compared with what prices actually did over the next 3 months.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Top 5 beat the average", f"{track['top5_hit_rate']:.0%}",
              help="Share of top-5 metros whose prices beat the average metro over the next 3 months. Random ≈ 50%.")
    k2.metric("Top-5 extra appreciation", f"{track['top5_excess_per_year_pct']:+.1f} pts/yr",
              help="Average home-value gain of the top 5 above the average metro, annualized. Not an investment return.")
    k3.metric("Months tested", track["months_tested"])
    l, rr = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=bands["band"], y=bands["avg_next_3m_vs_avg_pct"],
                               marker_color=["#B5533C" if v < 0 else "#2F6B45" for v in bands["avg_next_3m_vs_avg_pct"]],
                               text=[f"{v:+.2f}" for v in bands["avg_next_3m_vs_avg_pct"]],
                               textposition="outside"))
        fig.add_hline(y=0, line_color="#8C958F")
        fig.update_xaxes(title="Score this month", type="category")
        fig.update_yaxes(title="Next 3 months, vs. average metro (pts)")
        st.markdown("**Higher score, better next 3 months**")
        st.plotly_chart(style(fig, 340), width='stretch')
    with rr:
        sp = stage_perf.set_index("stage").reindex(STAGE_ORDER).reset_index()
        fig = go.Figure(go.Bar(x=sp["stage"], y=sp["mean"],
                               marker_color=[STAGE_COLORS[s] for s in sp["stage"]],
                               text=[f"{v:+.2f}" for v in sp["mean"]], textposition="outside"))
        fig.add_hline(y=0, line_color="#8C958F")
        fig.update_yaxes(title="Next 3 months, vs. average metro (pts)")
        st.markdown("**What happened next, by stage**")
        st.plotly_chart(style(fig, 340), width='stretch')
    dc = track.get("driver_rank_corr", {})
    if dc:
        st.markdown("#### What actually predicts the next 3 months")
        st.caption("Average rank correlation between each driver and what prices did next "
                   "(0 = no link, 1 = perfect ranking).")
        items = sorted(((k, v) for k, v in dc.items() if k != "score"), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                               marker_color=["#2F6B45" if v >= 0.3 else "#9DB8A6" for _, v in items],
                               text=[f"{v:.2f}" for _, v in items], textposition="outside"))
        fig.update_xaxes(range=[0, max(0.8, max(v for _, v in items) * 1.15)])
        st.plotly_chart(style(fig, 260), width='stretch')
    if card is not None:
        st.markdown("#### Does search data help?")
        st.markdown("**Not for prices, so far.** Adding Google search to the housing data barely "
                    "changed the forecast, and the difference isn't statistically significant. "
                    "Housing market data (price momentum, how fast homes sell) carries the signal.")
        names = {"Naive": "Naive (trend continues)", "A": "A: housing data only",
                 "B": "B: + search (buyer and out-of-town)", "C": "C: + home-improvement search",
                 "C_gbm": "C, gradient boosting (check)"}
        show = card.rename(index=names)[["rank_corr", "top5_hit_rate", "direction_acc", "MAE_pct"]]
        show.columns = ["Ranking accuracy", "Top-5 hit rate", "Direction right", "Avg error (pts)"]
        st.dataframe(show.style.format("{:.2f}"), width='stretch')
        if sig is not None:
            st.caption(" · ".join(f"{x.comparison}: {x.verdict} (p={x.p_value_one_sided:.3f})"
                                  for x in sig.itertuples()))
    if act:
        st.markdown("#### Does search predict market activity instead?")
        rows = []
        for k, v in act.items():
            sc = v["scorecard"]
            rows.append({"Target (next 3 months)": v["label"],
                         "Housing data only": sc["A"]["rank_corr"],
                         "+ search": sc["B"]["rank_corr"],
                         "Search adds signal?": "Yes" if "better" in v["B_vs_A"]["verdict"] else "No"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                     column_config={"Housing data only": st.column_config.NumberColumn(format="%.2f"),
                                    "+ search": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Ranking accuracy (rank correlation) of each model for that target, same "
                   "month-by-month test.")

    if cs is not None:
        st.markdown("#### 2022 turning point")
        st.caption("Score (0-100) through the 2022 rate shock. Dashed line: when home values "
                   "peaked (Sun Belt) or started beating the average metro (Chicago, Boston).")
        from plotly.subplots import make_subplots
        order = ["Phoenix", "Austin", "Las Vegas", "Chicago", "Boston", "Salt Lake City"]
        order = [m for m in order if m in set(cs["metro"])]
        fig = make_subplots(rows=2, cols=3, subplot_titles=order, shared_yaxes=True,
                            vertical_spacing=0.16, horizontal_spacing=0.05)
        tp = pd.read_csv(RESULTS / "turning_points.csv", parse_dates=["date"]) \
            if (RESULTS / "turning_points.csv").exists() else None
        for i, m in enumerate(order):
            row, col = i // 3 + 1, i % 3 + 1
            d = cs[cs["metro"] == m]
            fig.add_hrect(y0=50, y1=105, fillcolor="#2F6B45", opacity=0.05, line_width=0, row=row, col=col)
            fig.add_trace(go.Scatter(x=d["date"], y=d["score"], mode="lines",
                                     line=dict(color=INK, width=2), name=m,
                                     hovertemplate="%{x|%b %Y}: %{y:.0f}<extra></extra>"), row=row, col=col)
            if m in ("Chicago", "Boston") and tp is not None:
                f = tp[(tp["metro"] == m) & (tp["direction"] == "up") & (tp["date"] >= "2022-06-01")]
                mark, colr = (f["date"].iloc[0] if len(f) else None), "#2F6B45"
            else:
                mark, colr = d.loc[d["zhvi"].idxmax(), "date"], "#B5533C"
            if mark is not None:
                fig.add_vline(x=mark, line_dash="dash", line_color=colr, row=row, col=col)
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(style(fig, 460), width='stretch')
        tps = fnd.get("turning_points", {})
        if tps:
            st.markdown(f"Across all metros, the score had already crossed 50 in the right "
                        f"direction before **{tps['flagged_before']:.0%}** of {tps['turns']} lasting "
                        f"turns in a metro's price trend (typically "
                        f"{tps['median_months_early_up']:.0f} months early for upturns, "
                        f"{tps['median_months_early_down']:.0f} for downturns). It missed "
                        f"{tps['missed']:.0%}.")
    if fnd.get("periods"):
        st.markdown("#### When it works, and when it misses")
        pr = pd.DataFrame(fnd["periods"]).rename(columns={
            "period": "Period", "months": "Months", "ranking_accuracy": "Ranking accuracy",
            "worst_month": "Worst month", "top5_beat_avg": "Top 5 beat average"})
        st.dataframe(pr, hide_index=True, width='stretch',
                     column_config={"Top 5 beat average": st.column_config.NumberColumn(format="percent")})
        if misses is not None:
            who = ", ".join(dict.fromkeys(misses["metro"].head(5)))
            st.caption(f"Biggest misses ({who}) came in fast-moving markets, when prices swung "
                       "harder than any recent trend suggested:")
            mm = misses.rename(columns={"date": "Month", "metro": "Metro", "forecast": "Forecast (pts)",
                                        "realized_vs_avg_pct": "Actual (pts)", "miss": "Miss (pts)"})
            st.dataframe(mm.head(5).round(2), hide_index=True, width='stretch')

    if sigt is not None and len(sigt):
        st.markdown("#### Which extra data helps, and how far ahead?")
        st.caption("Each signal added to the housing-data model and tested month by month at "
                   "3, 6 and 12 months ahead. Change in ranking accuracy; ✓ = statistically "
                   "significant improvement (p < 0.05).")
        piv = sigt.assign(cell=[f"{g:+.3f}{' ✓' if h else ''}" for g, h in zip(sigt["gain"], sigt["helps"])])
        piv = piv.pivot(index="signal", columns="horizon_months", values="cell")
        piv = piv.reindex(list(dict.fromkeys(sigt["signal"])))
        piv.columns = [f"{c} months" for c in piv.columns]
        st.dataframe(piv, width='stretch')
        winners = sigt[sigt["helps"]]
        if len(winners):
            w = ", ".join(f"{r.signal.lower()} at {r.horizon_months} months" for r in winners.itertuples())
            st.markdown(f"**Helps:** {w}. Everything else adds nothing beyond the housing data.")
