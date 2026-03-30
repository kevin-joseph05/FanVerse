"""
app.py — FanVerse Streamlit dashboard entry point.

Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from data import (
    SPORT_OPTIONS,
    SOURCE_OPTIONS,
    PERIOD_OPTIONS,
    SEGMENT_COLOURS,
    SPORT_COLOURS,
    apply_filters,
    build_pca_df,
    kpi_affinity_score,
    kpi_churn_signals,
    kpi_conversion_signals,
    kpi_record_counts,
    affinity_trend,
    affinity_trend_annotations,
    segment_summary,
)

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FanVerse",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Global font */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Top bar */
  .topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0 14px 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 20px;
  }
  .wordmark {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #1a1a2e;
    margin-right: 8px;
  }
  .tagline {
    font-size: 11px;
    color: #888;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* KPI cards */
  .kpi-card {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 16px 20px;
  }
  .kpi-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #999;
    margin-bottom: 4px;
  }
  .kpi-value {
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    color: #1a1a2e;
  }
  .kpi-delta-pos { font-size: 12px; color: #3a9a3a; margin-top: 4px; }
  .kpi-delta-neg { font-size: 12px; color: #c03030; margin-top: 4px; }
  .kpi-delta-neu { font-size: 12px; color: #999;    margin-top: 4px; }

  /* Tag chips */
  .chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    margin: 2px 2px 2px 0;
  }
  .chip-red   { background: #fde8e8; color: #900; }
  .chip-green { background: #e8f5e8; color: #040; }
  .chip-blue  { background: #e8f0fd; color: #003; }

  /* Section headers */
  .section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #666;
    border-bottom: 1px solid #eee;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Top bar ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="topbar">
  <span class="wordmark">FANVERSE</span>
  <span class="tagline">Female Fan Intelligence</span>
</div>
""", unsafe_allow_html=True)

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 5])

with filter_col1:
    sport_filter = st.selectbox("Sport", SPORT_OPTIONS, index=0, label_visibility="collapsed",
                                 key="sport", help="Filter by sport")
    st.caption(f"Sport: **{sport_filter}**")

with filter_col2:
    source_filter = st.selectbox("Source", SOURCE_OPTIONS, index=0, label_visibility="collapsed",
                                  key="source", help="Social = Reddit  |  Research = industry reports")
    st.caption(f"Source: **{source_filter}**")

with filter_col3:
    period_filter = st.selectbox("Period", PERIOD_OPTIONS, index=2, label_visibility="collapsed",
                                  key="period", help="Restrict to records within this window")
    st.caption(f"Period: **{period_filter}**")

# Load data with current filters
signals, segments = apply_filters(
    sport=sport_filter,
    source=source_filter,
    period=period_filter,
)


# ── KPI Strip ──────────────────────────────────────────────────────────────

kpi_affinity  = kpi_affinity_score(signals)
kpi_churn     = kpi_churn_signals(signals)
kpi_conv      = kpi_conversion_signals(signals)
kpi_counts    = kpi_record_counts(signals)

k1, k2, k3, k4 = st.columns(4)

with k1:
    delta = kpi_affinity["delta"]
    delta_class = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
    arrow = "▲" if delta >= 0 else "▼"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Emotional Affinity Score</div>
      <div class="kpi-value">{kpi_affinity['score']}</div>
      <div class="{delta_class}">{arrow} {abs(delta)} vs prior period</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    churn_tags = ""
    for label, count in list(kpi_churn["priority_counts"].items())[:2]:
        if label != "none":
            churn_tags += f'<span class="chip chip-red">{label} ×{count}</span>'
    for label, count in list(kpi_churn["pathway_counts"].items())[:2]:
        if label not in ("none", churn_tags):
            churn_tags += f'<span class="chip chip-red">{label} ×{count}</span>'
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Active Churn / Stress Signals</div>
      <div class="kpi-value" style="color:#c03030;">{kpi_churn['total']}</div>
      <div style="margin-top:6px;">{churn_tags}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    conv_tags = ""
    for label, count in list(kpi_conv["pathway_counts"].items())[:3]:
        if label != "none":
            conv_tags += f'<span class="chip chip-green">{label} ×{count}</span>'
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Conversion Signals</div>
      <div class="kpi-value" style="color:#2a7a2a;">{kpi_conv['total']}</div>
      <div style="margin-top:6px;">{conv_tags}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    sport_chips = ""
    for sp, cnt in list(kpi_counts["by_sport"].items())[:4]:
        sport_chips += f'<span class="chip chip-blue">{sp} {cnt}</span>'
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Records Analysed</div>
      <div class="kpi-value">{kpi_counts['total']}</div>
      <div style="margin-top:6px;">
        <span class="chip chip-blue">social {kpi_counts['social']}</span>
        <span class="chip chip-blue">research {kpi_counts['research']}</span>
        {sport_chips}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)


# ── Row 2: Affinity trend + Segment overview ───────────────────────────────

trend_col, seg_col = st.columns([3, 2])

with trend_col:
    st.markdown('<div class="section-header">Emotional Affinity Score — Trend</div>', unsafe_allow_html=True)

    trend_df = affinity_trend(signals, freq="M")
    annotations = affinity_trend_annotations(signals, top_n=6)

    if trend_df.empty:
        st.info("No dated records match the current filters.")
    else:
        fig = go.Figure()

        sports_in_data = trend_df["sport"].unique()
        for sp in sports_in_data:
            sp_df = trend_df[trend_df["sport"] == sp].sort_values("period")
            colour = SPORT_COLOURS.get(sp, "#888")
            fig.add_trace(go.Scatter(
                x=sp_df["period"],
                y=sp_df["avg_affinity"],
                mode="lines+markers",
                name=sp.upper() if sp != "general" else "General",
                line=dict(color=colour, width=2),
                marker=dict(size=5),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"Sport: {sp}<br>"
                    "Avg Affinity: %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            ))

        # Annotations for notable events
        for ann in annotations:
            if ann["sport"] in sports_in_data:
                colour = SPORT_COLOURS.get(ann["sport"], "#888")
                signal_str = ann["signal"] if ann["signal"] != "none" else ""
                hover = f"{ann['snippet']}"
                if signal_str:
                    hover += f"<br><i>{signal_str}</i>"
                fig.add_annotation(
                    x=ann["date"],
                    y=ann["score"],
                    text="●",
                    showarrow=False,
                    font=dict(size=14, color=colour),
                    hovertext=hover,
                )

        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[0, 105], title="Avg Affinity Score"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # "What moved it" expander
        if annotations:
            with st.expander("What moved it — notable signal records", expanded=False):
                for ann in annotations:
                    arrow = "▲" if ann["score"] >= 70 else "▼"
                    colour = "#2a7a2a" if ann["score"] >= 70 else "#c03030"
                    signal_display = ann["signal"] if ann["signal"] != "none" else "—"
                    st.markdown(
                        f"<span style='color:{colour};font-weight:bold;'>{arrow} {ann['score']}</span> "
                        f"&nbsp;·&nbsp; <code>{ann['sport']}</code> "
                        f"&nbsp;·&nbsp; <span style='color:#888;font-size:12px;'>{ann['date'].strftime('%b %d, %Y') if hasattr(ann['date'], 'strftime') else ann['date']}</span> "
                        f"&nbsp;·&nbsp; <span style='font-size:11px;color:#4A90D9;'>{signal_display}</span><br>"
                        f"<span style='font-size:12px;color:#555;'>{ann['snippet']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")


with seg_col:
    st.markdown('<div class="section-header">Fan Segments — Overview</div>', unsafe_allow_html=True)

    seg_summary = segment_summary(segments)

    if seg_summary.empty:
        st.info("No segment data matches the current filters.")
    else:
        fig_donut = go.Figure(go.Pie(
            labels=seg_summary["segment"],
            values=seg_summary["count"],
            hole=0.55,
            marker_colors=[SEGMENT_COLOURS.get(s, "#ccc") for s in seg_summary["segment"]],
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value} records (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            height=210,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Segment table
        for _, row in seg_summary.iterrows():
            colour = SEGMENT_COLOURS.get(row["segment"], "#ccc")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"border-bottom:1px solid #f0f0f0;padding:4px 0;font-size:12px;'>"
                f"<span style='display:flex;align-items:center;gap:6px;'>"
                f"<span style='width:10px;height:10px;border-radius:50%;background:{colour};"
                f"display:inline-block;flex-shrink:0;'></span>{row['segment']}</span>"
                f"<span style='color:#888;'><b style='color:#333;'>{row['count']}</b> &nbsp; {row['pct']}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)


# ── Fan Segment Map — PCA 2D Scatter ──────────────────────────────────────

st.markdown('<div class="section-header">Fan Segment Map — PCA 2D Scatter</div>', unsafe_allow_html=True)

pca_df = build_pca_df()

tab_all, tab_wnba, tab_nwsl, tab_wta, tab_social, tab_research = st.tabs([
    "All Sports", "WNBA", "NWSL", "WTA", "Social Only", "Research Only"
])

def _render_scatter(df: pd.DataFrame, title_note: str = "") -> None:
    """Renders a PCA scatter for the given (already filtered) dataframe."""
    if df.empty:
        st.info("No records match this view.")
        return

    n = len(df)
    # Normalise dot size: confidence_score 0→1 mapped to marker size 6→18
    marker_sizes = (df["confidence_score"] * 12 + 6).clip(6, 18)

    fig = go.Figure()

    for segment, colour in SEGMENT_COLOURS.items():
        mask = df["segment"] == segment
        if not mask.any():
            continue
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x=sub["pc1"],
            y=sub["pc2"],
            mode="markers",
            name=segment,
            marker=dict(
                color=colour,
                size=marker_sizes[mask].tolist(),
                line=dict(width=0.5, color="white"),
                opacity=0.85,
            ),
            customdata=sub[["hover_text", "behavioral_pathway",
                             "priority_signal", "sport", "confidence_score"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pathway: %{customdata[1]}<br>"
                "Priority: %{customdata[2]}<br>"
                "Sport: %{customdata[3]}<br>"
                "Confidence: %{customdata[4]:.2f}<br>"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=24, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.01,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="PC1 — Overall Engagement (↑ sentiment · affinity · confidence)",
            showgrid=True, gridcolor="#f5f5f5", zeroline=True, zerolinecolor="#ddd",
            title_font=dict(size=11), tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="PC2 — Sentiment vs Affinity Trade-off",
            showgrid=True, gridcolor="#f5f5f5", zeroline=True, zerolinecolor="#ddd",
            title_font=dict(size=11), tickfont=dict(size=10),
        ),
        hovermode="closest",
        annotations=[dict(
            text=f"{n} records · 88.7% variance explained{' · ' + title_note if title_note else ''}",
            xref="paper", yref="paper",
            x=0, y=1.04, showarrow=False,
            font=dict(size=10, color="#aaa"),
        )],
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_all:
    # Deduplicate on record_id so multi-sport records appear once
    _render_scatter(pca_df.drop_duplicates("record_id"))

with tab_wnba:
    _render_scatter(pca_df[pca_df["sport"] == "WNBA"], "WNBA only")

with tab_nwsl:
    _render_scatter(pca_df[pca_df["sport"] == "NWSL"], "NWSL only")

with tab_wta:
    wta_df = pca_df[pca_df["sport"] == "WTA"]
    if len(wta_df) < 5:
        st.caption(f"Only {len(wta_df)} WTA records in dataset — scatter shown but interpret with caution.")
    _render_scatter(wta_df, "WTA only")

with tab_social:
    _render_scatter(
        pca_df[pca_df["source"] == "reddit"].drop_duplicates("record_id"),
        "Reddit / social only",
    )

with tab_research:
    research_df = pca_df[pca_df["source"] != "reddit"].drop_duplicates("record_id")
    if len(research_df) < 10:
        st.caption(f"Only {len(research_df)} research records — sparse by design at this stage.")
    _render_scatter(research_df, "Research reports only")

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)


# ── Placeholders (Person B — Sankey + Feed, Person C — Insight + Sim) ─────

b_col, feed_col = st.columns(2)

with b_col:
    st.markdown('<div class="section-header">Behavioural Pathways</div>', unsafe_allow_html=True)
    st.info("🔧  Person B — Sankey diagram here. Use `signals[['behavioral_pathway','priority_signal','segment']]`.")

with feed_col:
    st.markdown('<div class="section-header">Cultural Signal Feed</div>', unsafe_allow_html=True)
    st.info("🔧  Person B — filtered record list here. Use `signals` sorted by date, filtered on churn/stress signals.")

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

st.markdown('<div class="section-header">Insight Panel — Ask FanVerse</div>', unsafe_allow_html=True)
st.info("🔧  Person C — Claude API insight panel here. Import `signals` and `segments` from `data.apply_filters()`.")

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

st.markdown('<div class="section-header">Live Simulation — Action vs Status Quo</div>', unsafe_allow_html=True)
st.info("🔧  Person C — simulation bar charts here. Use segment cluster means from `segments`.")
