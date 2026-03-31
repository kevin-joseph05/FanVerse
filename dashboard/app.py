"""
app.py — FanVerse Streamlit dashboard entry point.

Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
from insights import PRESET_QUERIES, get_insight, compute_simulation

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FanVerse",
    page_icon="💫",
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

title_col = st.columns([1])[0]
with title_col:
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
                hover = ann["full_text"][:120].replace("\n", " ") + "…"
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
                    date_display = ann["date"].strftime("%b %d, %Y") if hasattr(ann["date"], "strftime") else str(ann["date"])
                    st.markdown(
                        f"<span style='color:{colour};font-weight:bold;'>{arrow} {ann['score']}</span>"
                        f"&nbsp;·&nbsp;<code>{ann['sport']}</code>"
                        f"&nbsp;·&nbsp;<span style='color:#888;font-size:12px;'>{date_display}</span>"
                        f"&nbsp;·&nbsp;<span style='font-size:11px;color:#4A90D9;'>{signal_display}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='font-size:12px;color:#333;line-height:1.6;white-space:pre-wrap;margin-top:4px;'>{ann['full_text']}</div>",
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

tab_all, tab_wnba, tab_nwsl, tab_social, tab_research = st.tabs([
    "All Sports", "WNBA", "NWSL", "Social Only", "Research Only"
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
            customdata=sub[["report_title", "hover_text", "behavioral_pathway",
                             "priority_signal", "sport", "confidence_score"]].values,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "<b>%{customdata[0]}</b><br>"
                "<br>"
                "%{customdata[1]}<br>"
                "<br>"
                "Pathway: %{customdata[2]}<br>"
                "Priority: %{customdata[3]}<br>"
                "Sport: %{customdata[4]}<br>"
                "Confidence: %{customdata[5]:.2f}"
                "<extra></extra>"
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
    _render_scatter(pca_df.drop_duplicates("record_id"))

with tab_wnba:
    _render_scatter(pca_df[pca_df["sport"] == "WNBA"], "WNBA only")

with tab_nwsl:
    _render_scatter(pca_df[pca_df["sport"] == "NWSL"], "NWSL only")

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


# ── Row: Behavioural Pathways + Cultural Signal Feed ──────────────────────

b_col, feed_col = st.columns(2)

with b_col:
    st.markdown('<div class="section-header">Priority Signals by Sport</div>', unsafe_allow_html=True)

    SIGNAL_COLOURS = {
        "loyalty_stress":       "#D95B5B",
        "identity_anchor":      "#4A90D9",
        "trust_split":          "#E8A838",
        "cross_sport_superfan": "#5BAD6F",
        "conversion_moment":    "#9B6FD9",
    }
    SIGNAL_LABELS = {
        "loyalty_stress":       "Loyalty Stress",
        "identity_anchor":      "Identity Anchor",
        "trust_split":          "Trust Split",
        "cross_sport_superfan": "Cross-Sport Super Fan",
        "conversion_moment":    "Conversion Moment",
    }
    SPORT_DISPLAY = {"general": "General / Multi-Sport", "WNBA": "WNBA", "NWSL": "NWSL"}

    bar_src = signals.drop_duplicates("record_id")
    bar_df  = bar_src[bar_src["priority_signal"] != "none"][["sport", "priority_signal"]].copy()
    bar_df["sport"] = bar_df["sport"].map(SPORT_DISPLAY).fillna(bar_df["sport"])

    if bar_df.empty:
        st.info("No priority signals in current filter.")
    else:
        counts = bar_df.groupby(["sport", "priority_signal"]).size().reset_index(name="n")
        sport_order = [s for s in ["WNBA", "NWSL", "General / Multi-Sport"] if s in counts["sport"].unique()]

        fig_bar = go.Figure()
        for signal, colour in SIGNAL_COLOURS.items():
            sub = counts[counts["priority_signal"] == signal]
            if sub.empty:
                continue
            fig_bar.add_trace(go.Bar(
                name=SIGNAL_LABELS.get(signal, signal),
                x=sub["sport"],
                y=sub["n"],
                marker_color=colour,
                hovertemplate="%{x}<br>" + SIGNAL_LABELS.get(signal, signal) + ": %{y} records<extra></extra>",
            ))

        fig_bar.update_layout(
            barmode="group",
            height=320,
            margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="Records", showgrid=True, gridcolor="#f0f0f0"),
            bargap=0.25,
            bargroupgap=0.08,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        total_sig = len(bar_df)
        total_all = len(bar_src)
        st.caption(f"{total_sig} of {total_all} records carry a priority signal ({round(total_sig / total_all * 100)}%)")

    with st.expander("What do these signals mean?"):
        st.markdown("""
| Signal | What it means |
|---|---|
| **Loyalty Stress** | Fan mentions threats to their attachment — trades, cuts, scandals, salary disputes, "last straw" language. Still engaged but under strain. |
| **Identity Anchor** | Fan's identity is tied to a specific player. Language like "she's the reason I watch", "follow her wherever she goes", "protect her". |
| **Trust Split** | Fan supports the players but has lost trust in the organisation. "Love the players, hate the front office / ownership / management." |
| **Cross-Sport Super Fan** | Fan explicitly follows multiple women's leagues. "WNBA and NWSL", "watch both", "also follow". High co-marketing value. |
| **Conversion Moment** | Fan describes the moment they became a fan. "First game", "got me hooked", "started watching because of her". New fan acquisition signal. |
        """)


with feed_col:
    st.markdown('<div class="section-header">Cultural Signal Feed</div>', unsafe_allow_html=True)

    # Signal → (hex colour, display label)
    FEED_SIGNAL_STYLE = {
        "churn_risk":            ("#D95B5B", "Churn Risk"),
        "disengagement_marker":  ("#D95B5B", "Disengagement"),
        "loyalty_stress":        ("#D95B5B", "Loyalty Stress"),
        "trust_split":           ("#E8A838", "Trust Split"),
        "purchase_intent":       ("#5BAD6F", "Purchase Intent"),
        "loyalty_signal":        ("#5BAD6F", "Loyalty Signal"),
        "conversion_trigger":    ("#5BAD6F", "Conversion Trigger"),
        "identity_anchor":       ("#4A90D9", "Identity Anchor"),
        "identity_attachment":   ("#4A90D9", "Identity Attachment"),
        "community_influence":   ("#9B6FD9", "Community Influence"),
        "cross_sport_superfan":  ("#5BAD6F", "Cross-Sport Super Fan"),
        "conversion_moment":     ("#9B6FD9", "Conversion Moment"),
    }

    feed_df = signals.drop_duplicates("record_id").copy()
    feed_df["date"] = pd.to_datetime(feed_df["date"], errors="coerce")
    feed_df = feed_df[
        (feed_df["behavioral_pathway"] != "none") |
        (feed_df["priority_signal"]    != "none")
    ].sort_values("date", ascending=False)

    if feed_df.empty:
        st.info("No signals in current filter.")
    else:
        n_shown = 0
        for _, row in feed_df.iterrows():
            if n_shown >= 10:
                break

            text = str(row.get("text", "")).strip()
            if len(text) < 40:
                continue

            title        = str(row.get("report_title", "")).strip()
            date_str     = row["date"].strftime("%b %d, %Y") if pd.notna(row["date"]) else "—"
            sport        = str(row.get("sport", "—"))
            subreddit    = str(row.get("subreddit", "")) or ""
            source_label = f"r/{subreddit}" if subreddit and subreddit != "nan" else str(row.get("source", "—"))

            # Pick the most specific signal for display
            active_signal = (
                row["priority_signal"]
                if row["priority_signal"] != "none"
                else row["behavioral_pathway"]
            )
            sig_colour, sig_label = FEED_SIGNAL_STYLE.get(active_signal, ("#aaa", active_signal))

            sent_colour = "#c03030" if row["sentiment"] == "negative" else (
                "#2a7a2a" if row["sentiment"] == "positive" else "#888"
            )

            expander_label = f"{sig_label}  ·  {date_str}  ·  {source_label}"

            with st.expander(expander_label, expanded=False):
                st.markdown(
                    f"<span style='display:inline-block;background:{sig_colour}22;color:{sig_colour};"
                    f"border:1px solid {sig_colour};border-radius:4px;padding:2px 8px;"
                    f"font-size:11px;font-weight:600;margin-bottom:6px;'>{sig_label}</span>",
                    unsafe_allow_html=True,
                )
                if title:
                    st.markdown(f"**{title}**")
                st.markdown(
                    f"<span style='font-size:11px;color:#aaa;'>{sport} · "
                    f"sentiment: <span style='color:{sent_colour};'>{row['sentiment']} "
                    f"({row['sentiment_score']:.2f})</span> · "
                    f"affinity: {int(row['emotional_affinity_score'])}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
                st.markdown(
                    f"<div style='font-size:13px;color:#333;line-height:1.6;white-space:pre-wrap;'>{text}</div>",
                    unsafe_allow_html=True,
                )
            n_shown += 1

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

# ── Insight Panel ──────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Insight Panel — Ask FanVerse</div>', unsafe_allow_html=True)

# Session state
if "fanverse_query"     not in st.session_state:
    st.session_state["fanverse_query"]     = None
if "fanverse_query_idx" not in st.session_state:
    st.session_state["fanverse_query_idx"] = None
if "fanverse_insight"   not in st.session_state:
    st.session_state["fanverse_insight"]   = None

# Preset query chips
st.markdown("<div style='font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Preset queries</div>", unsafe_allow_html=True)
chip_cols = st.columns(len(PRESET_QUERIES))
for i, q in enumerate(PRESET_QUERIES):
    with chip_cols[i]:
        if st.button(q, key=f"preset_{i}", use_container_width=True):
            st.session_state["fanverse_query"]     = q
            st.session_state["fanverse_query_idx"] = i
            st.session_state["fanverse_insight"]   = get_insight(q, signals, segments)

# Free-text input
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
free_col, btn_col = st.columns([5, 1])
with free_col:
    free_text = st.text_input(
        "Custom query",
        placeholder="Ask anything about your female fan base…",
        label_visibility="collapsed",
        key="free_query_input",
    )
with btn_col:
    if st.button("Ask →", use_container_width=True, type="primary"):
        if free_text.strip():
            st.session_state["fanverse_query"]     = free_text.strip()
            st.session_state["fanverse_query_idx"] = None
            st.session_state["fanverse_insight"]   = get_insight(free_text.strip(), signals, segments)

# Response card
st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
insight = st.session_state["fanverse_insight"]

if insight is None:
    st.markdown(
        "<div style='background:#fafafa;border:1px solid #eee;border-radius:8px;"
        "padding:20px;text-align:center;color:#bbb;font-size:13px;'>"
        "Select a preset query or type your own to generate an insight."
        "</div>",
        unsafe_allow_html=True,
    )
elif not insight["ready"]:
    st.markdown(
        "<div style='background:#fffbe6;border:1px solid #ffe58f;border-radius:8px;"
        "padding:16px 20px;font-size:13px;color:#7a6200;'>"
        "<b>Claude API not yet connected.</b><br>"
        "Open <code>dashboard/insights.py</code> and follow the instructions in "
        "<code>get_insight()</code> to wire in the API key and model call."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    r_finding, r_evidence, r_confidence, r_action = st.columns(4)
    confidence_colour = (
        "#2a7a2a" if insight["confidence"] >= 70
        else "#c07000" if insight["confidence"] >= 45
        else "#c03030"
    )

    with r_finding:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:1px;color:#999;margin-bottom:6px;'>Finding</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='font-size:13px;color:#333;line-height:1.5;'>{insight['finding']}</div>", unsafe_allow_html=True)

    with r_evidence:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:1px;color:#999;margin-bottom:6px;'>Evidence</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='font-size:13px;color:#555;line-height:1.5;'>{insight['evidence']}</div>", unsafe_allow_html=True)

    with r_confidence:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:1px;color:#999;margin-bottom:6px;'>Confidence</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:38px;font-weight:700;color:{confidence_colour};line-height:1;'>"
            f"{insight['confidence']}%</div>"
            f"<div style='font-size:11px;color:#aaa;margin-top:4px;'>signal match</div>",
            unsafe_allow_html=True,
        )

    with r_action:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:1px;color:#999;margin-bottom:6px;'>Recommended Action</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:13px;color:#333;line-height:1.5;'>{insight['recommended_action']}</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)


# ── Live Simulation ────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Live Simulation — Action vs Status Quo</div>', unsafe_allow_html=True)

sim = compute_simulation(st.session_state["fanverse_query_idx"])

sim_chart_col, sim_summary_col = st.columns([3, 2])

with sim_chart_col:
    colours = [SEGMENT_COLOURS.get(s, "#ccc") for s in sim["segments"]]

    fig_sim = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["Status Quo", "With Recommended Action"],
    )

    fig_sim.add_trace(go.Pie(
        labels=sim["segments"],
        values=sim["before"],
        hole=0.5,
        marker_colors=colours,
        textinfo="percent",
        textfont=dict(size=10),
        hovertemplate="%{label}: %{value}%<extra></extra>",
        showlegend=True,
        name="",
    ), row=1, col=1)

    fig_sim.add_trace(go.Pie(
        labels=sim["segments"],
        values=sim["after"],
        hole=0.5,
        marker_colors=colours,
        textinfo="percent",
        textfont=dict(size=10),
        hovertemplate="%{label}: %{value}%<extra></extra>",
        showlegend=False,
        name="",
    ), row=1, col=2)

    fig_sim.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(size=10),
        ),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

with sim_summary_col:
    s = sim["summary"]

    st.markdown(
        "<div style='background:#fafafa;border:1px solid #eee;border-radius:8px;padding:16px 18px;'>",
        unsafe_allow_html=True,
    )

    def _metric_row(label: str, value: str, colour: str) -> None:
        st.markdown(
            f"<div style='border-bottom:1px solid #f0f0f0;padding:8px 0;'>"
            f"<div style='font-size:10px;color:#999;text-transform:uppercase;letter-spacing:1px;'>{label}</div>"
            f"<div style='font-size:26px;font-weight:700;color:{colour};line-height:1.2;'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _metric_row("At-risk fans converted",       f"{s['fans_reengaged_pct']}%",  "#2a7a2a")
    _metric_row("Drop in at-risk share (pp)",   f"−{s['churn_reduction']}pp",   "#4A90D9")
    _metric_row("Growth in top-tier fans",      f"+{s['conversion_uplift']}%",  "#2a7a2a")
    _metric_row("Model confidence",             f"{s['model_confidence']}%",    "#888")

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Model-based projection · 90-day horizon · anchored to real cluster sentiment means")
