"""
data.py — FanVerse data layer.

Loads repository_signals.json and fan_segments.json, applies filters,
and returns clean dataframes for use by all dashboard components.
"""

import json
from pathlib import Path
from functools import lru_cache

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_ROOT = Path(__file__).parent.parent
_SIGNALS_PATH  = _ROOT / "repository" / "output" / "repository_signals.json"
_SEGMENTS_PATH = _ROOT / "repository" / "output" / "fan_segments.json"

SOURCE_OPTIONS = ["All", "Social", "Research"]
PERIOD_OPTIONS = ["1yr", "5yr", "All time"]

# Maps the source toggle to actual source values in the data.
# youtube and substack are community/social content, not research reports.
SOURCE_MAP = {
    "Social":   {"reddit", "youtube", "substack"},
    "Research": {"wasserman", "deloitte", "bcg", "nielsen", "mckinsey"},
}


def _build_sport_options() -> list[str]:
    """Derives available sport filter options directly from the signals data file."""
    try:
        with open(_SIGNALS_PATH) as f:
            records = json.load(f)
        seen: set[str] = {s for r in records for s in r.get("sports", [])}
        # Preferred display order: women's leagues first, then men's, then global/multi
        priority = [
            "WNBA", "NWSL", "WTA", "PWHL",
            "NFL", "NBA", "NHL", "MLB", "MLS",
            "formula1", "premierleague", "olympics",
            "general",
        ]
        ordered = [s for s in priority if s in seen]
        ordered += sorted(seen - set(priority))  # any future additions
        return ["All"] + ordered
    except Exception:
        return ["All", "WNBA", "NWSL"]


SPORT_OPTIONS = _build_sport_options()

SEGMENT_COLOURS = {
    "Superfan":                        "#4A90D9",
    "Core Engaged Fan":                "#5BAD6F",
    "Casual Enthusiast":               "#E8A838",
    "Frustrated Loyalist":             "#D95B5B",
    "Emotionally Invested, Weak Signal": "#9B6FD9",
    "Passive / Disengaged":            "#A0A0A0",
}

SPORT_COLOURS = {
    # Women's leagues
    "WNBA":         "#D95B5B",
    "NWSL":         "#5BAD6F",
    "WTA":          "#E8A838",
    "PWHL":         "#4A90D9",
    "volleyball":   "#9B6FD9",
    # Men's / North American
    "NFL":          "#C84B31",
    "NBA":          "#F07B39",
    "NHL":          "#5B8ED9",
    "MLB":          "#8B5BAD",
    "MLS":          "#3BAD8E",
    # Global / multi-sport
    "formula1":     "#E84141",
    "premierleague":"#8B2BE2",
    "olympics":     "#B8860B",
    "general":      "#888888",
}

PATHWAY_LABELS = {
    "loyalty_signal":        "Loyalty Signal",
    "churn_risk":            "Churn Risk",
    "conversion_trigger":   "Conversion Trigger",
    "community_influence":  "Community Influence",
    "purchase_intent":      "Purchase Intent",
    "identity_attachment":  "Identity Attachment",
    "disengagement_marker": "Disengagement",
    "none":                 "Unclassified",
}

PRIORITY_LABELS = {
    "loyalty_stress":       "Loyalty Stress",
    "identity_anchor":      "Identity Anchor",
    "conversion_moment":    "Conversion Moment",
    "cross_sport_superfan": "Cross-Sport Super Fan",
    "trust_split":          "Trust Split",
    "none":                 "None",
}


# ── Raw loaders (cached — files don't change during a session) ─────────────

@lru_cache(maxsize=1)
def _load_signals_raw() -> pd.DataFrame:
    with open(_SIGNALS_PATH) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Explode sports so each row represents one sport tag
    df = df.explode("sports").rename(columns={"sports": "sport"})
    df["is_social"]   = df["source"] == "reddit"
    df["is_research"] = df["source"] != "reddit"
    return df


@lru_cache(maxsize=1)
def _load_segments_raw() -> pd.DataFrame:
    with open(_SEGMENTS_PATH) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df = df.explode("sports").rename(columns={"sports": "sport"})
    return df


# ── Public filter function ─────────────────────────────────────────────────

def apply_filters(
    sport: str  = "All",
    source: str = "All",
    period: str = "All time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (signals_df, segments_df) with filters applied.

    Both share record_id so they can be joined if needed.
    signals_df  — full enriched records (text, scores, pathway, priority)
    segments_df — clustered records with segment labels
    """
    signals  = _load_signals_raw().copy()
    segments = _load_segments_raw().copy()

    # Sport filter
    if sport != "All":
        signals  = signals[signals["sport"]  == sport]
        segments = segments[segments["sport"] == sport]

    # Source filter
    if source != "All":
        allowed = SOURCE_MAP[source]
        signals  = signals[signals["source"].isin(allowed)]
        segments = segments[segments["source"].isin(allowed)]

    # Period filter — applied to signals only (segments don't carry dates)
    _period_days = {"1yr": 365, "5yr": 1825}
    if period in _period_days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=_period_days[period])
        signals = signals[signals["date"] >= cutoff]

    return signals.reset_index(drop=True), segments.reset_index(drop=True)


# ── KPI helpers ────────────────────────────────────────────────────────────

def kpi_affinity_score(signals: pd.DataFrame) -> dict:
    """
    Returns current avg emotional_affinity_score and delta vs prior period.
    'Prior period' is the first half of the date range in the filtered set.
    """
    if signals.empty:
        return {"score": 0, "delta": 0.0, "delta_pct": 0.0}

    valid = signals.dropna(subset=["date", "emotional_affinity_score"])
    if valid.empty:
        return {"score": 0, "delta": 0.0, "delta_pct": 0.0}

    current = round(valid["emotional_affinity_score"].mean())

    mid = valid["date"].min() + (valid["date"].max() - valid["date"].min()) / 2
    prior_avg  = valid[valid["date"] <  mid]["emotional_affinity_score"].mean()
    recent_avg = valid[valid["date"] >= mid]["emotional_affinity_score"].mean()

    delta = round(recent_avg - prior_avg, 1) if pd.notna(prior_avg) and pd.notna(recent_avg) else 0.0
    return {"score": current, "delta": delta, "delta_pct": round(delta / max(prior_avg, 1) * 100, 1)}


def kpi_churn_signals(signals: pd.DataFrame) -> dict:
    churn = signals[
        signals["behavioral_pathway"].isin(["churn_risk", "disengagement_marker"]) |
        signals["priority_signal"].isin(["loyalty_stress", "trust_split"])
    ]
    pathway_counts = churn["behavioral_pathway"].value_counts().to_dict()
    priority_counts = churn["priority_signal"].value_counts().to_dict()
    return {
        "total":           len(churn),
        "pathway_counts":  pathway_counts,
        "priority_counts": priority_counts,
    }


def kpi_conversion_signals(signals: pd.DataFrame) -> dict:
    conv = signals[
        signals["behavioral_pathway"].isin(["conversion_trigger", "purchase_intent", "loyalty_signal"]) |
        signals["priority_signal"].isin(["conversion_moment", "identity_anchor"])
    ]
    pathway_counts = conv["behavioral_pathway"].value_counts().to_dict()
    return {
        "total":          len(conv),
        "pathway_counts": pathway_counts,
    }


def kpi_record_counts(signals: pd.DataFrame) -> dict:
    return {
        "total":    len(signals),
        "social":   int(signals["is_social"].sum()),
        "research": int(signals["is_research"].sum()),
        "by_sport": (
            signals[signals["sport"] != "general"]["sport"]
            .value_counts()
            .to_dict()
        ),
    }


# ── Affinity trend ─────────────────────────────────────────────────────────

def affinity_trend(
    signals: pd.DataFrame,
    freq: str = "M",
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [period, sport, avg_affinity, n_records]
    suitable for plotting a multi-line trend chart.

    freq: pandas offset alias — 'ME' (month-end), 'W' (week), 'QE' (quarter)
    """
    valid = signals.dropna(subset=["date", "emotional_affinity_score"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["period", "sport", "avg_affinity", "n_records"])

    valid["period"] = valid["date"].dt.to_period(freq).dt.to_timestamp()

    grouped = (
        valid.groupby(["period", "sport"])["emotional_affinity_score"]
        .agg(avg_affinity="mean", n_records="count")
        .reset_index()
    )
    grouped["avg_affinity"] = grouped["avg_affinity"].round(1)
    return grouped.sort_values("period")


def affinity_trend_annotations(signals: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Returns up to top_n records that represent notable affinity events —
    the highest and lowest scoring records in the filtered set.
    Used to annotate the trend chart with "what moved it".
    """
    valid = signals.dropna(subset=["date", "emotional_affinity_score", "text"]).copy()
    if valid.empty:
        return []

    top    = valid.nlargest(top_n // 2 + 1, "emotional_affinity_score")
    bottom = valid.nsmallest(top_n // 2 + 1, "emotional_affinity_score")
    notable = pd.concat([top, bottom]).drop_duplicates("record_id")

    annotations = []
    for _, row in notable.iterrows():
        annotations.append({
            "date":      row["date"],
            "sport":     row["sport"],
            "score":     row["emotional_affinity_score"],
            "full_text": row["text"].strip(),
            "signal":    row["priority_signal"] if row["priority_signal"] != "none" else row["behavioral_pathway"],
            "source":    row["source"],
        })
    return sorted(annotations, key=lambda x: x["date"])


# ── PCA scatter data ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_pca_df() -> pd.DataFrame:
    """
    Pre-computes PCA coordinates for all segmented records.

    PCA is fit once on the full unfiltered set so coordinates are stable
    across all tab views (WNBA tab and NWSL tab share the same space).

    Returns one row per sport tag per record (exploded), so a record tagged
    [WNBA, NWSL] appears in both sport tab filters — but the 'All' view
    deduplicates on record_id to avoid double-counting on the scatter.

    Extra columns added:
      pc1, pc2        — 2D PCA coordinates
      hover_text      — 120-char text snippet for tooltips
      source_type     — "Social" or "Research"
    """
    # Load raw (un-exploded) segments so PCA sees exactly 93 records
    with open(_SEGMENTS_PATH) as f:
        raw = json.load(f)
    seg_df = pd.DataFrame(raw)

    feature_cols = ["sentiment_score", "emotional_affinity_score", "confidence_score"]
    X = seg_df[feature_cols].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    seg_df["pc1"] = coords[:, 0]
    seg_df["pc2"] = coords[:, 1]

    # Explode sport list so each row is filterable by sport
    seg_df = seg_df.explode("sports").rename(columns={"sports": "sport"})

    # Merge in hover text + signal labels from signals (deduped — avoid sport explosion)
    signals = _load_signals_raw()
    sig_cols = ["record_id", "text", "report_title", "behavioral_pathway", "priority_signal", "subreddit"]
    sig_dedup = signals.drop_duplicates("record_id")[sig_cols]

    df = seg_df.merge(sig_dedup, on="record_id", how="left")
    df["hover_text"] = (
        df["text"].str[:120].str.replace("\n", " ", regex=False).str.strip() + "…"
    )
    df["source_type"] = df["source"].apply(
        lambda s: "Social" if s == "reddit" else "Research"
    )
    return df.reset_index(drop=True)


# ── Segment summary ────────────────────────────────────────────────────────

def segment_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Returns segment counts and percentages for the donut + table.
    """
    if segments.empty:
        return pd.DataFrame(columns=["segment", "count", "pct"])

    counts = segments["segment"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts
