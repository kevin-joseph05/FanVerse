"""
data.py — FanVerse data layer.

Loads repository_signals.json and fan_segments.json, applies filters,
and returns clean dataframes for use by all dashboard components.
"""

import json
import pickle
from pathlib import Path
from functools import lru_cache

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_ROOT = Path(__file__).parent.parent
_SIGNALS_PATH = _ROOT / "repository" / "repository_signals.json"
_SEGMENTS_PATH = _ROOT / "notebooks" / "fan_segments.json"


def _get_sport_options():
    """Load unique sports from repository_signals.json."""
    with open(_SIGNALS_PATH) as f:
        records = json.load(f)
    sports = set()
    for rec in records:
        if "sports" in rec and isinstance(rec["sports"], list):
            sports.update(rec["sports"])
    # Filter out None values
    sports = {s for s in sports if s is not None}
    # Sort alphabetically, but put 'general' last
    sorted_sports = sorted([s for s in sports if s != "general"])
    if "general" in sports:
        sorted_sports.append("general")
    return ["All"] + sorted_sports


SPORT_OPTIONS = _get_sport_options()
SOURCE_OPTIONS = ["All", "Social", "Research", "Substack"]
PERIOD_OPTIONS = ["1yr", "5yr", "All time"]

# Maps the source toggle to actual source values in the data
SOURCE_MAP = {
    "Social": {"reddit", "substack", "youtube"},
    "Research": {"wasserman", "deloitte", "bcg", "nielsen"},
    "Substack": {"substack"},
}

SEGMENT_COLOURS = {
    "Superfan": "#4A90D9",
    "Core Engaged Fan": "#5BAD6F",
    "Casual Enthusiast": "#E8A838",
    "Frustrated Loyalist": "#D95B5B",
    "Emotionally Invested, Weak Signal": "#9B6FD9",
    "Passive / Disengaged": "#A0A0A0",
}

SPORT_COLOURS = {
    "WNBA": "#D95B5B",
    "NWSL": "#5BAD6F",
    "WTA": "#E8A838",
    "volleyball": "#9B6FD9",
    "general": "#4A90D9",
    "MLB": "#1f77b4",
    "MLS": "#ff7f0e",
    "NBA": "#2ca02c",
    "NFL": "#d62728",
    "NHL": "#9467bd",
    "PWHL": "#8c564b",
    "formula1": "#e377c2",
    "olympics": "#7f7f7f",
    "premierleague": "#bcbd22",
}

PATHWAY_LABELS = {
    "loyalty_signal": "Loyalty Signal",
    "churn_risk": "Churn Risk",
    "conversion_trigger": "Conversion Trigger",
    "community_influence": "Community Influence",
    "purchase_intent": "Purchase Intent",
    "identity_attachment": "Identity Attachment",
    "disengagement_marker": "Disengagement",
    "none": "Unclassified",
}

PRIORITY_LABELS = {
    "loyalty_stress": "Loyalty Stress",
    "identity_anchor": "Identity Anchor",
    "conversion_moment": "Conversion Moment",
    "cross_sport_superfan": "Cross-Sport Super Fan",
    "trust_split": "Trust Split",
    "none": "None",
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
    df["is_social"] = df["source"].isin(SOURCE_MAP["Social"])
    df["is_research"] = df["source"].isin(SOURCE_MAP["Research"])
    return df


@lru_cache(maxsize=1)
def _load_segments_raw() -> pd.DataFrame:
    pca_df = build_pca_df()
    return pca_df[["record_id", "sport", "source", "segment"]].copy()


# ── Public filter function ─────────────────────────────────────────────────


def apply_filters(
    sport: str = "All",
    source: str = "All",
    period: str = "All time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (signals_df, segments_df) with filters applied.

    Both share record_id so they can be joined if needed.
    signals_df  — full enriched records (text, scores, pathway, priority)
    segments_df — clustered records with segment labels
    """
    signals = _load_signals_raw().copy()
    segments = _load_segments_raw().copy()

    # Sport filter
    if sport != "All":
        signals = signals[signals["sport"] == sport]
        segments = segments[segments["sport"] == sport]

    # Source filter
    if source != "All":
        allowed = SOURCE_MAP[source]
        signals = signals[signals["source"].isin(allowed)]
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
    prior_avg = valid[valid["date"] < mid]["emotional_affinity_score"].mean()
    recent_avg = valid[valid["date"] >= mid]["emotional_affinity_score"].mean()

    delta = (
        round(recent_avg - prior_avg, 1)
        if pd.notna(prior_avg) and pd.notna(recent_avg)
        else 0.0
    )
    return {
        "score": current,
        "delta": delta,
        "delta_pct": round(delta / max(prior_avg, 1) * 100, 1),
    }


def kpi_churn_signals(signals: pd.DataFrame) -> dict:
    churn = signals[
        signals["behavioral_pathway"].isin(["churn_risk", "disengagement_marker"])
        | signals["priority_signal"].isin(["loyalty_stress", "trust_split"])
    ]
    pathway_counts = churn["behavioral_pathway"].value_counts().to_dict()
    priority_counts = churn["priority_signal"].value_counts().to_dict()
    return {
        "total": len(churn),
        "pathway_counts": pathway_counts,
        "priority_counts": priority_counts,
    }


def kpi_conversion_signals(signals: pd.DataFrame) -> dict:
    conv = signals[
        signals["behavioral_pathway"].isin(
            ["conversion_trigger", "purchase_intent", "loyalty_signal"]
        )
        | signals["priority_signal"].isin(["conversion_moment", "identity_anchor"])
    ]
    pathway_counts = conv["behavioral_pathway"].value_counts().to_dict()
    return {
        "total": len(conv),
        "pathway_counts": pathway_counts,
    }


def kpi_record_counts(signals: pd.DataFrame) -> dict:
    return {
        "total": len(signals),
        "social": int(signals["is_social"].sum()),
        "research": int(signals["is_research"].sum()),
        "by_sport": (
            signals[signals["sport"] != "general"]["sport"].value_counts().to_dict()
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

    top = valid.nlargest(top_n // 2 + 1, "emotional_affinity_score")
    bottom = valid.nsmallest(top_n // 2 + 1, "emotional_affinity_score")
    notable = pd.concat([top, bottom]).drop_duplicates("record_id")

    annotations = []
    for _, row in notable.iterrows():
        annotations.append(
            {
                "date": row["date"],
                "sport": row["sport"],
                "score": row["emotional_affinity_score"],
                "full_text": row["text"].strip(),
                "signal": row["priority_signal"]
                if row["priority_signal"] != "none"
                else row["behavioral_pathway"],
                "source": row["source"],
            }
        )
    return sorted(annotations, key=lambda x: x["date"])


# ── PCA scatter data ───────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_kmeans_model():
    """Load the pre-trained KMeans model."""
    with open(_ROOT / "repository" / "kmeans_k8.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_scaler():
    """Load the pre-fitted scaler."""
    with open(_ROOT / "repository" / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _get_cluster_to_segment_map():
    """Map cluster IDs (0-7) to segment labels."""
    # Hardcoded mapping based on cluster characteristics:
    # 0: High sentiment, High affinity, High confidence → Superfan
    # 1: Moderate sentiment, Moderate affinity, Low confidence → Casual Enthusiast
    # 2: High sentiment, Low affinity, Moderate confidence → Casual Enthusiast
    # 3: Very low sentiment, Moderate affinity, Moderate confidence → Frustrated Loyalist
    # 4: Moderate sentiment, High affinity, Moderate confidence → Emotionally Invested, Weak Signal
    # 5: High sentiment, Moderate affinity, High confidence → Core Engaged Fan
    # 6: Low sentiment, Low affinity, Very low confidence → Passive / Disengaged
    # 7: Very low sentiment, High affinity, High confidence → Frustrated Loyalist
    return {
        0: "Superfan",
        1: "Casual Enthusiast",
        2: "Casual Enthusiast",
        3: "Frustrated Loyalist",
        4: "Emotionally Invested, Weak Signal",
        5: "Core Engaged Fan",
        6: "Passive / Disengaged",
        7: "Frustrated Loyalist",
    }


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
    # Load raw (un-exploded) signals
    with open(_SIGNALS_PATH) as f:
        raw_records = json.load(f)
    raw_df = pd.DataFrame(raw_records)
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")

    # Extract features for all records
    feature_cols = ["sentiment_score", "emotional_affinity_score", "confidence_score"]
    raw_df[feature_cols] = raw_df[feature_cols].astype(float)
    # Drop rows with missing features
    valid = raw_df.dropna(subset=feature_cols).copy()

    # Scale features using pre-fitted scaler
    scaler = _load_scaler()
    X_scaled = scaler.transform(valid[feature_cols])

    # Predict cluster IDs using pre-trained KMeans model
    model = _load_kmeans_model()
    cluster_ids = model.predict(X_scaled)

    # Map cluster IDs to segment labels
    cluster_to_segment = _get_cluster_to_segment_map()
    valid["segment"] = [cluster_to_segment[cid] for cid in cluster_ids]

    # Compute PCA on scaled features of all records
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    valid["pc1"] = coords[:, 0]
    valid["pc2"] = coords[:, 1]

    # Explode sport list so each row is filterable by sport
    df = valid.explode("sports").rename(columns={"sports": "sport"})

    # Add hover text (first 120 chars of text)
    df["hover_text"] = (
        df["text"].str[:120].str.replace("\n", " ", regex=False).str.strip() + "…"
    )

    # Add source_type based on updated SOURCE_MAP
    social_sources = SOURCE_MAP["Social"]
    df["source_type"] = df["source"].apply(
        lambda s: "Social" if s in social_sources else "Research"
    )

    # Keep necessary columns (including those needed for tooltips)
    keep_cols = [
        "record_id",
        "sport",
        "source",
        "segment",
        "pc1",
        "pc2",
        "hover_text",
        "report_title",
        "behavioral_pathway",
        "priority_signal",
        "subreddit",
        "confidence_score",
    ]
    df = df[keep_cols]

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
