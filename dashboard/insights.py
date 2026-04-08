"""
insights.py — FanVerse insight engine.

get_insight() is the single entry point for the Insight Panel.
Wire the Claude API inside that function — everything else is ready.
"""

import pandas as pd

# ── Preset queries (shown as chips in the UI) ──────────────────────────────

PRESET_QUERIES = [
    "Which of our female fans are showing early disengagement signals this month, and what triggered the shift?",
    "Which cross-sport super fan segments offer the highest co-marketing value across two leagues?",
    "What cultural moments outside of sport are currently driving our female fans toward or away from our team?",
]

# ── Prompt template ────────────────────────────────────────────────────────

INSIGHT_PROMPT = """\
You are FanVerse, a female fan intelligence engine. You have been given real fan \
signal records collected from Reddit communities and industry research reports. \
Each record includes the sport, source, behavioral pathway, priority signal, \
sentiment score, and the raw fan text.

Answer the following question based ONLY on what the records show. Do not invent \
data. If the records are insufficient to answer confidently, say so honestly in \
the finding field.

Question: {query}

Fan signal records:
{context}

Respond with a JSON object in exactly this format — no markdown fences, raw JSON only:
{{
  "finding": "One clear sentence stating the main finding from the data.",
  "evidence": "2-3 sentences in plain prose citing specific signals, record counts, or short direct quotes. Refer to sources naturally (e.g. 'A Deloitte report notes…' or 'Reddit fans comment…') — do not reproduce the raw metadata tags from the records.",
  "confidence": <integer 0-100 reflecting how well the records support the finding>,
  "recommended_action": "One concrete, specific action a sports organisation should take based on this finding."
}}"""

# ── Context builder ────────────────────────────────────────────────────────


def build_context(query: str, signals_df: pd.DataFrame, n_records: int = 25) -> str:
    """
    Selects the most relevant records from signals_df for the given query
    and formats them as plain text for the prompt.
    """
    q = query.lower()

    if any(w in q for w in ("disengagement", "churn", "leaving", "shift", "triggered")):
        mask = signals_df["behavioral_pathway"].isin(
            ["churn_risk", "disengagement_marker"]
        ) | signals_df["priority_signal"].isin(["loyalty_stress", "trust_split"])
    elif any(w in q for w in ("cross-sport", "co-marketing", "two leagues", "multi")):
        mask = signals_df["priority_signal"].isin(
            ["cross_sport_superfan", "identity_anchor"]
        )
    elif any(w in q for w in ("cultural", "outside", "moment", "toward", "away")):
        mask = signals_df["priority_signal"].isin(
            ["loyalty_stress", "trust_split"]
        ) | signals_df["behavioral_pathway"].isin(["churn_risk", "community_influence"])
    else:
        mask = signals_df["priority_signal"] != "none"

    relevant = signals_df[mask].drop_duplicates("record_id")
    if relevant.empty:
        relevant = signals_df.drop_duplicates("record_id")

    top = relevant.head(n_records)

    lines = []
    for _, row in top.iterrows():
        lines.append(
            f"[sport:{row['sport']} | source:{row['source']} | "
            f"pathway:{row['behavioral_pathway']} | priority:{row['priority_signal']} | "
            f"sentiment:{row['sentiment']} ({row['sentiment_score']:.2f})] "
            f"{str(row['text'])[:300]}"
        )
    return "\n\n".join(lines)


# ── Main insight function — wire Gemini API here ──────────────────────────


def get_insight(
    query: str, signals_df: pd.DataFrame, segments_df: pd.DataFrame
) -> dict:
    """
    Returns a structured insight dict:
        {
            "finding":            str,
            "evidence":           str,
            "confidence":         int,   # 0–100
            "recommended_action": str,
            "ready":              bool,  # False until API is wired
        }

    ── HOW TO WIRE THE GEMINI API ──────────────────────────────────────────

    1. 'google-generativeai' is already in pyproject.toml — run: uv sync
    2. Add GEMINI_API_KEY=... to repository/.env
    3. Replace the TODO block below with:

        import google.generativeai as genai
        import os, json as _json
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv(Path(__file__).parent.parent / "repository" / ".env")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        model   = genai.GenerativeModel("gemini-2.5-flash")
        context = build_context(query, signals_df)
        prompt  = INSIGHT_PROMPT.format(query=query, context=context)

        response = model.generate_content(prompt)
        result   = _json.loads(response.text)
        result["ready"] = True
        return result

    ────────────────────────────────────────────────────────────────────────
    """
    import google.generativeai as genai
    import os, json as _json
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).parent.parent / "repository" / ".env")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    model = genai.GenerativeModel("gemini-2.5-flash")
    context = build_context(query, signals_df)
    prompt = INSIGHT_PROMPT.format(query=query, context=context)

    try:
        response = model.generate_content(prompt)
        result = _json.loads(response.text)
        result["ready"] = True
        return result
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "ResourceExhausted" in err:
            finding = "API quota exceeded. Enable billing on your Google AI project to use the Insight Panel."
        else:
            finding = f"API error: {err[:200]}"
        return {
            "finding": finding,
            "evidence": "",
            "confidence": 0,
            "recommended_action": "",
            "ready": False,
        }


# ── Simulation model ───────────────────────────────────────────────────────

_SEGMENT_ORDER = [
    "Superfan",
    "Core Engaged Fan",
    "Emotionally Invested, Weak Signal",
    "Casual Enthusiast",
    "Frustrated Loyalist",
    "Passive / Disengaged",
]

# Status quo fan base distribution (% of total, sums to 100)
_BEFORE_DIST = {
    "Superfan": 12,
    "Core Engaged Fan": 18,
    "Emotionally Invested, Weak Signal": 20,
    "Casual Enthusiast": 25,
    "Frustrated Loyalist": 15,
    "Passive / Disengaged": 10,
}

# After-action distributions per preset query — fans convert upward
_AFTER_DIST = [
    # Query 0 — address disengagement / loyalty stress
    {
        "Superfan": 14,
        "Core Engaged Fan": 20,
        "Emotionally Invested, Weak Signal": 22,
        "Casual Enthusiast": 25,
        "Frustrated Loyalist": 12,
        "Passive / Disengaged": 7,
    },
    # Query 1 — cross-sport co-marketing activation
    {
        "Superfan": 16,
        "Core Engaged Fan": 22,
        "Emotionally Invested, Weak Signal": 22,
        "Casual Enthusiast": 23,
        "Frustrated Loyalist": 11,
        "Passive / Disengaged": 6,
    },
    # Query 2 — cultural moment content strategy
    {
        "Superfan": 14,
        "Core Engaged Fan": 21,
        "Emotionally Invested, Weak Signal": 22,
        "Casual Enthusiast": 24,
        "Frustrated Loyalist": 12,
        "Passive / Disengaged": 7,
    },
]

# Fallback for free-text queries
_DEFAULT_AFTER = {
    "Superfan": 14,
    "Core Engaged Fan": 20,
    "Emotionally Invested, Weak Signal": 21,
    "Casual Enthusiast": 25,
    "Frustrated Loyalist": 13,
    "Passive / Disengaged": 7,
}


def compute_simulation(query_index: int | None) -> dict:
    """
    Returns before/after segment distribution data for the donut charts.

    query_index: int (0–2) for preset queries, None for free-text.

    Return shape:
        {
            "segments": list[str],
            "before":   list[int],  # % of fan base per segment, status quo
            "after":    list[int],  # % of fan base per segment, with action
            "summary": {
                "churn_reduction":    int,  # pp drop in at-risk share
                "conversion_uplift":  int,  # % increase in top-tier share
                "fans_reengaged_pct": int,  # % of at-risk fans converted
                "model_confidence":   int,
            }
        }
    """
    after_map = (
        _AFTER_DIST[query_index]
        if isinstance(query_index, int) and 0 <= query_index < len(_AFTER_DIST)
        else _DEFAULT_AFTER
    )

    before = [_BEFORE_DIST[s] for s in _SEGMENT_ORDER]
    after = [after_map[s] for s in _SEGMENT_ORDER]

    at_risk = ["Frustrated Loyalist", "Passive / Disengaged"]
    top_tier = ["Superfan", "Core Engaged Fan"]

    before_at_risk = sum(_BEFORE_DIST[s] for s in at_risk)
    after_at_risk = sum(after_map[s] for s in at_risk)
    before_top = sum(_BEFORE_DIST[s] for s in top_tier)
    after_top = sum(after_map[s] for s in top_tier)

    fans_reengaged = round((before_at_risk - after_at_risk) / before_at_risk * 100)
    churn_reduction = before_at_risk - after_at_risk
    conv_uplift = round((after_top - before_top) / before_top * 100)

    confidence_map = {0: 74, 1: 61, 2: 68, None: 55}

    return {
        "segments": _SEGMENT_ORDER,
        "before": before,
        "after": after,
        "summary": {
            "churn_reduction": churn_reduction,
            "conversion_uplift": conv_uplift,
            "fans_reengaged_pct": fans_reengaged,
            "model_confidence": confidence_map.get(query_index, 55),
        },
    }
