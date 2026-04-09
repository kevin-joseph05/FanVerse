# ingest.py
# Handles saving records to the repository file and reading them back out.
# The main thing it does: never save the same record twice.

import json
from datetime import datetime, timezone
from pathlib import Path
from schema import build_record

# Auto-tag sports based on subreddit name or keywords in report_title/text.
# Checked in order — first match wins. Falls back to whatever the scraper provided.
_SPORT_TAG_RULES = [
    (["wnba", "women's basketball", "women's nba"],         "WNBA"),
    (["nwsl", "women's soccer", "women's football"],        "NWSL"),
    (["pwhl", "women's hockey"],                            "PWHL"),
    (["wta", "women's tennis"],                             "WTA"),
    (["r/nba", "subreddit=nba", "nba "],                    "NBA"),
    (["r/nfl", "subreddit=nfl", "nfl "],                    "NFL"),
    (["r/mlb", "subreddit=mlb", "mlb "],                    "MLB"),
    (["r/nhl", "subreddit=nhl", "nhl "],                    "NHL"),
    (["r/mls", "subreddit=mls", " mls "],                   "MLS"),
    (["r/premierleague", "premier league"],                  "premierleague"),
    (["r/laliga", "la liga", "laliga"],                      "laliga"),
    (["r/formula1", "formula 1", "formula1", " f1 "],        "formula1"),
    (["r/olympics", "olympic games", "paris 2024"],          "olympics"),
    (["r/volleyball", "volleyball"],                         "volleyball"),
]


def infer_sports_from_context(entry: dict) -> list:
    """Return a more specific sports tag if we can infer one from context."""
    current = entry.get("sports", ["general"])
    if current != ["general"]:
        return current  # already tagged specifically

    # Build a lookup string from subreddit, report_title, and the start of text
    extra = entry.get("extra") or {}
    subreddit = extra.get("subreddit", "").lower()
    pub = extra.get("publication", "").lower()
    haystack = " ".join([
        f"subreddit={subreddit}",
        f"r/{subreddit}",
        pub,
        entry.get("report_title", "").lower(),
        entry.get("text", "")[:300].lower(),
    ])

    for keywords, sport in _SPORT_TAG_RULES:
        if any(kw in haystack for kw in keywords):
            return [sport]

    return current

REPO_PATH = Path(__file__).parent.parent / "data" / "processed" / "repository.jsonl"    # one record per line
LOG_PATH = Path(__file__).parent.parent / "output" / "logs" / "ingestion_log.jsonl"  # one entry per run

def load_existing_ids() -> set[str]:
    # Returns all record fingerprints already in the repo — used for deduplication.
    if not REPO_PATH.exists():
        return set()

    ids = set()
    with open(REPO_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["record_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids

def load_all() -> list[dict]:
    # Loads every record into memory. Used by query() and repo_stats().
    if not REPO_PATH.exists():
        return []

    records = []
    with open(REPO_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

def append_records(records: list[dict]) -> dict:
    # Appends new records to the repo file, skipping duplicates.
    # Append-only — never overwrites existing data.
    existing_ids = load_existing_ids()
    added = 0
    skipped = 0

    with open(REPO_PATH, "a") as f:  # "a" = append mode, never overwrites
        for record in records:
            if record["record_id"] in existing_ids:
                skipped += 1
                continue
            f.write(json.dumps(record) + "\n")
            existing_ids.add(record["record_id"])
            added += 1

    total = len(existing_ids)

    # Save a summary of this run to the log
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "added": added,
        "skipped_duplicates": skipped,
        "total_in_repo": total,
    }
    with open(LOG_PATH, "a") as log:
        log.write(json.dumps(summary) + "\n")

    print(f"[Ingestion] + {added} added  |  {skipped} skipped (duplicates)  |  {total} total in repo")
    return summary

def query(source=None, sport=None, season_phase=None) -> list[dict]:
    # Filters the repository by source, sport, or season phase.
    # All arguments are optional — pass only what you want to filter by.
    # Example: query(source="nielsen", sport="NBA")
    records = load_all()

    if source:
        records = [r for r in records if r["source"] == source.lower()]
    if sport:
        records = [r for r in records if sport in r.get("sports", [])]
    if season_phase:
        records = [r for r in records if r.get("season_phase") == season_phase]

    return records


def repo_stats() -> dict:
    # Prints a breakdown of everything in the repository.
    # Good for a quick health check - total records, broken down by source/sport/phase.
    records = load_all()

    if not records:
        print("[Repo] Empty.")
        return {}

    from collections import Counter
    stats = {
        "total_records": len(records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_sport": dict(Counter(s for r in records for s in r.get("sports", []))),
        "by_season_phase": dict(Counter(r.get("season_phase", "unknown") for r in records)),
    }
    print(json.dumps(stats, indent=2))
    return stats


def ingest(raw_entries: list[dict]) -> dict:
    # The main function. Pass it a list of data points, it saves them to the repository.
    # Builds each record using schema.py, then calls append_records to save them.
    for entry in raw_entries:
        entry["sports"] = infer_sports_from_context(entry)
    records = [build_record(**entry) for entry in raw_entries]
    return append_records(records)