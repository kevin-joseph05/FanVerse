# schema.py
# Defines what a single record looks like in the repository.
# Every piece of data we collect gets passed through here before it gets saved.

import hashlib
import uuid
from datetime import datetime, date as today_date
from typing import Optional

# The only values we accept for these fields.
# If something outside this list gets passed in, the system throws an error.
VALID_SOURCES = {
    "wasserman", "deloitte", "bcg", "nielsen",
    "mckinsey",  # reserved — site blocks scrapers, add manually if needed
    "reddit", "youtube", "substack"
}
VALID_SPORTS = {
    # Women's leagues
    "WNBA", "NWSL", "WTA", "PWHL",
    # Men's leagues
    "NFL", "NBA", "MLB", "MLS", "NHL",
    # International
    "formula1", "olympics", "laliga", "premierleague",
    # General
    "volleyball", "general"
}
VALID_SEASON_PHASES = {"preseason", "midseason", "playoff", "finals", "offseason", "unknown"}


def make_record_id(source: str, text: str) -> str:
    # Creates a unique fingerprint for a record based on its source and text.
    # The same source + text will always produce the same fingerprint.
    # This is how we detect duplicates. If the fingerprint already exists, we skip it.
    raw = f"{source.lower()}::{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

def build_record(
    text: str,
    source: str,
    report_title: str,
    url: str,
    sports: list[str],
    record_date: Optional[str] = None,
    date: Optional[str] = None,
    week: Optional[int] = None,
    season_phase: Optional[str] = "unknown",
    extra: Optional[dict] = None,
) -> dict:
    # Takes raw data and packages it into a clean, validated record.
    # Called once per data point — one call produces one record.

    # Make sure source and sports are valid before saving anything
    source = source.lower().strip()
    assert source in VALID_SOURCES, f"Unknown source: {source}. Must be one of {VALID_SOURCES}"

    sports = [s.strip() for s in sports]
    for s in sports:
        assert s in VALID_SPORTS, f"Unknown sport: {s}. Must be one of {VALID_SPORTS}"

    return {
        "record_id": make_record_id(source, text),  # fingerprint used for deduplication
        "post_id": str(uuid.uuid4()),                # unique ID for this specific record
        "text": text.strip(),
        "source": source,
        "report_title": report_title,
        "url": url,
        "sports": sports,                            # always an array — can cover multiple leagues
        "date": record_date or date or today_date.today().isoformat(),  # falls back to today if not provided
        "week": week,
        "season_phase": season_phase or "unknown",
        "ingested_at": datetime.utcnow().isoformat() + "Z",  # when this record was added to the system
        **(extra or {}),                             # any extra fields get added here
    }