# cleanup.py — Remove off-topic records from the repository.
# Keeps only records that are genuinely about/by female fans of sports.
# Run: python3 cleanup.py

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_JSONL = Path("repository.jsonl")
REPO_JSON  = Path("repository.json")

# ── Signal phrases ────────────────────────────────────────────────────────────
# Same list used by all scrapers. Records from general subreddits must match
# at least one of these to be kept.
GOOD_SIGNALS = re.compile("|".join(re.escape(p) for p in [
    # Explicit fan identity
    "as a woman", "as a female", "as a girl",
    "i'm a woman", "i am a woman", "im a woman", "i'm a girl", "i am a girl", "im a girl",
    "as a woman fan", "as a female fan", "as a girl who", "woman who watches", "girl who watches",
    "women fans", "female fans", "female fan", "women who watch", "women who follow",
    "lady fans", "girl fans", "female fandom", "women's fandom",
    "female sports fan", "women sports fan", "women's sports fan", "female sports fans",
    "being a woman", "being female", "being a girl",
    "as a female sports", "as a woman sports", "experience as a woman", "experience as a female",
    "she/her", "woman here", "girl here", "female here",
    # Self-identification in context
    "i'm a female", "i am a female", "im a female",
    "year old woman", "year old female", "year-old woman", "year-old female",
    "young woman", "older woman",
    "only woman", "only female", "only girl",
    "as a mom", "as a mother",
    "women in sports", "women who play",
    # Fan origin / conversion stories
    "got me into sports", "got me hooked on", "first game i ever watched",
    "i became a fan when", "i became a fan after", "never watched sports before", "didn't watch until",
    "she's the reason i watch", "she's why i watch", "she got me into",
    "i've been a fan since", "grew up watching",
    # Loyalty / disillusionment
    "been a fan for years", "will always support her", "lost me as a fan",
    "hard to keep supporting", "love the player hate the",
    "still support her even", "the league let her down",
    # Cross-sport following
    "also follow", "watch both", "big fan of both",
    "wnba and", "nwsl and", "women's soccer and",
]), re.IGNORECASE)

# ── Subreddits / publications that are inherently female-fan focused ───────────
# Records from these don't need an explicit signal phrase.
WOMENS_SUBS = {"wnba", "NWSL", "WomensSoccer", "PWHL", "womenssports", "WomensSports", "FemaleSports"}
WOMENS_PUBS = {"sportswomen", "womenssportsfan", "womeninsports", "thebreakaway", "beyondthearc"}

# ── Research sources — always keep ────────────────────────────────────────────
RESEARCH_SOURCES = {"wasserman", "deloitte", "nielsen", "bcg", "mckinsey"}


def is_relevant(r: dict) -> bool:
    # Always keep research reports
    if r.get("source") in RESEARCH_SOURCES:
        return True

    # Drop downvoted content (community rejected it)
    if r.get("score", 0) < 0:
        return False

    # Women's subreddits — inherently about female fans/players
    if r.get("subreddit") in WOMENS_SUBS:
        return True

    # Women's Substack publications — inherently relevant
    if r.get("publication") in WOMENS_PUBS:
        return True

    # Drop records from off-topic search queries that have no good signal
    if r.get("search_query") in ("my wife", "my girlfriend"):
        return GOOD_SIGNALS.search(r.get("text", "")) is not None

    # Everything else — must have an explicit female fan signal
    return bool(GOOD_SIGNALS.search(r.get("text", "")))


def main():
    if not REPO_JSONL.exists():
        print("repository.jsonl not found")
        sys.exit(1)

    records = []
    with open(REPO_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    before = len(records)
    kept    = [r for r in records if is_relevant(r)]
    removed = [r for r in records if not is_relevant(r)]

    # ── Stats ────────────────────────────────────────────────────────────────
    print(f"Before: {before:,}  |  Kept: {len(kept):,}  |  Removed: {len(removed):,}")
    print()

    removed_reasons = Counter()
    for r in removed:
        if r.get("score", 0) < 0:
            removed_reasons["negative score"] += 1
        elif r.get("search_query") in ("my wife", "my girlfriend"):
            removed_reasons["off-topic search query"] += 1
        else:
            removed_reasons["no female fan signal"] += 1
    print("Removed by reason:")
    for reason, n in removed_reasons.most_common():
        print(f"  {n:5,}  {reason}")
    print()

    kept_by_source = Counter(r["source"] for r in kept)
    print("Kept by source:", dict(kept_by_source.most_common()))
    kept_by_sport = Counter(s for r in kept for s in r.get("sports", []))
    print("Kept by sport:", dict(kept_by_sport.most_common()))
    print()

    # ── Write ────────────────────────────────────────────────────────────────
    confirm = input(f"Write {len(kept):,} records back to repository? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted — no changes made.")
        return

    with open(REPO_JSONL, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    with open(REPO_JSON, "w") as f:
        json.dump(kept, f, indent=2)

    print(f"Done. {len(kept):,} records saved.")


if __name__ == "__main__":
    main()
