#!/usr/bin/env python3
"""
Scrape Reddit comments from specific URLs and save to FanVerse format JSONL.

Usage:
  python3 scrape_specific_urls.py
"""

import json
import os
import re
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / "pipeline" / ".env", override=False)

USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "script:FanVerse-scraper:v0.1 (research project)")
BASE_URL = "https://www.reddit.com"
REQUEST_DELAY = 0.5
COMMENT_LIMIT = 100
MIN_TEXT_LENGTH = 50

MONTPELLIER_URLS = [
    "https://www.reddit.com/r/australia/comments/1p1srsm/mary_fowler_reveals_she_experienced_depression/",
    "https://www.reddit.com/r/Aleague/comments/1p2m056/montpellier_football_club_has_strongly_rejected/",
]

ROSENGARD_URLS = [
    "https://www.reddit.com/r/WomensSoccer/comments/1qd38e0/swedish_fa_today_approved_multiclub_owners_crux/",
    "https://www.reddit.com/r/WomensSoccer/comments/1rcsqxm/rosengård_stands_empty_as_fans_abandon_club_after/",
    "https://www.reddit.com/r/WomensSoccer/comments/1i7fi7v/rosengård_situation/",
    "https://www.reddit.com/r/WomensSoccer/comments/1s7jq4f/hammarby_3_0_rosengård_fanny_peterson_goal_how/",
    "https://www.reddit.com/r/WomensSoccer/comments/1oj40f9/completely_missed_this_yesterday_but_caroline/",
    "https://www.reddit.com/r/WomensSoccer/comments/1nvxxn8/swedish_league_at_twenty_games_with_six_to_go/",
    "https://www.reddit.com/r/WomensSoccer/comments/1khndso/rosengård_want_to_reschedule_swedish_league_games/",
    "https://www.reddit.com/r/NWSL/comments/1m5hgqi/racing_transfers_pikkujämsä_to_swedens_fc/",
    "https://www.reddit.com/r/ArsenalWFC/comments/1qu9stp/fc_rosengard_on_instagram_cecily_wellesleysmith/",
    "https://www.reddit.com/r/ArsenalWFC/comments/1rvo7nf/cecily_wellesleysmith_scores_her_first_goal_for/",
]


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def reddit_get(session, url, params=None) -> Optional[dict]:
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                print(f"    404 — post not found")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"    Request failed: {exc}")
                return None
            time.sleep(5)
    return None


def extract_post_id(url: str) -> Optional[str]:
    """Extract post ID from Reddit URL."""
    match = re.search(r'/comments/([a-z0-9]+)/', url)
    return match.group(1) if match else None


def extract_subreddit(url: str) -> Optional[str]:
    """Extract subreddit name from Reddit URL."""
    match = re.search(r'/r/([a-zA-Z0-9_]+)/', url)
    return match.group(1) if match else None


def ts_to_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def ts_to_week(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-W%V")


def infer_season_phase(text: str) -> str:
    """Infer season phase from comment text."""
    lower = text.lower()
    phases = {
        "finals": ["finals", "championship", "title game"],
        "playoff": ["playoff", "playoffs", "postseason"],
        "preseason": ["preseason", "pre-season", "training camp"],
        "offseason": ["offseason", "off-season", "free agency"],
        "midseason": ["all-star", "midseason"],
    }
    for phase, keywords in phases.items():
        if any(kw in lower for kw in keywords):
            return phase
    return "unknown"


def is_valid_comment(body: str) -> bool:
    """Check if comment is valid (not deleted/removed, English, min length)."""
    if not body or body in ("[deleted]", "[removed]"):
        return False
    if len(body.strip()) < MIN_TEXT_LENGTH:
        return False
    # Basic English check (mostly ASCII)
    ascii_ratio = sum(1 for c in body if ord(c) < 128) / len(body)
    if ascii_ratio < 0.85:
        return False
    return True


def make_record_id(text: str) -> str:
    """Generate deterministic record ID from text."""
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def fetch_comments_from_url(session, url: str, report_title: str, client_name: str) -> list:
    """Fetch comments from a specific Reddit post URL."""
    post_id = extract_post_id(url)
    subreddit = extract_subreddit(url)

    if not post_id or not subreddit:
        print(f"  ✗ Could not parse URL: {url}")
        return []

    print(f"  Fetching from r/{subreddit}/comments/{post_id}...")
    data = reddit_get(session, f"{BASE_URL}/r/{subreddit}/comments/{post_id}.json",
                      params={"limit": COMMENT_LIMIT, "depth": 1})
    time.sleep(REQUEST_DELAY)

    if not data or len(data) < 2:
        print(f"    0 comments fetched")
        return []

    comments_data = [c["data"] for c in data[1]["data"]["children"] if c.get("kind") == "t1"]
    print(f"    {len(comments_data)} comments found")

    entries = []
    valid_count = 0

    for comment in comments_data:
        body = comment.get("body", "").strip()
        if not is_valid_comment(body):
            continue

        record_id = make_record_id(body)
        comment_date = ts_to_date(comment.get("created_utc", 0))
        comment_week = ts_to_week(comment.get("created_utc", 0))

        entry = {
            "record_id": record_id,
            "post_id": post_id,
            "text": body,
            "source": "reddit",
            "report_title": report_title,
            "url": url,
            "sports": ["football"],
            "date": comment_date,
            "week": comment_week,
            "season_phase": infer_season_phase(body),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "content_type": "comment",
            "score": comment.get("score", 0),
        }
        entries.append(entry)
        valid_count += 1

    print(f"    ✓ {valid_count} valid comments")
    return entries


def scrape_club(session, urls: list, club_name: str, output_file: str) -> int:
    """Scrape all URLs for a club and save to JSONL."""
    print(f"\n{'='*60}")
    print(f"  {club_name.upper()}")
    print(f"{'='*60}")

    all_entries = []
    url_stats = {}

    for url in urls:
        report_title = f"Reddit: {club_name}"
        entries = fetch_comments_from_url(session, url, report_title, club_name)
        all_entries.extend(entries)
        url_stats[url] = len(entries)

    # Save to JSONL
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n  Summary:")
    for url, count in url_stats.items():
        url_short = url.split("/comments/")[1].split("/")[0]
        print(f"    {url_short[:8]}: {count} comments")

    print(f"\n  Total: {len(all_entries)} comments → {output_file}")
    return len(all_entries)


def main():
    print("\n" + "="*60)
    print("  FanVerse Specific URL Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*60)

    session = make_session()

    montpellier_count = scrape_club(
        session,
        MONTPELLIER_URLS,
        "montpellier",
        "repository/data/clients/montpellier_raw.jsonl"
    )

    rosengard_count = scrape_club(
        session,
        ROSENGARD_URLS,
        "rosengard",
        "repository/data/clients/rosengard_raw.jsonl"
    )

    print(f"\n{'='*60}")
    print(f"  Final Summary")
    print(f"{'='*60}")
    print(f"  Montpellier: {montpellier_count} comments")
    print(f"  Rosengård:   {rosengard_count} comments")
    print(f"  Total:       {montpellier_count + rosengard_count} comments")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
