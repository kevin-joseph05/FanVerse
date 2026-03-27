# scraper_youtube.py — YouTube comment scraper for female fan conversations.
# Searches for sports and league content, collects comments with female fan signals.
# Run: python3 scraper_youtube.py
# Install: pip install requests python-dotenv

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from ingest import ingest

load_dotenv(Path(__file__).parent / ".env", override=False)

API_KEY = os.environ.get("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"
REQUEST_DELAY = 0.5
MAX_COMMENTS_PER_VIDEO = 50
MIN_TEXT_LENGTH = 100

SEARCH_QUERIES = [
    # Men's leagues — finding female fan engagement organically
    "NBA highlights",
    "NBA fan reaction",
    "NFL highlights",
    "NFL fan reaction",
    "Premier League highlights",
    "Premier League fan reaction",
    "La Liga highlights",
    "Formula 1 race highlights",
    "NHL highlights",
    "MLS highlights",
    "MLB highlights",
    # Women's leagues
    "WNBA highlights",
    "NWSL highlights",
    "PWHL highlights",
    # Podcasts and analysis
    "sports podcast 2024",
    "sports analysis podcast",
]

# Same signal list as the Reddit scraper
FEMALE_FAN_SIGNALS = [
    "as a woman", "as a female", "as a girl",
    "i'm a woman", "i am a woman", "im a woman",
    "i'm a girl", "i am a girl", "im a girl",
    "as a woman fan", "as a female fan", "as a girl who",
    "woman who watches", "girl who watches",
    "women fans", "female fans", "female fan",
    "women who watch", "women who follow",
    "lady fans", "girl fans", "female fandom",
    "female sports fan", "women sports fan",
    "being a woman", "being female", "being a girl",
    "she/her", "woman here", "girl here", "female here",
    "got me into", "got me hooked", "first game i ever",
    "fell in love with", "changed everything for me",
    "i became a fan when", "never watched before",
    "she's the reason", "she's why i", "she got me",
    "my favorite player", "i've been a fan since",
    "grew up watching", "been a fan for years",
    "can't watch anymore", "stopped watching",
    "lost me as a fan", "used to love",
    "love the player hate the", "the league let",
    "also follow", "watch both", "big fan of both",
    "wnba and", "nwsl and", "women's soccer and",
]

_SIGNAL_RE = re.compile(
    "|".join(re.escape(p) for p in FEMALE_FAN_SIGNALS),
    re.IGNORECASE,
)

def has_female_fan_signal(text: str) -> bool:
    return bool(_SIGNAL_RE.search(text))

def search_videos(query: str, max_results: int = 5) -> list:
    resp = requests.get(f"{BASE_URL}/search", params={
        "key": API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
        "relevanceLanguage": "en",
        "order": "relevance",
    })
    resp.raise_for_status()
    return resp.json().get("items", [])

def get_comments(video_id: str) -> list:
    resp = requests.get(f"{BASE_URL}/commentThreads", params={
        "key": API_KEY,
        "videoId": video_id,
        "part": "snippet",
        "maxResults": MAX_COMMENTS_PER_VIDEO,
        "order": "relevance",
        "textFormat": "plainText",
    })
    if resp.status_code == 403:
        # Comments disabled on this video
        return []
    resp.raise_for_status()
    return resp.json().get("items", [])

def build_entry(comment: str, video_title: str, video_id: str, query: str) -> dict:
    return {
        "text": comment[:2000],
        "source": "youtube",
        "report_title": f"YouTube: {video_title[:120]}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "sports": ["general"],
        "record_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "season_phase": "unknown",
        "extra": {
            "content_type": "comment",
            "video_id": video_id,
            "search_query": query,
        },
    }

def main():
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not set in .env")
        return

    print("=" * 55)
    print("  FanVerse YouTube Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    all_entries = []
    seen_comment_ids = set()

    for query in SEARCH_QUERIES:
        print(f"\n[Search] {query}")
        try:
            videos = search_videos(query)
        except Exception as e:
            print(f"  x Search failed: {e}")
            continue
        time.sleep(REQUEST_DELAY)

        for video in videos:
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]

            try:
                comments = get_comments(video_id)
            except Exception as e:
                print(f"  x Comments failed for {video_id}: {e}")
                time.sleep(REQUEST_DELAY)
                continue
            time.sleep(REQUEST_DELAY)

            collected = 0
            for item in comments:
                comment_id = item["id"]
                if comment_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(comment_id)

                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                if len(text) < MIN_TEXT_LENGTH:
                    continue
                if not has_female_fan_signal(text):
                    continue

                all_entries.append(build_entry(text, video_title, video_id, query))
                collected += 1

            if collected:
                print(f"  [video] {video_title[:60]} — {collected} comments kept")

    print(f"\n── Ingesting {len(all_entries)} total entries ───────────────")
    if all_entries:
        ingest(all_entries)
    else:
        print("Nothing collected.")
    print("\nDone.")

main()