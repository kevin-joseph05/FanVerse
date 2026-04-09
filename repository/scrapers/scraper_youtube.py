# scraper_youtube.py — YouTube comment scraper for female fan conversations.
# Searches for sports and league content, collects comments with female fan signals.
# Run: python3 scraper_youtube.py
# Install: pip install requests python-dotenv

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.ingest import ingest

load_dotenv(Path(__file__).parent.parent / "pipeline" / ".env", override=False)

API_KEY = os.environ.get("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"
REQUEST_DELAY = 0.3
MAX_RESULTS_PER_QUERY = 15
MAX_COMMENTS_PER_VIDEO = 100
MAX_WORKERS = 6
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
    "WNBA 2024 season",
    "NWSL highlights",
    "NWSL 2024",
    "PWHL highlights",
    "women's World Cup 2023 highlights",
    "women's basketball highlights",
    "women's soccer highlights",
    # Star athletes — high engagement, many female fans in comments
    "Caitlin Clark highlights",
    "Angel Reese",
    "Sabrina Ionescu",
    "A'ja Wilson",
    "Alex Morgan",
    "Trinity Rodman NWSL",
    "Serena Williams tribute",
    "Simone Biles",
    "Sydney McLaughlin",
    # Women-focused sports content
    "women sports fans reaction",
    "female sports fan",
    "women watching sports",
    "women basketball fans",
    "women soccer fans",
    "became a sports fan",
    "sports fan story",
    # Podcasts and analysis
    "sports podcast 2024",
    "sports analysis podcast",
    "women in sports podcast",
    "female sports podcast",
    "Her Hoop Stats podcast",
    "Just Women's Sports",
    # Tennis (large female fanbase)
    "tennis highlights 2024",
    "tennis fan reaction",
    "WTA highlights",
    "Coco Gauff",
    # Olympics
    "Olympics 2024 highlights",
    "Olympics fan reaction",
    "Paris 2024 Olympics",
    # Other sports
    "gymnastics highlights",
    "swimming Olympics highlights",
    "volleyball highlights",
    "golf LPGA highlights",
]

# Same signal list as the Reddit scraper — keep these in sync
FEMALE_FAN_SIGNALS = [
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
]

_SIGNAL_RE = re.compile(
    "|".join(re.escape(p) for p in FEMALE_FAN_SIGNALS),
    re.IGNORECASE,
)


def has_female_fan_signal(text: str) -> bool:
    return bool(_SIGNAL_RE.search(text))


def search_videos(query: str) -> list:
    resp = requests.get(f"{BASE_URL}/search", params={
        "key": API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": MAX_RESULTS_PER_QUERY,
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


def fetch_video_entries(video_id: str, video_title: str, query: str, seen_comment_ids: set) -> list:
    """Fetch and filter comments for one video. Called in a thread pool."""
    try:
        comments = get_comments(video_id)
    except Exception as e:
        print(f"  x Comments failed for {video_id}: {e}")
        return []
    time.sleep(REQUEST_DELAY)

    entries = []
    for item in comments:
        comment_id = item["id"]
        if comment_id in seen_comment_ids:
            continue
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        if len(text) < MIN_TEXT_LENGTH:
            continue
        if not has_female_fan_signal(text):
            continue
        entries.append((comment_id, build_entry(text, video_title, video_id, query)))
    return entries


def main():
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not set in .env")
        return

    print("=" * 55)
    print("  FanVerse YouTube Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    # Collect all (video_id, video_title, query) tuples first, then fetch comments in parallel
    video_tasks = []
    seen_video_ids: set = set()

    for query in SEARCH_QUERIES:
        print(f"\n[Search] {query}")
        try:
            videos = search_videos(query)
        except Exception as e:
            print(f"  x Search failed: {e}")
            continue
        time.sleep(REQUEST_DELAY)

        for video in videos:
            video_id = video["id"].get("videoId")
            if not video_id:
                continue
            if video_id not in seen_video_ids:
                seen_video_ids.add(video_id)
                video_tasks.append((video_id, video["snippet"]["title"], query))

    print(f"\n── Fetching comments for {len(video_tasks)} videos ({MAX_WORKERS} workers) ──")
    all_entries = []
    seen_comment_ids: set = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(fetch_video_entries, vid, title, query, seen_comment_ids): (vid, title)
            for vid, title, query in video_tasks
        }
        for future in as_completed(future_to_video):
            vid, title = future_to_video[future]
            results = future.result()
            new = [(cid, entry) for cid, entry in results if cid not in seen_comment_ids]
            for cid, entry in new:
                seen_comment_ids.add(cid)
                all_entries.append(entry)
            if new:
                print(f"  [video] {title[:60]} — {len(new)} comments kept")

    print(f"\n── Ingesting {len(all_entries)} total entries ───────────────")
    if all_entries:
        ingest(all_entries)
    else:
        print("Nothing collected.")
    print("\nDone.")


main()
