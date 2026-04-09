# scraper_reddit.py — Reddit female fan conversation scraper.
# Uses public .json endpoints (no API key).
# Run: python3 scraper_reddit.py
# Install: pip install requests python-dotenv

import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.ingest import ingest  # noqa: E402

load_dotenv(Path(__file__).parent.parent / "pipeline" / ".env", override=False)

# Vague User-Agents get throttled — Reddit requires a descriptive string.
USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "script:FanVerse-scraper:v0.1 (research project)")

BASE_URL = "https://www.reddit.com"
REQUEST_DELAY = 0.5   # per-thread delay; parallelism spreads load across workers
POST_LIMIT = 100
COMMENT_LIMIT = 100
MAX_WORKERS = 4       # parallel comment fetches — conservative to avoid 429s
MIN_TEXT_LENGTH = 150
REDDIT_KIND_POST = "t3"
REDDIT_KIND_COMMENT = "t1"

# Collection method (hot/top posts vs search) is auto-selected by subscriber count.
SUBREDDITS = {
    # Women's leagues — primary targets
    "wnba":              ["WNBA"],
    "NWSL":              ["NWSL"],
    "WomensSoccer":      ["MLS", "NWSL"],
    "PWHL":              ["PWHL"],
    "womenssports":      ["general"],
    "WomensSports":      ["general"],
    "FemaleSports":      ["general"],
    # Olympic / individual sports with large female fanbases
    "olympics":          ["olympics"],
    "tennis":            ["WTA"],
    "gymnastics":        ["general"],
    "swimming":          ["general"],
    "volleyball":        ["volleyball"],
    "golf":              ["general"],
    "athletics":         ["general"],
    # Major men's leagues — searched for female fan signals
    "nba":               ["NBA"],
    "nfl":               ["NFL"],
    "mlb":               ["MLB"],
    "mls":               ["MLS"],
    "laliga":            ["laliga"],
    "premierleague":     ["premierleague"],
    "nhl":               ["NHL"],
    "formula1":          ["formula1"],
    "soccer":            ["MLS"],
    "basketball":        ["NBA"],
    "CFB":               ["NFL"],
    "collegebasketball": ["NBA"],
    # Additional leagues / communities
    "rugby":             ["general"],
    "cricket":           ["general"],
    "cycling":           ["general"],
    "sports":            ["general"],
    "gcb":               ["general"],  # r/gcb — women's college basketball
}

LARGE_THRESHOLD = 500_000  # above this: search queries; below: hot/top posts

# Expanded search queries for large subreddits
SEARCH_QUERIES = [
    "female fan", "women fans", "as a woman fan", "girl fan",
    "women who watch", "female sports fan", "as a woman",
    "as a female", "women watching", "she got me into",
    "woman here", "girl here", "as a woman who",
    "women in sports", "female fandom",
]

# Sort orders scraped for small/medium subreddits — deduped by post ID
SORT_ORDERS = [
    ("hot",  None),
    ("top",  "week"),
    ("top",  "month"),
]

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
    # Self-identification in context (e.g. "I'm a 30 year old woman", "I'm a female cyclist")
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

_SIGNAL_RE = re.compile("|".join(re.escape(p) for p in FEMALE_FAN_SIGNALS), re.IGNORECASE)

MOD_PHRASES = [
    "mod post", "moderator", "weekly thread", "free talk", "megathread", "match thread",
    "game thread", "daily thread", "weekly discussion", "jobs", "listings", "broadcast details",
    "pinned", "[mod]", "auto-generated", "bot", "unisex kit", "kit sizing", "3xl",
    "panini", "trading card", "vod", "streaming link",
]

_PHASE_KEYWORDS = {
    "finals":    ["finals", "championship", "title game"],
    "playoff":   ["playoff", "playoffs", "postseason", "bracket"],
    "preseason": ["preseason", "pre-season", "training camp", "draft"],
    "offseason": ["offseason", "off-season", "free agency", "trade deadline"],
    "midseason": ["all-star", "all star", "midseason", "mid-season"],
}

_thread_local = threading.local()


def get_thread_session() -> requests.Session:
    """Returns a per-thread Session so concurrent workers don't share state."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session()
    return _thread_local.session


def is_mod_post(title: str, body: str) -> bool:
    if len(title) < 15:
        return True
    return any(phrase in (title + " " + body).lower() for phrase in MOD_PHRASES)


def is_english(text: str) -> bool:
    return bool(text) and sum(1 for c in text if ord(c) < 128) / len(text) >= 0.85


def is_url_only(text: str) -> bool:
    return len(re.sub(r'http\S+', '', text).strip()) < MIN_TEXT_LENGTH


def has_female_fan_signal(text: str) -> bool:
    return bool(_SIGNAL_RE.search(text))


def infer_season_phase(text: str) -> str:
    lower = text.lower()
    return next((phase for phase, kws in _PHASE_KEYWORDS.items() if any(w in lower for w in kws)), "unknown")


def is_valid_comment(body: str, require_signal: bool) -> bool:
    if not body or body in ("[deleted]", "[removed]"):
        return False
    if len(body) < MIN_TEXT_LENGTH or not is_english(body) or is_url_only(body):
        return False
    return not require_signal or has_female_fan_signal(body)


def _entry(text, report_title, url, sports, record_date, extra):
    return {"text": text, "source": "reddit", "report_title": report_title, "url": url,
            "sports": sports, "record_date": record_date, "season_phase": infer_season_phase(text), "extra": extra}


def build_post_entry(post_text, report_title, url, sports, record_date, post_id, subreddit, score, search_query=None):
    extra = {"subreddit": subreddit, "reddit_post_id": post_id, "content_type": "post", "score": score,
             **({"search_query": search_query} if search_query else {})}
    return _entry(post_text[:3000], report_title, url, sports, record_date, extra)


def build_comment_entry(body, report_title, url, sports, record_date, post_id, comment_id, subreddit, score, search_query=None):
    extra = {"subreddit": subreddit, "reddit_post_id": post_id, "reddit_comment_id": comment_id,
             "content_type": "comment", "score": score,
             **({"search_query": search_query} if search_query else {})}
    return _entry(body[:2000], report_title, url, sports, record_date, extra)


def ts_to_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


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
                print(f"  Rate limited — waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                print(f"  404 — subreddit not found or private: {url}")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  Request failed after 3 attempts: {exc}")
                return None
            time.sleep(5)
    return None


def get_subscriber_count(session, subreddit_name: str) -> int:
    data = reddit_get(session, f"{BASE_URL}/r/{subreddit_name}/about.json")
    return data.get("data", {}).get("subscribers", 0) if data else 0


def fetch_posts_sorted(session, subreddit_name: str, sort: str = "hot", time_filter: str = None) -> list:
    params = {"limit": POST_LIMIT}
    if time_filter:
        params["t"] = time_filter
    data = reddit_get(session, f"{BASE_URL}/r/{subreddit_name}/{sort}.json", params=params)
    time.sleep(REQUEST_DELAY)
    if not data:
        return []
    return [c["data"] for c in data["data"]["children"] if c["kind"] == REDDIT_KIND_POST]


def _fetch_comments_worker(subreddit_name: str, post_id: str) -> list:
    """Worker that fetches comments using a thread-local session."""
    session = get_thread_session()
    data = reddit_get(session, f"{BASE_URL}/r/{subreddit_name}/comments/{post_id}.json",
                      params={"limit": COMMENT_LIMIT, "depth": 1})
    time.sleep(REQUEST_DELAY)
    if not data or len(data) < 2:
        return []
    return [c["data"] for c in data[1]["data"]["children"] if c["kind"] == REDDIT_KIND_COMMENT]


def scrape_subreddit(session, subreddit_name: str, sports: list, require_signal=False) -> list:
    # Collect posts from multiple sort orders, deduplicating by post ID
    seen_post_ids: set = set()
    all_posts = []
    for sort, time_filter in SORT_ORDERS:
        for post in fetch_posts_sorted(session, subreddit_name, sort, time_filter):
            if post["id"] not in seen_post_ids:
                seen_post_ids.add(post["id"])
                all_posts.append(post)

    entries = []
    posts_needing_comments = []
    collected_posts = 0
    rejected = {"not_text_post": 0, "stickied": 0, "mod_post": 0,
                "too_short": 0, "url_only": 0, "low_score": 0, "non_english": 0, "no_signal": 0}

    for post in all_posts:
        post_body = post.get("selftext", "").strip()
        post_title = post.get("title", "").strip()
        post_text = f"{post_title}\n\n{post_body}".strip()
        post_date = ts_to_date(post["created_utc"])
        post_url = f"https://www.reddit.com{post['permalink']}"
        report_title = f"r/{subreddit_name}: {post_title[:120]}"
        post_id, score = post["id"], post.get("score", 0)

        post_passes = False
        fetch_post_comments = True
        if not post.get("is_self") or not post_body:
            rejected["not_text_post"] += 1
            # Link posts still get comments — fan discussion happens there.
        elif post.get("stickied", False):
            rejected["stickied"] += 1
            fetch_post_comments = False
        elif is_mod_post(post_title, post_body):
            rejected["mod_post"] += 1
            fetch_post_comments = False
        elif len(post_body) < MIN_TEXT_LENGTH:
            rejected["too_short"] += 1
        elif score < 2:
            rejected["low_score"] += 1
            fetch_post_comments = False
        elif not is_english(post_text):
            rejected["non_english"] += 1
            fetch_post_comments = False
        elif is_url_only(post_text):
            rejected["url_only"] += 1
            fetch_post_comments = False
        elif require_signal and not has_female_fan_signal(post_text):
            rejected["no_signal"] += 1
        else:
            post_passes = True

        if post_passes:
            entries.append(build_post_entry(post_text, report_title, post_url, sports, post_date, post_id, subreddit_name, score))
            collected_posts += 1

        if fetch_post_comments:
            posts_needing_comments.append((post_id, report_title, post_url, sports, post_date, subreddit_name, score))

    # Parallel comment fetching
    collected_comments = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {
            executor.submit(_fetch_comments_worker, subreddit_name, pid): (pid, rtitle, purl, sp, pdate, sr, sc)
            for pid, rtitle, purl, sp, pdate, sr, sc in posts_needing_comments
        }
        for future in as_completed(future_to_meta):
            pid, report_title, post_url, sports_, post_date, subreddit_, score = future_to_meta[future]
            for comment in future.result():
                body = comment.get("body", "").strip()
                if comment.get("score", 0) < 0:
                    continue
                if not is_valid_comment(body, require_signal):
                    continue
                entries.append(build_comment_entry(
                    body, report_title, f"https://www.reddit.com{comment['permalink']}",
                    sports_, post_date, pid, comment["id"], subreddit_, comment.get("score", 0)))
                collected_comments += 1

    rejection_parts = [f"{v} {k.replace('_', ' ')}" for k, v in rejected.items() if v > 0]
    print(f"  → {len(all_posts)} posts — {', '.join(rejection_parts) or 'none'} | {collected_posts} posts + {collected_comments} comments kept")
    return entries


def search_subreddit(session, subreddit_name: str, sports: list) -> list:
    seen_post_ids: set = set()
    posts_to_process = []

    for query in SEARCH_QUERIES:
        data = reddit_get(session, f"{BASE_URL}/r/{subreddit_name}/search.json",
                          params={"q": query, "sort": "relevance", "limit": 100, "restrict_sr": 1})
        time.sleep(REQUEST_DELAY)
        if not data:
            continue
        for child in data["data"]["children"]:
            if child["kind"] != REDDIT_KIND_POST:
                continue
            post = child["data"]
            if post["id"] not in seen_post_ids:
                seen_post_ids.add(post["id"])
                posts_to_process.append((post, query))

    entries = []
    collected_posts = 0
    posts_needing_comments = []

    for post, query in posts_to_process:
        post_body = post.get("selftext", "").strip()
        post_title = post.get("title", "").strip()
        post_text = f"{post_title}\n\n{post_body}".strip()
        post_date = ts_to_date(post["created_utc"])
        post_url = f"https://www.reddit.com{post['permalink']}"
        report_title = f"r/{subreddit_name}: {post_title[:120]}"
        score = post.get("score", 0)
        post_id = post["id"]

        if (post.get("is_self") and len(post_body) >= MIN_TEXT_LENGTH and score >= 2
                and not post.get("stickied", False) and not is_mod_post(post_title, post_body)
                and is_english(post_text) and not is_url_only(post_text)):
            entries.append(build_post_entry(post_text, report_title, post_url, sports, post_date, post_id, subreddit_name, score, search_query=query))
            collected_posts += 1

        # Always fetch comments — search surfaced this post for a reason.
        posts_needing_comments.append((post_id, report_title, post_url, sports, post_date, subreddit_name, score, query))

    # Parallel comment fetching
    collected_comments = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {
            executor.submit(_fetch_comments_worker, subreddit_name, pid): (pid, rtitle, purl, sp, pdate, sr, sc, q)
            for pid, rtitle, purl, sp, pdate, sr, sc, q in posts_needing_comments
        }
        for future in as_completed(future_to_meta):
            pid, report_title, post_url, sports_, post_date, subreddit_, score, query = future_to_meta[future]
            for comment in future.result():
                body = comment.get("body", "").strip()
                if comment.get("score", 0) < 0:
                    continue
                if not is_valid_comment(body, require_signal=True):
                    continue
                entries.append(build_comment_entry(
                    body, report_title, f"https://www.reddit.com{comment['permalink']}",
                    sports_, post_date, pid, comment["id"], subreddit_, comment.get("score", 0), search_query=query))
                collected_comments += 1

    print(f"  → {len(seen_post_ids)} posts scanned | {collected_posts} posts + {collected_comments} comments kept")
    return entries


def main():
    print("=" * 55)
    print("  FanVerse Reddit Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)
    print(f"  User-Agent: {USER_AGENT}\n")

    session = make_session()
    all_entries = []

    print("── Scraping all subreddits (method auto-selected by size) ──")
    for subreddit_name, sports in SUBREDDITS.items():
        count = get_subscriber_count(session, subreddit_name)
        if count >= LARGE_THRESHOLD:
            print(f"\n[r/{subreddit_name}] {count:,} subscribers — using search queries...")
            entries = search_subreddit(session, subreddit_name, sports)
        else:
            print(f"\n[r/{subreddit_name}] {count:,} subscribers — scraping hot/top posts...")
            entries = scrape_subreddit(session, subreddit_name, sports, require_signal=True)
        all_entries.extend(entries)

    print(f"\n── Ingesting {len(all_entries)} total entries ───────────────")
    if all_entries:
        ingest(all_entries)
    else:
        print("[Scraper] Nothing collected — Reddit may be throttling or subreddits are private.")
    print("\nDone.")


if __name__ == "__main__":
    main()
