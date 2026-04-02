# scraper_substack.py — Substack post + comment scraper for female fan conversations.
# Uses Substack's undocumented but stable public JSON API (no auth required for free posts).
# Run: python3 scraper_substack.py
# Install: pip install requests python-dotenv beautifulsoup4

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from ingest import ingest

load_dotenv(Path(__file__).parent / ".env", override=False)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FanVerse-scraper/0.1; research project)"}
REQUEST_DELAY = 0.5
POSTS_PER_PAGE = 50
MAX_WORKERS = 5
MIN_TEXT_LENGTH = 150

# Two tiers:
#   require_signal=False  — women's sports / female-fan focused publications;
#                           all posts are relevant by definition.
#   require_signal=True   — general sports publications; only keep content
#                           that contains an explicit female fan signal.

PUBLICATIONS = [
    # ── Women's sports / female-fan focused ────────────────────────────
    {"slug": "sportswomen",     "sports": ["general"],           "require_signal": False},
    {"slug": "womenssportsfan", "sports": ["WNBA", "general"],   "require_signal": False},
    {"slug": "womeninsports",   "sports": ["general"],           "require_signal": False},
    {"slug": "thebreakaway",    "sports": ["general"],           "require_signal": False},
    {"slug": "beyondthearc",    "sports": ["general"],           "require_signal": False},
    {"slug": "thegist",         "sports": ["general"],           "require_signal": False},
    # ── General / men's sports (signal filter applied) ─────────────────
    {"slug": "extratime",        "sports": ["MLS", "general"], "require_signal": True},
    {"slug": "footballdaily",    "sports": ["MLS", "general"], "require_signal": True},
    {"slug": "insidethepress",   "sports": ["MLS", "general"], "require_signal": True},
    {"slug": "thepocketpass",    "sports": ["general"],           "require_signal": True},
    {"slug": "tennisweekly",     "sports": ["general"],           "require_signal": True},
    {"slug": "nhldraft",         "sports": ["general"],           "require_signal": True},
    {"slug": "sportsweekly",     "sports": ["general"],           "require_signal": True},
    {"slug": "f1newsletter",     "sports": ["formula1"],          "require_signal": True},
    {"slug": "premierleagueweekly", "sports": ["premierleague"],  "require_signal": True},
]

# Same signal list as the Reddit/YouTube scrapers — keep these in sync
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

_SIGNAL_RE = re.compile("|".join(re.escape(p) for p in FEMALE_FAN_SIGNALS), re.IGNORECASE)


def has_signal(text: str) -> bool:
    return bool(_SIGNAL_RE.search(text))


def is_english(text: str) -> bool:
    return bool(text) and sum(1 for c in text if ord(c) < 128) / len(text) >= 0.85


class _StripHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(html: str) -> str:
    p = _StripHTML()
    p.feed(html)
    return " ".join(p.parts).strip()


def substack_get(url: str, params: dict = None) -> dict | list | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  Request failed: {exc}")
                return None
            time.sleep(3)
    return None


def fetch_all_posts(pub_slug: str) -> list:
    """Paginate through all posts for a publication."""
    posts = []
    offset = 0
    while True:
        data = substack_get(
            f"https://{pub_slug}.substack.com/api/v1/archive",
            params={"sort": "new", "limit": POSTS_PER_PAGE, "offset": offset},
        )
        time.sleep(REQUEST_DELAY)
        if not data:
            break
        posts.extend(data)
        if len(data) < POSTS_PER_PAGE:
            break
        offset += POSTS_PER_PAGE
    return posts


def fetch_full_post_body(pub_slug: str, post_slug: str) -> str:
    """Fetch the full post body for a free post."""
    data = substack_get(f"https://{pub_slug}.substack.com/api/v1/posts/{post_slug}")
    time.sleep(REQUEST_DELAY)
    if not data:
        return ""
    if data.get("body_html"):
        return strip_html(data["body_html"])
    return data.get("truncated_body_text", "")


def fetch_comments(pub_slug: str, post_id: int) -> list:
    """Fetch all comments for a post."""
    data = substack_get(
        f"https://{pub_slug}.substack.com/api/v1/post/{post_id}/comments",
        params={"token": "", "all_comments": "true"},
    )
    time.sleep(REQUEST_DELAY)
    if not data:
        return []
    comments = data.get("comments", [])
    # Flatten top-level comments + their children (replies)
    flat = []
    for c in comments:
        flat.append(c)
        flat.extend(c.get("children", []))
    return flat


def build_post_entry(text: str, pub_slug: str, post: dict, sports: list) -> dict:
    post_date = post.get("post_date", "")[:10]
    url = post.get("canonical_url", f"https://{pub_slug}.substack.com/p/{post['slug']}")
    return {
        "text": text[:3000],
        "source": "substack",
        "report_title": f"Substack/{pub_slug}: {post['title'][:120]}",
        "url": url,
        "sports": sports,
        "record_date": post_date,
        "season_phase": "unknown",
        "extra": {
            "publication": pub_slug,
            "post_id": post["id"],
            "post_slug": post["slug"],
            "content_type": "post",
            "audience": post.get("audience", "unknown"),
        },
    }


def build_comment_entry(body: str, pub_slug: str, post: dict, comment: dict, sports: list) -> dict:
    post_date = post.get("post_date", "")[:10]
    url = post.get("canonical_url", f"https://{pub_slug}.substack.com/p/{post['slug']}")
    return {
        "text": body[:2000],
        "source": "substack",
        "report_title": f"Substack/{pub_slug}: {post['title'][:120]}",
        "url": url,
        "sports": sports,
        "record_date": post_date,
        "season_phase": "unknown",
        "extra": {
            "publication": pub_slug,
            "post_id": post["id"],
            "comment_id": comment["id"],
            "content_type": "comment",
        },
    }


def _fetch_post_worker(args) -> tuple:
    """Worker: fetch full body + comments for one post. Returns (post, body, comments)."""
    pub_slug, post = args
    body = ""
    if post.get("audience") == "everyone":
        body = fetch_full_post_body(pub_slug, post["slug"])
    else:
        body = post.get("truncated_body_text", "")

    # Skip the comment API call if the post has no comments — saves a round-trip per post
    if not post.get("comment_count", 0) and not post.get("child_comment_count", 0):
        return post, body, []

    comments = fetch_comments(pub_slug, post["id"])
    return post, body, comments


def scrape_publication(pub_slug: str, sports: list, require_signal: bool) -> list:
    print(f"\n[{pub_slug}] Fetching post list...")
    posts = fetch_all_posts(pub_slug)
    print(f"  → {len(posts)} posts found")
    if not posts:
        return []

    entries = []
    collected_posts = collected_comments = skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_post_worker, (pub_slug, post)): post for post in posts}
        for future in as_completed(futures):
            post, body, comments = future.result()

            # ── Post content ────────────────────────────────────────────
            combined = f"{post.get('title', '')} {post.get('subtitle', '')} {body}".strip()
            if (len(combined) >= MIN_TEXT_LENGTH
                    and is_english(combined)
                    and (not require_signal or has_signal(combined))):
                entries.append(build_post_entry(combined, pub_slug, post, sports))
                collected_posts += 1
            else:
                skipped += 1

            # ── Comments ────────────────────────────────────────────────
            for comment in comments:
                if comment.get("deleted") or comment.get("suppressed"):
                    continue
                cbody = (comment.get("body") or "").strip()
                if (len(cbody) >= MIN_TEXT_LENGTH
                        and is_english(cbody)
                        and (not require_signal or has_signal(cbody))):
                    entries.append(build_comment_entry(cbody, pub_slug, post, comment, sports))
                    collected_comments += 1

    print(f"  → {collected_posts} posts + {collected_comments} comments kept  ({skipped} skipped)")
    return entries


def main():
    print("=" * 55)
    print("  FanVerse Substack Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    all_entries = []
    for pub in PUBLICATIONS:
        entries = scrape_publication(pub["slug"], pub["sports"], pub["require_signal"])
        all_entries.extend(entries)

    print(f"\n── Ingesting {len(all_entries)} total entries ───────────────")
    if all_entries:
        ingest(all_entries)
    else:
        print("Nothing collected.")
    print("\nDone.")


if __name__ == "__main__":
    main()
