# scraper_research.py — Research firm report scraper.
# Fetches real content from known report URLs and ingests into the repository.
# Run: python3 scraper_research.py
# Install: pip install requests beautifulsoup4 python-dotenv

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from ingest import ingest

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FanVerse-scraper/0.1)"}
REQUEST_DELAY = 2.0

# Add new sources here as new reports are found. Each entry is one repository record.
RESEARCH_SOURCES = [
    {
        "url": "https://the.team/news/the-team-the-collective-launches-research-proving-that-women-are-driving-the-global-growth-in-sports-fandom/",
        "source": "wasserman",
        "report_title": "Her Love of the Game: A Global Overview of Female Fans of Sports",
        "sports": ["general"],
        "date": "2024-10-02",
    },
    {
        "url": "https://www.deloitte.com/us/en/insights/industry/technology/female-sports-fans-engagement.html",
        "source": "deloitte",
        "report_title": "Women Sports Fans Are Just as Engaged—If Not More—Than Men",
        "sports": ["general"],
        "date": "2023-06-01",
    },
    {
        "url": "https://www.nielsen.com/insights/2024/whats-next-for-womens-sports-fueling-growth-proving-value/",
        "source": "nielsen",
        "report_title": "What's Next for Women's Sports: Fueling Growth by Proving Value",
        "sports": ["general"],
        "date": "2024-01-01",
    },
    {
        "url": "https://www.bcg.com/publications/2023/accelarating-canadian-womens-professional-sports",
        "source": "bcg",
        "report_title": "Accelerating Canadian Women's Professional Sports",
        "sports": ["general"],
        "date": "2023-01-01",
    },
]

def fetch_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove nav, footer, scripts, styles
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    # Get main content — try article/main first, fall back to body
    main = soup.find("article") or soup.find("main") or soup.find("body")
    return " ".join(main.get_text(separator=" ").split()) if main else ""

def build_entry(text: str, source: dict) -> dict:
    return {
        "text": text[:5000], # cap it at 5000 chars, it should be enough for signal extraction
        "source": source["source"],
        "report_title": source["report_title"],
        "url": source["url"],
        "sports": source["sports"],
        "record_date": source["date"],
        "season_phase": "unknown",
        "extra": {"content_type": "research_report"},
    }

def main():
    print("=" * 55)
    print("  FanVerse Research Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    entries = []
    for source in RESEARCH_SOURCES:
        print(f"\n[{source['source']}] Fetching {source['url'][:60]}...")
        try:
            text = fetch_text(source["url"])
            if len(text) < 200:
                print(f"  x Too short ({len(text)} chars) — skipping")
                continue
            entries.append(build_entry(text, source))
            print(f"  + {len(text):,} chars extracted")
        except Exception as e:
            print(f"  x Failed: {e}")
        time.sleep(REQUEST_DELAY)

    print(f"\n── Ingesting {len(entries)} research records ───────────────")
    if entries:
        ingest(entries)
    else:
        print("Nothing collected.")
    print("\nDone.")