#!/usr/bin/env python3
"""
Convert Apify Instagram export to FanVerse repository format.

Reads AllezSports.json (Apify export), filters comments, and converts them
to FanVerse records with proper deduplication and sport tagging.
"""

import json
import hashlib
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Allow input file to be specified as command-line argument, default to AllezSports.json
INPUT_FILE = Path(__file__).parent.parent / "data" / "raw" / (sys.argv[1] if len(sys.argv) > 1 else "AllezSports.json")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"

MIN_TEXT_LENGTH = 10  # Filter out short comments (mostly emojis)


def is_emoji_only(text: str) -> bool:
    """Check if text is mostly emojis/whitespace."""
    # Remove common emoji patterns and check what's left
    cleaned = re.sub(r'[🀀-🿿❤️💪🏻🇺🇸🥌🩷🌟⭐😍🔥👍❤💯✨😊🎉👀🙌😭🎊💫💖💝💗🎯🏅🏆📍]', '', text)
    cleaned = cleaned.strip()
    return len(cleaned) < 5


def make_record_id(source: str, text: str) -> str:
    """Create SHA256 fingerprint for deduplication."""
    raw = f"{source.lower()}::{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def extract_account_name(post_url: str) -> str:
    """Extract Instagram account name from post URL."""
    # Format: https://www.instagram.com/{account}/p/{post_id}/
    try:
        parts = post_url.strip('/').split('/')
        # Find www.instagram.com or instagram.com and get the next part
        for i, part in enumerate(parts):
            if 'instagram.com' in part and i + 1 < len(parts):
                return parts[i + 1]
    except (ValueError, IndexError):
        pass
    return "unknown"


def parse_timestamp(ts_str: str) -> str:
    """Convert ISO timestamp to YYYY-MM-DD date string."""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        return datetime.now().strftime('%Y-%m-%d')


def build_fanverse_record(comment: dict, account_name: str) -> dict:
    """Convert Apify comment to FanVerse record format."""
    text = (comment.get('text') or '').strip()
    timestamp = comment.get('timestamp', '')
    post_url = comment.get('postUrl', '')
    comment_id = comment.get('id', '')
    likes = comment.get('likesCount', 0)

    return {
        "record_id": make_record_id("instagram", text),
        "post_id": str(uuid.uuid4()),
        "text": text,
        "source": "instagram",
        "report_title": f"Instagram: {account_name}",
        "url": post_url,
        "sports": ["general"],  # Will be tagged by ingest.py's rules
        "date": parse_timestamp(timestamp),
        "week": None,
        "season_phase": "unknown",
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "content_type": "comment",
        "score": likes,
    }


def main():
    print("=" * 60)
    print("  FanVerse Instagram Converter (Apify Export)")
    print("=" * 60)

    # Load input
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found")
        return

    with open(INPUT_FILE) as f:
        data = json.load(f)
    print(f"\n[Input] Loaded {len(data)} comments from {INPUT_FILE}")

    # Process comments by account
    records_by_account = defaultdict(list)
    stats = {
        'total': len(data),
        'filtered_emoji': 0,
        'filtered_too_short': 0,
        'kept': 0,
        'by_account': defaultdict(lambda: {'kept': 0, 'filtered': 0}),
    }

    for comment in data:
        text = (comment.get('text') or '').strip()
        post_url = comment.get('postUrl', '')
        account_name = extract_account_name(post_url)

        # Filter: emoji-only
        if is_emoji_only(text):
            stats['filtered_emoji'] += 1
            stats['by_account'][account_name]['filtered'] += 1
            continue

        # Filter: too short
        if len(text) < MIN_TEXT_LENGTH:
            stats['filtered_too_short'] += 1
            stats['by_account'][account_name]['filtered'] += 1
            continue

        # Keep this record
        record = build_fanverse_record(comment, account_name)
        records_by_account[account_name].append(record)
        stats['kept'] += 1
        stats['by_account'][account_name]['kept'] += 1

    # Write output files
    print(f"\n[Output] Writing records to JSONL files...")
    for account, records in records_by_account.items():
        if not records:
            continue

        output_file = OUTPUT_DIR / f"instagram_{account}.jsonl"
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        print(f"  ✓ {output_file} ({len(records)} records)")

    # Print summary
    print(f"\n[Summary]")
    print(f"  Total comments processed:     {stats['total']}")
    print(f"  ✓ Kept (passed filters):      {stats['kept']}")
    print(f"  ✗ Filtered (emoji-only):      {stats['filtered_emoji']}")
    print(f"  ✗ Filtered (too short):       {stats['filtered_too_short']}")
    print(f"  Filtered rate:                {(stats['total'] - stats['kept']) / stats['total'] * 100:.1f}%")

    print(f"\n[By Account]")
    for account in sorted(records_by_account.keys()):
        kept = stats['by_account'][account]['kept']
        filtered = stats['by_account'][account]['filtered']
        total = kept + filtered
        print(f"  {account}:")
        print(f"    ✓ Kept: {kept}/{total}")
        print(f"    ✗ Filtered: {filtered}")

    print(f"\nDone! Ready to ingest with: python3 -c \"from ingest import ingest; ingest([...])\"")


if __name__ == "__main__":
    main()
