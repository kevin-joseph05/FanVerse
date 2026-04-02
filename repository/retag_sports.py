# retag_sports.py — One-time migration to fix sport tags on existing records.
# Re-runs the infer_sports_from_context logic over repository.jsonl in place.
# Run: python3 retag_sports.py

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ingest import infer_sports_from_context

REPO_PATH = Path("repository.jsonl")

if not REPO_PATH.exists():
    print("repository.jsonl not found")
    sys.exit(1)

records = []
with open(REPO_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

changed = 0
for r in records:
    # Build a fake entry dict that infer_sports_from_context understands
    entry = {
        "sports": r.get("sports", ["general"]),
        "report_title": r.get("report_title", ""),
        "text": r.get("text", ""),
        "extra": {k: r.get(k) for k in ("subreddit", "publication") if r.get(k)},
    }
    new_sports = infer_sports_from_context(entry)
    if new_sports != r.get("sports"):
        r["sports"] = new_sports
        changed += 1

# Write back
with open(REPO_PATH, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Re-tagged {changed}/{len(records)} records.")

# Show breakdown
from collections import Counter
sports_counts = Counter(s for r in records for s in r.get("sports", []))
print("New sport distribution:")
for sport, n in sports_counts.most_common():
    print(f"  {n:5d}  {sport}")
