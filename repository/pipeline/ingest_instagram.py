from pathlib import Path
from ingest import ingest
import json

all_records = []
for account_file in [
    'instagram_allezsportstv.jsonl',
    'instagram_sideoutsociety.jsonl',
    'instagram_strongerthanyouthink.jsonl',
    'instagram_alitruwit.jsonl',
    'instagram_watchme_sportsbar.jsonl',
    'instagram_ktrain115.jsonl',
    'instagram_the_officialjade297.jsonl',
    'instagram_youseela_.jsonl',
]:
    path = Path(account_file)
    if path.exists():
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))

print(f"Total records to ingest: {len(all_records)}")
ingest(all_records)