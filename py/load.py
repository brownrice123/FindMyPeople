import csv
import os
import subprocess
import time
from db import get_client
from datetime import datetime, timezone
from metros import METROS

DATA_DIR = "/Users/brianweiss/Documents/Projects/FindMyPeople/Reddit/csv/"
DUMPS_DIR = "/Users/brianweiss/Documents/Projects/FindMyPeople/Reddit/torrents/"
PARSED_DIR = "/Users/brianweiss/Documents/Projects/FindMyPeople/Reddit/csv/"
CHECKPOINT_FILE = "/Users/brianweiss/Documents/Projects/FindMyPeople/Reddit/load_checkpoint.txt"
BATCH_SIZE = 500

def upsert_with_retry(client, table_name, rows, on_conflict, max_retries=3):
    """Upsert with exponential backoff on rate limit."""
    for attempt in range(max_retries):
        try:
            client.table(table_name).upsert(rows, on_conflict=on_conflict).execute()
            return
        except Exception as e:
            if "rate" in str(e).lower() or "502" in str(e):
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")

def get_loaded_cities():
    """Read checkpoint file to see which cities are already loaded."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(line.strip() for line in f if line.strip())

def mark_loaded(city):
    """Record that a city has been loaded."""
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(city + "\n")

def load_city_csv(filepath, city, is_city=True):
    """Load a parsed (author, subreddit) CSV for one city into Supabase."""
    client = get_client()

    # Aggregate activity counts in memory before upserting
    users_seen = set()
    activity_map = {}  # (author, subreddit) -> count

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # Expects headers: author, subreddit
        for row in reader:
            author = row['author'].strip()
            subreddit = row['subreddit'].strip().lower()
            if not author or author == '[deleted]':
                continue
            users_seen.add(author)
            key = (author, subreddit)
            activity_map[key] = activity_map.get(key, 0) + 1

    # Upsert users (city subs only)
    if is_city:
        user_rows = [
            {"username": u, "home_city": city, "collected_at": datetime.now(timezone.utc).isoformat()}
            for u in users_seen
        ]
        for i in range(0, len(user_rows), BATCH_SIZE):
            batch = user_rows[i:i+BATCH_SIZE]
            upsert_with_retry(client, "users", batch, on_conflict="username")
    else:
        user_rows = []

    # Upsert activity
    activity_rows = [
        {"username": k[0], "subreddit": k[1], "comment_count": v}
        for k, v in activity_map.items()
    ]
    for i in range(0, len(activity_rows), BATCH_SIZE):
        batch = activity_rows[i:i+BATCH_SIZE]
        upsert_with_retry(client, "activity", batch, on_conflict="username,subreddit")

    print(f"  {city}: {len(user_rows)} users, {len(activity_rows)} activity rows loaded")

def parse_and_load(sub_name, city_label=None, is_city=True):
    """Parse a .zst dump and load it into Supabase. city_label = home_city value."""
    label = city_label or sub_name
    dump_path = os.path.join(DUMPS_DIR, f"{sub_name}_comments.zst")
    csv_path = os.path.join(PARSED_DIR, f"{sub_name}.csv")

    if not os.path.exists(dump_path):
        print(f"  Missing dump: {dump_path} — skipping")
        return

    if not os.path.exists(csv_path):
        print(f"  Parsing {sub_name}...")
        subprocess.run(["python3", "parse_dumps.py", dump_path, csv_path], check=True)

    print(f"  Loading {sub_name} as '{label}'...")
    load_city_csv(csv_path, label, is_city=is_city)
    mark_loaded(sub_name)

if __name__ == "__main__":
    os.makedirs(PARSED_DIR, exist_ok=True)
    loaded = get_loaded_cities()
    
    # City subs — label = city name
    for city in METROS:
        if city in loaded:
            print(f"  Skipping {city} (already loaded)")
            continue
        parse_and_load(city, city_label=city)
    
    # Interest subs — label = subreddit name (no home_city semantics)
    INTEREST_SUBS = [
        "sourdough", "homebrewing", "climbing", "boardgames",
        "datascience", "urbanplanning", "hiking", "cycling",
        # ... add your full list here
    ]
    for sub in INTEREST_SUBS:
        if sub in loaded:
            continue
        parse_and_load(sub, city_label=sub, is_city=False)
    
    print("Done.")