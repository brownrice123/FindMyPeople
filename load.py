import csv
import os
from db import get_client
from datetime import datetime

DATA_DIR = "/Users/brianweiss/Documents/Projects/FindMyPeople/Reddit/reddit/subreddits24"  # Directory where parse_dumps.py writes CSVs
BATCH_SIZE = 500

def load_city_csv(filepath, city):
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
    
    # Upsert users
    user_rows = [
        {"username": u, "home_city": city, "collected_at": datetime.utcnow().isoformat()}
        for u in users_seen
    ]
    for i in range(0, len(user_rows), BATCH_SIZE):
        client.table("users").upsert(user_rows[i:i+BATCH_SIZE], on_conflict="username,home_city").execute()
    
    # Upsert activity
    activity_rows = [
        {"username": k[0], "subreddit": k[1], "comment_count": v}
        for k, v in activity_map.items()
    ]
    for i in range(0, len(activity_rows), BATCH_SIZE):
        client.table("activity").upsert(activity_rows[i:i+BATCH_SIZE], on_conflict="username,subreddit").execute()
    
    print(f"  {city}: {len(user_rows)} users, {len(activity_rows)} activity rows loaded")

if __name__ == "__main__":
    # Test with one city
    load_city_csv("/Users/brianweiss/Documents/Projects/FindMyPeople/Reddit/reddit/subreddits24/seattle_output.csv", "seattle")