import pandas as pd
import sys

# ---- Config ----
CITY_SUBS = ["Seattle", "chicago", "Austin"]
INTEREST_SUBS = ["sourdough", "homebrewing", "climbing", "boardgames", "datascience", "urbanplanning"]
CSV_FILES = [
    "seattle_output.csv",
    "chicago_output.csv",
    "austin_output.csv",
    "sourdough_output.csv",
    "homebrewing_output.csv",
    "climbing_output.csv",
    "boardgames_output.csv",
    "datascience_output.csv",
    "urbanplanning_output.csv",
]

# ---- Load all CSVs ----
print("Loading files...")
df = pd.concat([pd.read_csv(f) for f in CSV_FILES], ignore_index=True)
print(f"Total rows: {len(df):,}")

# ---- For each city, find crossovers ----
for city in CITY_SUBS:
    city_users = set(df[df["subreddit"] == city]["author"])
    print(f"\n--- {city} ({len(city_users):,} unique users) ---")

    for interest in INTEREST_SUBS:
        interest_users = set(df[df["subreddit"] == interest]["author"])
        overlap = city_users & interest_users
        print(f"  r/{interest}: {len(overlap):,} users in common")