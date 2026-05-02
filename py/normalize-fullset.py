import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from metros import METROS

CSV_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/csv/"
OUTPUT_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/data/outputs/"

# ── tunables ──────────────────────────────────────────────────────────────────
# Subreddits with fewer than this many co-members (users who appear in both
# a city sub AND this interest sub) are dropped before any matrix math.
# Statistical floor + memory control in one knob.
MIN_COMEMBERS = 10

BLOCKLIST = {
    "askreddit", "funny", "worldnews", "news", "todayilearned",
    "pics", "videos", "gaming", "movies", "music", "science",
    "iama", "showerthoughts", "tifu", "mildlyinteresting",
    "lifeprotips", "bestof", "gifs", "explainlikeimfive",
}

METROS_SET = {m.lower() for m in METROS}


# ── data loading ─────────────────────────────────────────────────────────────
#
# Two-pass approach to stay under 8 GB RAM:
#
# Pass 1: Read all city CSVs, build a set of city users + their home city.
#          City CSVs are small (one per metro) so this fits easily.
#
# Pass 2: Read interest CSVs ONE AT A TIME. For each file:
#          - inner-join against city users immediately (drops 90%+ of rows)
#          - append only the matched rows to the result list
#          - the raw file gets garbage-collected before the next one loads
#
# Net effect: peak memory ≈ city_df + one interest CSV + accumulated matches.
# The accumulated matches are small because most interest sub users aren't
# in any city sub.

def load_data():
    city_files = []
    interest_files = []

    # Sort files into city vs interest
    for fname in sorted(os.listdir(CSV_DIR)):
        if not fname.endswith('.csv') or '_output' in fname:
            continue
        stem = fname.removesuffix('.csv').lower()
        fpath = os.path.join(CSV_DIR, fname)
        if stem in METROS_SET:
            city_files.append((fpath, stem))
        elif stem.lower() not in BLOCKLIST:
            interest_files.append(fpath)

    # Pass 1: build city user lookup
    print(f"  Pass 1: loading {len(city_files)} city files...")
    city_frames = []
    for fpath, city in city_files:
        raw = pd.read_csv(fpath, usecols=['author', 'subreddit'])
        raw = raw[raw['author'].notna() & (raw['author'].str.strip() != '[deleted]')]
        raw['author'] = raw['author'].str.strip()
        raw['home_city'] = city
        city_frames.append(raw[['author', 'home_city']].drop_duplicates('author'))

    city_df = pd.concat(city_frames, ignore_index=True).drop_duplicates('author')
    city_users = set(city_df['author'])
    print(f"  {len(city_users)} unique city users across {len(city_files)} metros")

    # Pass 2: stream interest files, merge immediately, keep only matches
    print(f"  Pass 2: streaming {len(interest_files)} interest files...")
    matched_frames = []
    for i, fpath in enumerate(interest_files):
        fname = os.path.basename(fpath)
        raw = pd.read_csv(fpath, usecols=['author', 'subreddit'])
        raw = raw[raw['author'].notna() & (raw['author'].str.strip() != '[deleted]')]
        raw['author'] = raw['author'].str.strip()
        raw['subreddit'] = raw['subreddit'].str.strip().str.lower()

        # Filter to city users immediately — this is what keeps memory down
        matched = raw[raw['author'].isin(city_users)][['author', 'subreddit']]
        matched = matched.drop_duplicates()

        if not matched.empty:
            print(f"    [{i+1}/{len(interest_files)}] {fname}: "
                  f"{len(raw)} rows -> {len(matched)} matched")
            matched_frames.append(matched)
        else:
            print(f"    [{i+1}/{len(interest_files)}] {fname}: 0 matches, skipping")

        del raw, matched  # free memory before next file

    interest_df = pd.concat(matched_frames, ignore_index=True)
    del matched_frames

    # Merge to attach home_city
    df = (interest_df
          .merge(city_df, on='author')
          .rename(columns={'author': 'username'})
          .drop_duplicates(['username', 'subreddit']))
    df = df[~df['subreddit'].isin(BLOCKLIST)]
    df = df[~df['subreddit'].isin(METROS_SET)]

    # Drop low-signal subreddits before any matrix construction
    sub_counts = df.groupby('subreddit')['username'].nunique()
    keep_subs = sub_counts[sub_counts >= MIN_COMEMBERS].index
    before = df['subreddit'].nunique()
    df = df[df['subreddit'].isin(keep_subs)]
    after = df['subreddit'].nunique()
    print(f"  MIN_COMEMBERS filter ({MIN_COMEMBERS}): {before} -> {after} subreddits")

    return df


# ── interest x city ──────────────────────────────────────────────────────────

def make_interest_x_city_raw(df):
    return (df.groupby(['subreddit', 'home_city'])['username']
              .nunique()
              .unstack(fill_value=0))


def make_interest_x_city_pmi(pivot, df):
    total = df['username'].nunique()
    p_cs = pivot / total
    p_c = df.groupby('home_city')['username'].nunique() / total
    p_s = df.groupby('subreddit')['username'].nunique() / total
    expected = pd.DataFrame(
        np.outer(p_s.loc[pivot.index], p_c.loc[pivot.columns]),
        index=pivot.index, columns=pivot.columns
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.where(p_cs > 0, np.log2(p_cs / expected), 0)
    return pd.DataFrame(pmi, index=pivot.index, columns=pivot.columns)


# ── interest x interest ───────────────────────────────────────────────────────

def make_interest_x_interest_pmi(df):
    user_cat = df['username'].astype('category')
    sub_cat  = df['subreddit'].astype('category')
    sub_names = sub_cat.cat.categories

    n_users = user_cat.cat.categories.size
    n_subs  = sub_names.size

    print(f"  interest x interest: {n_users} users x {n_subs} subs "
          f"(dense output ~ {n_subs**2 * 8 / 1e6:.0f} MB)")

    X = csr_matrix(
        (np.ones(len(df), dtype=np.float32),
         (user_cat.cat.codes.values, sub_cat.cat.codes.values)),
        shape=(n_users, n_subs)
    )
    X.data[:] = 1.0

    cooc = (X.T @ X).toarray() / n_users
    p_s = np.asarray(X.sum(axis=0)).flatten() / n_users
    expected = np.outer(p_s, p_s)

    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(cooc > 0, np.log2(cooc / expected), 0)

    np.fill_diagonal(mat, 0)
    return pd.DataFrame(mat, index=sub_names, columns=sub_names)


# ── city x city ───────────────────────────────────────────────────────────────

def make_city_x_city(ixc):
    cities = ixc.T
    sim = cosine_similarity(cities.values)
    np.fill_diagonal(sim, 0)
    return pd.DataFrame(sim, index=cities.index, columns=cities.index)


# ── printing ──────────────────────────────────────────────────────────────────

def print_top(label, table, city=None):
    print(f"\n=== {label} ===")
    if city and city in table.columns:
        print(table[city].sort_values(ascending=False).head(10).to_string())
    elif city and city in table.index:
        print(table.loc[city].sort_values(ascending=False).head(10).to_string())
    else:
        stacked = (table.where(np.triu(np.ones(table.shape, dtype=bool), k=1))
                        .stack()
                        .sort_values(ascending=False))
        print(stacked.head(10).to_string())


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows | {df['username'].nunique()} users | "
          f"{df['home_city'].nunique()} cities | {df['subreddit'].nunique()} interest subs")

    if df.empty:
        print("No overlapping users between city and interest subs — check your CSVs.")
        raise SystemExit(1)

    # interest x city
    raw_ixc = make_interest_x_city_raw(df)
    pmi_ixc = make_interest_x_city_pmi(raw_ixc, df)

    for name, tbl in [('raw', raw_ixc), ('pmi', pmi_ixc)]:
        tbl.to_csv(os.path.join(OUTPUT_DIR, f"{name}_interest_x_city.csv"))
        print_top(f"{name} interest_x_city — top 10 Seattle", tbl, city='seattle')

    # interest x interest (PMI, sparse build)
    ixi_pmi = make_interest_x_interest_pmi(df)
    ixi_pmi.to_csv(os.path.join(OUTPUT_DIR, "pmi_interest_x_interest.csv"))
    print_top("pmi interest_x_interest — top global pairs", ixi_pmi)

    # city x city
    for name, ixc in [('raw', raw_ixc), ('pmi', pmi_ixc)]:
        tbl = make_city_x_city(ixc)
        tbl.to_csv(os.path.join(OUTPUT_DIR, f"{name}_city_x_city.csv"))
        print_top(f"{name} city_x_city — top 10 Seattle", tbl, city='seattle')

    print(f"\nSaved CSVs to {OUTPUT_DIR}")
