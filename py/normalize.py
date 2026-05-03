import os
import numpy as np
import pandas as pd
import duckdb

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from metros import METROS

CSV_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/csv/"
CSV_CITIES_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/csv/cities/"
CSV_INTERESTS_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/csv/interests/"
OUTPUT_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/data/outputs/"

METROS_SET = {m.lower() for m in METROS}

BLOCKLIST = {
    "askreddit", "funny", "worldnews", "news", "todayilearned",
    "pics", "videos", "gaming", "movies", "music", "science",
    "iama", "showerthoughts", "tifu", "mildlyinteresting",
    "lifeprotips", "bestof", "gifs", "explainlikeimfive"
}

GEO_BLOCKLIST = {
    # US states and regions
    "washington", "oregon", "california", "texas", "colorado", "illinois",
    "florida", "newyork", "massachusetts", "virginia", "georgia", "ohio",
    "michigan", "pennsylvania", "arizona", "northcarolina", "tennessee",
    "missouri", "minnesota", "wisconsin", "maryland", "nevada", "utah",
    "indiana", "kentucky", "alabama", "louisiana", "southcarolina",
    "newmexico", "kansas", "iowa", "oklahoma", "arkansas", "mississippi",
    "pacificnorthwest", "southwest", "midwest", "southeast", "northeast",

    # Cities/metros not in METROS_SET (seen in output or likely to appear)
    "portlandor", "portlandoregon", "greaterlosangeles", "sanfrancisco",
    "newyorkcity", "washingtondc", "nyc", "cleveland", "columbus",
    "columbusga", "charlottesville", "bayarea", "socal", "norcal",
    "seattle", "chicago", "austin",  # your own city subs
}

# Maps subreddit aliases → canonical name.
# Applied after loading so both CSV files contribute to the same interest bucket.
# Originals are preserved on disk — this only affects in-memory grouping.
#
# OBVIOUS MERGES: same community, different subreddit name
CANONICAL_SUBS = {
    "motorcycle":        "motorcycles",        # r/motorcycle and r/motorcycles are the same crowd
    "mountainbikes":     "mountainbiking",     # r/mountainbikes vs r/mountainbiking
    "warhammer40k":      "warhammer",          # 40k is the dominant game; merge into warhammer
    "homestead":         "homesteading",       # r/homestead vs r/homesteading
    "bicycling":         "cycling",            # r/bicycling (casual/commute) merged into r/cycling

    # JUDGMENT CALLS: related but meaningfully distinct — review before confirming
    # Uncomment to merge, leave commented to keep separate.

    # Bike touring vs bikepacking: similar ethos, significant user overlap
    # "bikepacking": "bicycletouring",

    # Kayaking and whitewater: large overlap but whitewater is more specific
    # "whitewater": "kayaking",

    # Fishing and fly fishing: distinct enough that merging may dilute signal
    # "flyfishing": "fishing",

    # DIY electronics cluster: diyaudio/diyelectronics/diypedals share a lot of users
    # "diyaudio":      "diyelectronics",
    # "diypedals":     "diyelectronics",

    # Fermentation cluster: homebrewing/mead/fermentation/cheesemaking are related
    # but distinct hobbies — merging loses specificity
    # "mead":          "homebrewing",
    # "fermentation":  "homebrewing",

    # Running: trail running and road running are different enough to keep separate
    # "trailrunning": "running",

    # Skiing: downhill and XC are very different communities
    # "xcountryskiing": "skiing",
}




def load_data():
    con = duckdb.connect()

    # Load city CSVs — filename becomes home_city
    city_df = con.execute(f"""
        SELECT
            trim(author) AS author,
            lower(regexp_replace(filename, '.*/([^/]+)\\.csv$', '\\1')) AS home_city
        FROM read_csv(
            '{CSV_CITIES_DIR}*.csv',
            columns={{'author': 'VARCHAR', 'subreddit': 'VARCHAR'}},
            filename=true,
            ignore_errors=true
        )
        WHERE author IS NOT NULL
          AND trim(author) != '[deleted]'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY trim(author)) = 1
    """).df()

    # Load interest CSVs
    interest_df = con.execute(f"""
        SELECT
            trim(author) AS author,
            lower(trim(subreddit)) AS subreddit
        FROM read_csv(
            '{CSV_INTERESTS_DIR}*.csv',
            columns={{'author': 'VARCHAR', 'subreddit': 'VARCHAR'}},
            ignore_errors=true
        )
        WHERE author IS NOT NULL
          AND trim(author) != '[deleted]'
    """).df()

    con.close()

    df = (interest_df
          .merge(city_df[['author', 'home_city']], on='author')
          .rename(columns={'author': 'username'})
          .drop_duplicates(['username', 'subreddit']))
    df = df[~df['subreddit'].isin(BLOCKLIST)]
    df = df[~df['subreddit'].isin(METROS_SET)]
    df = df[~df['subreddit'].isin(GEO_BLOCKLIST)]
    df['subreddit'] = df['subreddit'].map(lambda s: CANONICAL_SUBS.get(s, s))
    df = df.drop_duplicates(['username', 'subreddit'])
    return df

# ── interest × city ──────────────────────────────────────────────────────────

def make_interest_x_city_raw(df):
    return (df.groupby(['subreddit', 'home_city'])['username']
              .nunique()
              .unstack(fill_value=0))


def make_interest_x_city_tfidf(pivot):
    # Cities as documents, interest subs as terms
    cities_docs = pivot.T
    tfidf = TfidfTransformer().fit_transform(cities_docs.values)
    result = pd.DataFrame(tfidf.toarray(), index=cities_docs.index, columns=cities_docs.columns)
    return result.T


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


# ── interest × interest ───────────────────────────────────────────────────────

def make_interest_x_interest(df, method):
    user_sub = df.pivot_table(index='username', columns='subreddit', aggfunc='size', fill_value=0)
    subs = user_sub.columns

    if method == 'raw':
        mat = (user_sub.T @ user_sub).values.astype(float)

    elif method == 'tfidf':
        tfidf = TfidfTransformer().fit_transform(user_sub.values)
        mat = cosine_similarity(tfidf.T)

    elif method == 'pmi':
        n = len(user_sub)
        cooc = (user_sub.T @ user_sub).values / n
        p_s = user_sub.sum(axis=0).values / n
        expected = np.outer(p_s, p_s)
        with np.errstate(divide='ignore', invalid='ignore'):
            mat = np.where(cooc > 0, np.log2(cooc / expected), 0)

    np.fill_diagonal(mat, 0)
    return pd.DataFrame(mat, index=subs, columns=subs)


# ── city × city ───────────────────────────────────────────────────────────────

def make_city_x_city(ixc):
    cities = ixc.T  # rows=cities, cols=interest subs
    sim = cosine_similarity(cities.values)
    np.fill_diagonal(sim, 0)
    return pd.DataFrame(sim, index=cities.index, columns=cities.index)


# ── printing ──────────────────────────────────────────────────────────────────

def print_seattle(label, table):
    print(f"\n=== {label} — top 10 for Seattle ===")
    if 'seattle' in table.columns:
        print(table['seattle'].sort_values(ascending=False).head(10).to_string())
    elif 'seattle' in table.index:
        print(table.loc['seattle'].sort_values(ascending=False).head(10).to_string())
    else:
        # interest_x_interest has no city dimension — show top global pairs instead
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

    # interest × city
    raw_ixc = make_interest_x_city_raw(df)
    # TF-IDF degenerated on Phase 0 data (all subs present in all cities → IDF constant).
    # PMI chosen as primary method. Function retained for reference.
    # tfidf_ixc = make_interest_x_city_tfidf(raw_ixc)
    pmi_ixc = make_interest_x_city_pmi(raw_ixc, df)

    for name, tbl in [('raw', raw_ixc), ('pmi', pmi_ixc)]:
        tbl.to_csv(os.path.join(OUTPUT_DIR, f"{name}_interest_x_city.csv"))
        print_seattle(f"{name} interest_x_city", tbl)

    # interest × interest
    for method in ('raw', 'pmi'):
        tbl = make_interest_x_interest(df, method)
        tbl.to_csv(os.path.join(OUTPUT_DIR, f"{method}_interest_x_interest.csv"))
        print_seattle(f"{method} interest_x_interest", tbl)

    # city × city
    for name, ixc in [('raw', raw_ixc), ('pmi', pmi_ixc)]:
        tbl = make_city_x_city(ixc)
        tbl.to_csv(os.path.join(OUTPUT_DIR, f"{name}_city_x_city.csv"))
        print_seattle(f"{name} city_x_city", tbl)

    print(f"\nSaved 9 CSVs to {OUTPUT_DIR}")
