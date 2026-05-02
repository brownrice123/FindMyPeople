# FindMyPeople — Reddit Pipeline

## Project overview
Match people across cities based on shared Reddit interests. Pipeline: parse Pushshift dumps → build user-interest and user-city mappings → compute similarity matrices (PMI) → store results.

## Directory layout
```
Reddit/
  py/               # all Python scripts
  csv/              # parsed CSVs (author, subreddit per file)
  torrents/subreddits25/  # raw .zst Pushshift dumps (consumed and deleted by parse_dumps.py)
  data/outputs/     # computed matrices (CSV)
```

## Key scripts

### parse_dumps.py
Parses Pushshift `.zst` comment dumps → `(author, subreddit)` CSVs.
- **No args**: processes all `.zst` files in `torrents/subreddits25/`, deletes each after successful parse.
- **One arg**: `python parse_dumps.py Austin_comments.zst` — processes a single file.
- Delete-after-parse means re-running is always safe (no repeated work).

### normalize-smallset.py
Original dense pipeline — safe for the current small sample dataset (6 interest subs × 4 cities). Will OOM on full data.

### normalize-fullset.py
Sparse-matrix pipeline for the full dataset. Key difference: `make_interest_x_interest_pmi` builds a `scipy.sparse.csr_matrix` instead of a dense `pivot_table`, so memory scales with actual (user, sub) pairs rather than the full grid.
- Only computes PMI (not raw or TF-IDF).
- **Known issue**: `(X.T @ X).toarray()` still materializes a dense n_subs × n_subs matrix — can OOM if n_subs is large (e.g. 50k subs → ~10 GB).

## Data flow
1. City CSVs (e.g. `seattle.csv`) — rows with `author` mapped to `home_city`
2. Interest CSVs (e.g. `climbing.csv`) — rows with `author` and `subreddit`
3. `load_data()` merges on `author`, returns `(username, subreddit, home_city)` deduplicated
4. Three matrix types computed:
   - **interest × city** — which subs are over/under-represented per city (PMI)
   - **interest × interest** — sub-to-sub co-occurrence similarity (PMI)
   - **city × city** — cosine similarity between cities based on interest vectors

## Known issues / pending work

### OOM on full dataset
`normalize-fullset.py` was killed by the OS before completing. Two likely causes:
1. `(X.T @ X).toarray()` in `make_interest_x_interest_pmi` — dense allocation of n_subs² matrix
2. `load_data()` concatenating all raw CSV rows into RAM before aggregating

**Planned fixes (not yet implemented):**
- Min-user filter: drop subs with fewer than N total users before building the matrix
- Keep co-occurrence matrix sparse — compute PMI element-wise without calling `.toarray()`
- Stream `load_data()` — aggregate counts incrementally instead of concatenating raw rows
- Longer term: push aggregates into Supabase and query from there

### max_df filter
Filter to drop subreddits present in ≥80% of cities (they carry no city-specific signal, similar to blocklist entries). Logic is correct but vacuous on small sample data — every sub appears in all 4 cities. Will be meaningful on the full 25-city dataset.

### normalize.py CSV path
`normalize.py` still has a stale `CSV_DIR` pointing to `/Users/brianweiss/Documents/...`. The correct path is `/Users/brianweiss/Projects/FindMyPeople/Reddit/csv/`. The `-smallset` and `-fullset` variants have the correct path.

## Analysis findings (small dataset)
- Seattle: climbing > boardgames > sourdough (top PMI)
- Chicago: urbanplanning > datascience > homebrewing
- Austin: datascience > climbing > homebrewing
- All 4 cities well above 50-user threshold (Seattle ~8,265, Philadelphia ~3,556)
- interest×interest and universal-top-10 results are vacuous at 6 subs — need full data

## Metros
25 cities defined in `metros.py`. City CSVs must match these names exactly (lowercase).
