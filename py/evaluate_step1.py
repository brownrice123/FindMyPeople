import pandas as pd

OUTPUT_DIR = "/Users/brianweiss/Projects/FindMyPeople/Reddit/data/outputs/"

# Load the PMI interest × city matrix (rows=subreddits, cols=cities)
pmi = pd.read_csv(OUTPUT_DIR + "pmi_interest_x_city.csv", index_col=0)

for city in ["seattle", "chicago", "austin"]:
    if city not in pmi.columns:
        print(f"\n=== r/{city} === NOT FOUND IN DATA")
        continue
    print(f"\n=== r/{city} ===")
    top10 = pmi[city].sort_values(ascending=False).head(10)
    for sub, score in top10.items():
        print(f"  r/{sub}: {score:.4f}")