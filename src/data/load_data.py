# src/data/load_data.py

import os
import pandas as pd

RAW_DATA_PATH = "data/raw"

def load_tsv(filename: str, nrows: int = None) -> pd.DataFrame:
    """Loads a TSV file into a pandas DataFrame."""
    path = os.path.join(RAW_DATA_PATH, filename)
    print(f"Loading: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False, na_values="\\N", nrows=nrows)

def load_all_raw_data(nrows: int = None) -> dict:
    """Loads all relevant IMDb .tsv files into a dictionary of DataFrames."""
    return {
        "title_basics": load_tsv("title.basics.tsv", nrows),
        "title_ratings": load_tsv("title.ratings.tsv", nrows),
        "title_crew": load_tsv("title.crew.tsv", nrows),
        "title_principals": load_tsv("title.principals.tsv", nrows),
        "name_basics": load_tsv("name.basics.tsv", nrows),
        "title_akas": load_tsv("title.akas.tsv", nrows),
        "title_episode": load_tsv("title.episode.tsv", nrows)
    }

if __name__ == "__main__":
    dfs = load_all_raw_data(nrows=500)
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")
