# src/data/preprocess.py

import pandas as pd
#from src.data.load_data import load_all_raw_data
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.load_data import load_all_raw_data

def preprocess_imdb_data(nrows=None) -> pd.DataFrame:
    dfs = load_all_raw_data(nrows)

    # 1. Filter title.basics
    basics = dfs["title_basics"]
    basics = basics[basics["titleType"] == "movie"]
    basics = basics[basics["isAdult"] == 0]
    basics = basics.dropna(subset=["primaryTitle", "startYear", "genres"])

    # Convert types
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
    basics = basics.dropna(subset=["startYear"])

    # 2. Merge with title.ratings
    ratings = dfs["title_ratings"]
    ratings["numVotes"] = pd.to_numeric(ratings["numVotes"], errors="coerce")
    ratings = ratings[ratings["numVotes"] > 1000]
    merged = pd.merge(basics, ratings, on="tconst")

    # 3. Merge with title.crew (optional for now)
    crew = dfs["title_crew"]
    merged = pd.merge(merged, crew, on="tconst", how="left")

    # 4. Rename and select relevant columns
    merged = merged.rename(columns={
        "primaryTitle": "title",
        "startYear": "year",
        "averageRating": "rating"
    })

    merged = merged[["tconst", "title", "year", "genres", "rating", "numVotes", "directors", "writers"]]

    return merged


def save_processed_data(df: pd.DataFrame, path="data/processed/merged_data.parquet"):
    print(f"Saving cleaned data to {path} ...")
    df.to_parquet(path, index=False)
    print("✅ Saved.")


if __name__ == "__main__":
    df = preprocess_imdb_data(nrows=100000)
    save_processed_data(df)
    print(df.head())
