# scripts/generate_user_data.py

import pandas as pd
import numpy as np
import random
import os

RAW_MOVIES_PATH = "data/processed/merged_data.parquet"
OUTPUT_PATH = "data/external/user_interactions.csv"

def generate_user_interactions(num_users=100, min_movies=10, max_movies=30):
    df = pd.read_parquet(RAW_MOVIES_PATH)
    movie_ids = df["tconst"].tolist()

    data = []

    for i in range(1, num_users + 1):
        user_id = f"u{i}"
        watched = random.sample(movie_ids, random.randint(min_movies, max_movies))
        for movie_id in watched:
            data.append({"user_id": user_id, "movie_id": movie_id, "interaction": 1})

    interactions_df = pd.DataFrame(data)
    interactions_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Simulated user data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_user_interactions()
