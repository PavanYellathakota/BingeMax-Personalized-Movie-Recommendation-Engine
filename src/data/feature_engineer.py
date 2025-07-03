# src/data/feature_engineer.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
import os
from scipy.sparse import hstack

def load_merged_data(path="data/processed/merged_data.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

def tfidf_genres(df: pd.DataFrame) -> np.ndarray:
    df["genres"] = df["genres"].fillna("").str.replace(",", " ")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["genres"])
    return tfidf_matrix

def tfidf_title(df: pd.DataFrame) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(df["title"].fillna(""))

def generate_similarity_matrix(tfidf_matrix, save_path="data/processed/similarity_matrix.npz"):
    sim = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim, 0)  # zero out self-similarity

    from scipy.sparse import csr_matrix
    sim_sparse = csr_matrix(sim)

    save_npz(save_path, sim_sparse)
    print(f"✅ Saved similarity matrix to {save_path}")

def save_features(df: pd.DataFrame, path="data/processed/features.parquet"):
    df.to_parquet(path, index=False)
    print(f"✅ Saved features to {path}")


if __name__ == "__main__":
    df = load_merged_data()

    # Compute TF-IDF features (combine genres + title)
    tfidf_genre = tfidf_genres(df)
    tfidf_title_mat = tfidf_title(df)

    # Weighted average (e.g., genres=0.7, title=0.3)
    # combined = tfidf_genre * 0.7 + tfidf_title_mat * 0.3
    combined = hstack([
    tfidf_genre * 0.7,
    tfidf_title_mat * 0.3
])
    # Save similarity matrix
    generate_similarity_matrix(combined)

    # Save movie metadata for API to use
    save_features(df[["tconst", "title", "year", "genres", "rating", "numVotes"]])
