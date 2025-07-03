# src/models/collaborative.py

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


class CollaborativeRecommender:
    def __init__(self,
                 interactions_path="data/external/user_interactions.csv",
                 movie_path="data/processed/features.parquet"):
        print("🔄 Loading interaction and movie data...")

        if not os.path.exists(interactions_path):
            raise FileNotFoundError(
                f"❌ File not found: '{interactions_path}'.\n"
                f"📌 To use Collaborative Recommender, please provide user_interactions.csv in data/external/"
            )

        if not os.path.exists(movie_path):
            raise FileNotFoundError(
                f"❌ File not found: '{movie_path}'.\n"
                f"📌 Make sure feature engineering is completed and saved to data/processed/features.parquet."
            )

        self.interactions = pd.read_csv(interactions_path)
        self.movies = pd.read_parquet(movie_path)

        self.user_ids = self.interactions["user_id"].unique()
        self.movie_ids = self.interactions["movie_id"].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(self.movie_ids)}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}

        self.user_item_matrix = self._build_sparse_matrix()
        self.model = self._train_model()

    def _build_sparse_matrix(self):
        print("📊 Building user-item interaction matrix...")
        rows = self.interactions["user_id"].map(self.user_to_idx)
        cols = self.interactions["movie_id"].map(self.movie_to_idx)
        data = self.interactions["interaction"]

        return csr_matrix((data, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids)))

    def _train_model(self, factors=50, regularization=0.01, iterations=15):
        print("🧠 Training ALS model...")
        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations)
        model.fit(self.user_item_matrix.T)  # Transpose for implicit
        print("✅ ALS training complete.")
        return model

    def get_user_recommendations(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID '{user_id}' not found.")

        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix[user_idx]

        item_scores = np.array(self.model.recommend(user_idx, user_vector, N=top_k))
        recommended_ids = [self.idx_to_movie[int(i)] for i in item_scores[:, 0]]
        scores = item_scores[:, 1].tolist()

        result_df = self.movies[self.movies["tconst"].isin(recommended_ids)].copy()
        result_df["score"] = result_df["tconst"].map(dict(zip(recommended_ids, scores)))
        result_df = result_df.sort_values("score", ascending=False)

        return result_df[["tconst", "title", "year", "rating", "genres", "score"]]


if __name__ == "__main__":
    rec = CollaborativeRecommender()
    sample_user = rec.user_ids[0]
    print(f"🔍 Recommendations for user: {sample_user}")
    print(rec.get_user_recommendations(sample_user, top_k=5))
