import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

FEATURES_PATH = "data/processed/features.parquet"
SIM_MATRIX_PATH = "data/processed/similarity_matrix.npz"

class ContentBasedRecommender:  # ✅ Renamed to match the import
    def __init__(self):
        print("🔄 Loading features and sparse similarity matrix...")
        self.features = pd.read_parquet(FEATURES_PATH)
        self.sim_matrix = load_npz(SIM_MATRIX_PATH)
        self.movie_index = {tconst: idx for idx, tconst in enumerate(self.features["tconst"])}
        print("✅ Content-based model loaded.")

    def _get_similarities(self, idx):
        sim_row = cosine_similarity(self.sim_matrix[idx], self.sim_matrix)
        return sim_row.flatten()

    def get_recommendations(self, movie_id: str, top_k: int = 10):
        if movie_id not in self.movie_index:
            raise ValueError(f"Movie ID '{movie_id}' not found.")
        idx = self.movie_index[movie_id]
        sim_scores = self._get_similarities(idx)
        sim_scores[idx] = 0
        top_indices = sim_scores.argsort()[::-1][:top_k]
        results = self.features.iloc[top_indices][["tconst", "title", "year", "rating", "genres"]].copy()
        results["similarity_score"] = sim_scores[top_indices]
        return results

    def get_content_vector_score(self, movie_id: str):
        if movie_id not in self.movie_index:
            return 0.0
        idx = self.movie_index[movie_id]
        row = self.sim_matrix[idx]
        return row.multiply(row).sum()


if __name__ == "__main__":
    rec = ContentBasedRecommender()
    movie_id = rec.features["tconst"].iloc[0]
    print(f"Recommendations for {movie_id}:")
    print(rec.get_recommendations(movie_id, top_k=5))
