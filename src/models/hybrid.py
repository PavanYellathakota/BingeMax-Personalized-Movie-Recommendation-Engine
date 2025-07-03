# src/models/hybrid.py

from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
import pandas as pd

class HybridRecommender:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.collab_model = CollaborativeRecommender()
        self.content_model = None  # Lazy load for memory

    def get_hybrid_recommendations(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        # Load content model only when needed
        if self.content_model is None:
            self.content_model = ContentBasedRecommender()

        collab_df = self.collab_model.get_user_recommendations(user_id, top_k=top_k * 2)
        collab_df = collab_df.dropna(subset=["tconst", "score"])

        results = []
        for _, row in collab_df.iterrows():
            movie_id = row["tconst"]
            collab_score = row["score"]

            try:
                content_score = self.content_model.get_content_vector_score(movie_id)
            except:
                content_score = 0.0

            final_score = self.alpha * content_score + (1 - self.alpha) * collab_score

            results.append({
                "tconst": row["tconst"],
                "title": row["title"],
                "year": row["year"],
                "rating": row["rating"],
                "genres": row["genres"],
                "score": final_score
            })

        result_df = pd.DataFrame(results).sort_values("score", ascending=False).head(top_k)
        return result_df
