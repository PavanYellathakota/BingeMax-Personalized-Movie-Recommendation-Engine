# src/api/main.py
from fastapi import FastAPI, HTTPException, Query, Path
from typing import List
import pandas as pd
import random
import mlflow
from mlflow.tracking import MlflowClient
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender

app = FastAPI()
content_recommender = ContentBasedRecommender()
mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()
MODEL_NAME = "als-model"

try:
    collab_recommender = CollaborativeRecommender()
except FileNotFoundError as e:
    collab_recommender = None
    print(f"⚠️ Collaborative model not loaded: {e}")

hybrid_recommender = HybridRecommender(alpha=0.5)

@app.get("/")
def root():
    return {"message": "Welcome to the Movie Recommendation API!"}

@app.get("/titles")
def get_all_titles() -> List[str]:
    df = content_recommender.features.copy()
    if df.empty:
        raise HTTPException(status_code=500, detail="No features found.")
    return df["title"].dropna().sort_values().unique().tolist()

@app.get("/recommend")
def recommend(title: str = Query(...), top_k: int = 10) -> List[dict]:
    df = content_recommender.features
    match = df[df["title"].str.lower().str.strip() == title.strip().lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Movie title '{title}' not found.")
    movie_id = match.iloc[0]["tconst"]
    recs = content_recommender.get_recommendations(movie_id, top_k)
    return recs.drop(columns=["tconst"]).to_dict(orient="records")

@app.get("/recommend_user")
def recommend_user(user_id: str = Query(...), top_k: int = 10) -> List[dict]:
    try:
        recs = collab_recommender.get_user_recommendations(user_id, top_k)
        return recs.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/search")
def search_movies(title: str = Query(...), max_results: int = 10) -> List[dict]:
    df = content_recommender.features.copy()
    df = df[df["title"].notnull()]
    matches = df[df["title"].str.lower().str.contains(title.strip().lower(), na=False)]
    matches = matches[["tconst", "title", "year", "rating", "genres"]].head(max_results)
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No matches found for title: '{title}'")
    return matches.to_dict(orient="records")

@app.get("/search_by_name")
def search_by_name(movie_name: str = Query(...)) -> dict:
    df = content_recommender.features.copy()
    df = df[df["title"].notnull()]
    match = df[df["title"].str.lower().eq(movie_name.strip().lower())].head(1)
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No movie found with name: '{movie_name}'")
    return match.iloc[0].to_dict()

@app.get("/top_rated")
def top_rated_movies(limit: int = 10) -> List[dict]:
    df = content_recommender.features.copy()
    top_movies = df.sort_values(by="rating", ascending=False).head(limit)
    return top_movies[["tconst", "title", "year", "rating", "genres"]].to_dict(orient="records")

@app.get("/random")
def random_movies(limit: int = 10) -> List[dict]:
    df = content_recommender.features.copy()
    sampled = df.sample(n=min(limit, len(df)), random_state=random.randint(0, 10000))
    return sampled[["tconst", "title", "year", "rating", "genres"]].to_dict(orient="records")

@app.get("/genre/{genre}")
def movies_by_genre(genre: str = Path(...), limit: int = 10) -> List[dict]:
    df = content_recommender.features.copy()
    df = df[df["genres"].str.contains(genre, case=False, na=False)]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No movies found for genre: '{genre}'")
    return df.head(limit)[["tconst", "title", "year", "rating", "genres"]].to_dict(orient="records")

@app.get("/collaborative")
def recommend_collab(user_id: str, top_k: int = 10):
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Collaborative model not available.")
    return collab_recommender.get_user_recommendations(user_id, top_k).to_dict(orient="records")

@app.post("/model/promote")
def promote_model(version: int = Query(...), stage: str = Query(...)):
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=str(version),
            stage=stage.capitalize(),  # Must be 'Staging', 'Production', etc.
            archive_existing_versions=True
        )
        return {"message": f"Model version {version} promoted to stage '{stage}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/versions")
def list_model_versions():
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id
            }
            for v in versions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
