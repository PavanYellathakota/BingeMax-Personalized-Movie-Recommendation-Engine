from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import mlflow
import mlflow.pyfunc
import tempfile
import pickle

# === Configuration ===
INTERACTIONS_PATH = "/opt/airflow/data/external/user_interactions.csv"
FEATURES_PATH = "/opt/airflow/data/processed/features.parquet"

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT = "Movie-Recommender-ALS"
REGISTER_MODEL_NAME = "ALSRecommender"  # Optional model registry name
BATCH_SIZE = 100

# === Custom PyFunc Wrapper ===
class ALSRecommenderWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["als_model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        # Expects: model_input = (user_id, user_interaction_vector)
        user_id, user_vector = model_input
        return self.model.recommend(user_id, user_vector, N=5)

# === Main Training Function ===
def retrain_and_log():
    print("🔄 Loading data...")
    interactions = pd.read_csv(INTERACTIONS_PATH)
    movies = pd.read_parquet(FEATURES_PATH)

    print("📊 Building sparse matrix...")
    user_ids = interactions["user_id"].unique()
    movie_ids = interactions["movie_id"].unique()
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}

    rows = interactions["user_id"].map(user_to_idx)
    cols = interactions["movie_id"].map(movie_to_idx)
    data = interactions["interaction"]
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    print("🧠 Training ALS model...")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15, use_gpu=False)
    model.fit(matrix.T)

    print("📈 Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        # Parameters
        mlflow.log_param("factors", 50)
        mlflow.log_param("regularization", 0.01)
        mlflow.log_param("iterations", 15)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # Sample Metric
        try:
            user_id = 0
            recommendations = model.recommend(user_id, matrix[user_id], N=5)
            if isinstance(recommendations, tuple) and len(recommendations) == 2:
                scores = recommendations[1]
                if len(scores) > 0:
                    mlflow.log_metric("sample_score", float(scores[0]))
                    print(f"✅ Logged sample_score: {float(scores[0])}")
                else:
                    print("⚠️ No recommendation scores returned.")
            else:
                print(f"⚠️ Unexpected recommendation format: {recommendations}")
        except Exception as e:
            print(f"❌ Error during recommendation: {e}")

        # Save ALS model locally
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "als_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.pyfunc.log_model(
                artifact_path="als_model",
                python_model=ALSRecommenderWrapper(),
                artifacts={"als_model": model_path},
            )

        # Optional: Register the model
        try:
            mlflow.register_model(f"runs:/{run.info.run_id}/als_model", REGISTER_MODEL_NAME)
            print(f"📌 Model registered as '{REGISTER_MODEL_NAME}'")
        except Exception as e:
            print(f"⚠️ Model registration failed: {e}")

# === DAG Definition ===
with DAG(
    dag_id="retrain_als_model",
    description="Retrain ALS model and log to MLflow",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["recommender", "mlflow"],
) as dag:
    retrain_task = PythonOperator(
        task_id="retrain_and_log_model",
        python_callable=retrain_and_log
    )
