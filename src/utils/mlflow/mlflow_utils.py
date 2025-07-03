# File: src/utils/mlflow/mlflow_utils.py
import mlflow
from pyspark.ml.recommendation import ALSModel

def get_latest_model_uri(stage="Production"):
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(name="als-model", stages=[stage])
    if not versions:
        raise ValueError(f"No model found in stage: {stage}")
    return versions[0].source  # e.g., 'mlruns/0/.../artifacts/model'

def load_latest_als_model(stage="Production"):
    uri = get_latest_model_uri(stage)
    return ALSModel.load(uri)

def promote_model_version(version: str, stage: str = "Production"):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name="als-model", version=version, stage=stage)
