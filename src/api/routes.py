# Add to your FastAPI app file in src/api/
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.mlflow.mlflow_utils import load_latest_als_model, get_latest_model_uri, promote_model_version

router = APIRouter()

class PredictRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10

@router.post("/model/predict")
def predict_from_model(req: PredictRequest):
    try:
        model = load_latest_als_model()
        df = spark.createDataFrame([(req.user_id, movie_id) for movie_id in range(0, 1000)], ["userId", "movieId"])
        predictions = model.transform(df).orderBy("prediction", ascending=False).limit(req.num_recommendations)
        return predictions.toPandas().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/latest")
def get_latest_model():
    try:
        uri = get_latest_model_uri()
        return {"model_uri": uri}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/model/promote")
def promote_model(version: str, stage: str = "Production"):
    try:
        promote_model_version(version, stage)
        return {"message": f"Model version {version} promoted to {stage}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
