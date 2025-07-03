# File: src/models/als/train_als.py
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession

def train_and_log_model():
    spark = SparkSession.builder.appName("ALSModelTraining").getOrCreate()

    ratings_df = spark.read.csv("data/processed/ratings.csv", header=True, inferSchema=True)
    ArrayType = spark.sparkContext.parallelize
    (train_df, test_df) = ratings_df.randomSplit([0.8, 0.2])

    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=10,
        maxIter=10,
        regParam=0.1
    )
    model = als.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    mlflow.set_experiment("als_recommender")
    with mlflow.start_run() as run:
        mlflow.spark.log_model(model, artifact_path="model", registered_model_name="als-model")
        mlflow.log_metric("rmse", rmse)
        run_id = run.info.run_id

    return rmse, run_id
