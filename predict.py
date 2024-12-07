from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

from app import preprocess_data

def load_model(spark, model_path, model_type):
    if model_type == "LogisticRegression":
        return LogisticRegressionModel.load(model_path)
    elif model_type == "RandomForest":
        return RandomForestClassificationModel.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    spark = SparkSession.builder \
        .appName("WineQualityTraining") \
        .getOrCreate() \
    
    try:
        # Load saved model
        MODEL_PATH = "best_model"  # Adjust the path if needed
        BEST_MODEL_TYPE = "RandomForest"
        best_model = load_model(spark, MODEL_PATH, BEST_MODEL_TYPE)

        # Load test dataset
        TESTCSV = "TestDataset.csv"
        testdf = spark.read.csv(TESTCSV, header=True, inferSchema=True, sep=';')
        testdf = preprocess_data(testdf)

        # Make predictions
        preds = model.transform(testdf)

        # Save predictions to output
        preds.select("features", "prediction").write.csv("s3://my-bucket/wine-quality-predictions")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
