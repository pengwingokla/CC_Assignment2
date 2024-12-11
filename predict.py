from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from app import preprocess_data
import os
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WineQualityPrediction")

def initialize_spark():
    return SparkSession.builder \
        .appName("WineQualityPrediction") \
        .master("spark://172.31.87.32:7077") \
        .config("spark.executor.memory", "3g") \
        .config("spark.executor.cores", "2") \
        .config("spark.task.cpus", "1") \
        .getOrCreate()

def load_model(model_path):
    model_type = get_model_type(model_path)

    if model_type == "LogisticRegressionModel":
        return LogisticRegressionModel.load(model_path)
    elif model_type == "RandomForestClassificationModel":
        return RandomForestClassificationModel.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_model_type(model_path):
    # Path to metadata file
    metadata_path = os.path.join(model_path, "metadata", "part-00000")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    model_type = metadata["class"].split(".")[-1]  # Extract the class name
    return model_type

def evaluate_model(predictions):
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", 
                                                           metricName="accuracy")
    f1s_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", 
                                                     metricName="f1")
    accuracy = acc_eval.evaluate(predictions)
    f1_score = f1s_eval.evaluate(predictions)

    return accuracy, f1_score

def main():
    spark = initialize_spark()
    
    try:
        # Load test dataset
        logger.info("Loading test dataset...")
        TESTCSV = "TestDataset.csv"
        testdf = spark.read.csv(TESTCSV, header=True, inferSchema=True, sep=';')
        testdf = preprocess_data(testdf)

        # Load saved model
        logger.info("Loading the trained model...")
        MODEL_PATH = "best_model"           # /home/ubuntu/code/best_model
        best_model = load_model(MODEL_PATH)

        # Make predictions
        logger.info("Making predictions...")
        preds = best_model.transform(testdf)
        preds.select("features", "prediction").show()

        # Evaluate model
        logger.info("Evaluating model performance...")
        accuracy, f1_score = evaluate_model(preds)
        logger.info(f"Model Performance: \
                    \n Accuracy = {accuracy:.4f} \
                    \n F1 Score = {f1_score:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()
