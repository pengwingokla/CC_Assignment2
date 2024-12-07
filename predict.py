from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from app import preprocess_data
import os
import json

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
    spark = SparkSession.builder \
        .appName("WineQualityTraining") \
        .getOrCreate() \
    
    try:
        # Load test dataset
        TESTCSV = "TestDataset.csv"
        testdf = spark.read.csv(TESTCSV, header=True, inferSchema=True, sep=';')
        testdf = preprocess_data(testdf)

        # Load saved model
        MODEL_PATH = "best_model"                       # Adjust the path to /home/ubuntu/code/best_model
        best_model = load_model(MODEL_PATH)

        # Make predictions
        preds = best_model.transform(testdf)
        preds.select("features", "prediction").show()

        # Evaluate model
        accuracy, f1_score = evaluate_model(preds)
        print(f"Model Performance: \
              \n Accuracy = {accuracy:.4f} \
              \n F1 Score = {f1_score:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
