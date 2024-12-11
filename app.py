from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col, count
from pyspark.sql.types import NumericType
import os
import logging

# TRAINCSV = "/home/ubuntu/code/TrainingDataset.csv"
# VALCSV   = "/home/ubuntu/code/ValidationDataset.csv"

TRAINCSV = "TrainingDataset.csv"
VALCSV   = "ValidationDataset.csv"
MODEL_PATH = "best_model"

# Configure logging
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

      # .master("local[*]") uncomment to make Spark run locally
        

def load_dataset(spark, train_csv, val_csv):
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("The dataset files do not exist.")
    train_df = spark.read.csv(train_csv, header=True, inferSchema=True, sep=';')
    val_df = spark.read.csv(val_csv, header=True, inferSchema=True, sep=';')

    return train_df.union(val_df)


def preprocess_data(df):
    # Remove extra quotes from column names
    df = df.toDF(*[c.replace('"', '') for c in df.columns])
    
    # Features: Exclude the 'quality' column
    feature_cols = df.columns[:-1]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)
    
    # Target: Ensure 'quality' column is numeric
    if isinstance(df.schema['quality'].dataType, NumericType):
        # If 'quality' is numeric, rename it to 'label'
        df = df.withColumnRenamed('quality', 'label')
    else:
        # If 'quality' is not numeric, index it
        indexer = StringIndexer(inputCol="quality", outputCol="label")
        df = indexer.fit(df).transform(df)

    return df


def oversample_minority_classes(df):
    # Check class distribution
    tot_count = df.count()

    print("Original class distribution:")
    cls_dist = df.groupBy("label") \
        .agg(count("label") \
        .alias("cls_count")) \
        .withColumn("percentage", (col("cls_count") / tot_count) * 100) \
        .orderBy("label")
    cls_dist.show()

    # Maximum class count
    max_count = cls_dist.agg({"cls_count": "max"}).collect()[0][0]

    # Oversample minority classes
    oversampled_df = df
    for row in cls_dist.collect():
        label = row["label"]
        cls_count = row["cls_count"]
        if cls_count < max_count:
            fraction = (max_count - cls_count) / cls_count
            sampled_df = df.filter(col("label") == label).sample(withReplacement=True, fraction=fraction, seed=42)
            oversampled_df = oversampled_df.union(sampled_df)

    # Verify new class distribution
    print("New class distribution:")
    new_dist = oversampled_df.groupBy("label") \
        .agg(count("label").alias("cls_count")) \
        .withColumn("percentage", (col("cls_count") / tot_count) * 100) \
        .orderBy("label")
    new_dist.show()

    return oversampled_df

def perform_grid_search_rf(train_df, val_df):
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 50, 75, 100, 150, 200]) \
        .addGrid(rf.maxDepth, [5, 10, 15, 20, 25]) \
        .addGrid(rf.maxBins, [16, 32, 48, 64]) \
        .build()
    eval_ac = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    tvs = TrainValidationSplit(estimator=rf,
                               estimatorParamMaps=paramGrid,
                               evaluator=eval_ac,
                               trainRatio=0.8)
    
    model = tvs.fit(train_df)
    best_rf = model.bestModel
    
    val_preds = best_rf.transform(val_df)
    best_accuracy = eval_ac.evaluate(val_preds)
    best_f1_score = eval_f1.evaluate(val_preds)

    print()
    print("\nBest Random Forest Model:")
    print(f"numTrees = {best_rf.getNumTrees}")
    print(f"maxDepth = {best_rf.getMaxDepth}")
    print(f"maxBins  = {best_rf.getMaxBins}")
    print()
    print(f"BEST ACCURACY ACHIEVED: {best_accuracy:.4f}")
    print(f"BEST F1 SCORE ACHIEVED: {best_f1_score:.4f}")
    print()

    return best_rf

def train_and_evaluate(models, train_df, val_df):
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    results = []

    for name, model in models:
        print(f"\nTraining {name}...")
        spark_model = model.fit(train_df)
        preds = spark_model.transform(val_df)
        accur = eval_acc.evaluate(preds)
        f1_score = eval_f1.evaluate(preds)
        print(f"Accuracy: {accur:.4f}, \nF1 Score: {f1_score:.4f}")
        results.append((name, accur, f1_score))
        # trained_models[name] = spark_model

    return results

def main():
    spark = initialize_spark()
    try:
        logger.info("Starting Spark application...")
        df = load_dataset(spark, TRAINCSV, VALCSV)
        df = preprocess_data(df)
        df = oversample_minority_classes(df)
        train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

        logger.info("Training Random Forest model with grid search...")
        best_rf_model = perform_grid_search_rf(train_df, val_df)

        logger.info(f"Saving the best model to {MODEL_PATH}...")
        best_rf_model.write().overwrite().save(MODEL_PATH)

    finally:
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()