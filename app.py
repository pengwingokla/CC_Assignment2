from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col, count, lit
import matplotlib.pyplot as plt
from pyspark.sql.types import NumericType
import os

# TRAINCSV = "/home/ubuntu/code/TrainingDataset.csv"
# VALCSV   = "/home/ubuntu/code/ValidationDataset.csv"

TRAINCSV = "TrainingDataset.csv"
VALCSV   = "ValidationDataset.csv"

def initialize_spark():
    return SparkSession.builder \
        .appName("WineQualityPrediction") \
        .getOrCreate()

      # .master("local[*]") \ # make Spark run locally
        

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

def train_model(train_df):
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy"),
                               trainRatio=0.8)
    return tvs.fit(train_df)

def train_and_evaluate(models, train_df, val_df):
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    results = []
    trained_models = {}

    for name, model in models:
        print(f"\nTraining {name}...")
        spark_model = model.fit(train_df)
        preds = spark_model.transform(val_df)
        accur = eval_acc.evaluate(preds)
        f1_score = eval_f1.evaluate(preds)
        print(f"Accuracy: {accur:.4f}, \nF1 Score: {f1_score:.4f}")
        results.append((name, accur, f1_score))
        trained_models[name] = spark_model

    return results, trained_models

def main():
    spark = initialize_spark()
    try:
        df = load_dataset(spark, TRAINCSV, VALCSV)
        df = preprocess_data(df)
        df = oversample_minority_classes(df)
        train, val  = df.randomSplit([0.8, 0.2], seed=42)

        models = [
            ("Logistic Regression", LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)),
            ("Random Forest", RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)),
        ]

        # Train and evaluate models
        results, trained_models = train_and_evaluate(models, train, val)

        # Print results
        print("\nModel Comparison Results")
        for name, accuracy, f1_score in results:
            print(f"-- {name}:\
                  \n Accuracy = {accuracy:.4f}\
                  \n F1 Score = {f1_score:.4f}")

        # Find the best model
        best_modname, _, _ = max(results, key=lambda x: x[1])
        best_mod = trained_models[best_modname]
        print(f"\nBest Model: {best_modname}")
        print()
        
        # Save
        best_mod.write().overwrite().save("best_model")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()