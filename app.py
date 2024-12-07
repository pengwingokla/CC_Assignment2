from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col, count, lit
import matplotlib.pyplot as plt
from pyspark.sql.types import NumericType
import os

TRAINCSV = "TrainingDataset.csv"
VALCSV   = "ValidationDataset.csv"

def initialize_spark():
    return SparkSession.builder \
        .appName("WineQualityPrediction") \
        .master("local[*]") \
        .getOrCreate()


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

def evaluate_model(model, val_df):
    val_pred = model.transform(val_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    return evaluator.evaluate(val_pred), val_pred

def plot_loss(best_model):
    if hasattr(best_model, 'summary') and hasattr(best_model.summary, 'objectiveHistory'):
        loss_values = best_model.summary.objectiveHistory
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loss_values)), loss_values, marker='o')
        plt.title('Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_over_iters.png', dpi=300, bbox_inches='tight')

def train_and_evaluate(models, train_df, val_df):
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    results = []

    for name, model in models:
        print(f"\nTraining {name}...")
        trained_model = model.fit(train_df)
        predictions = trained_model.transform(val_df)
        accuracy = eval_acc.evaluate(predictions)
        f1_score = eval_f1.evaluate(predictions)
        print(f"Accuracy: {accuracy:.4f}, \nF1 Score: {f1_score:.4f}")
        results.append((name, accuracy, f1_score))

    return results

def main():
    spark = initialize_spark()
    try:
        combined_df = load_dataset(spark, TRAINCSV, VALCSV)
        processed_df= preprocess_data(combined_df)
        balanced_df = oversample_minority_classes(processed_df)
        train, val  = balanced_df.randomSplit([0.8, 0.2], seed=42)
        # model = train_model(train)
        # val_acc, val_pred = evaluate_model(model, val)
        # print(f"Validation Accuracy: {val_acc:.2f}")
        # # plot_loss(model.bestModel)
        # model.bestModel.write().overwrite().save("optimized_model")

        models = [
            ("Logistic Regression", LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)),
            ("Random Forest", RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)),
        ]

        # Train and evaluate models
        results = train_and_evaluate(models, train, val)

        # Print results
        print("\nModel Comparison Results")
        for name, accuracy, f1_score in results:
            print(f"-- {name}:\
                  \n Accuracy = {accuracy:.4f}\
                  \n F1 Score = {f1_score:.4f}")

        # Find the best model
        best_model = max(results, key=lambda x: x[1])
        print(f"\nBest Model: {best_model[0]} with Accuracy = {best_model[1]:.4f}")
        print()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()