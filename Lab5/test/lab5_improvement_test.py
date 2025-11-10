# ...existing code...
import os
import csv
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def clean_text(df, text_col="text", out_col="clean"):
    df = df.withColumn(out_col, lower(col(text_col)))
    df = df.withColumn(out_col, regexp_replace(col(out_col), r"https?://\S+\s?", " "))
    df = df.withColumn(out_col, regexp_replace(col(out_col), r"<[^>]+>", " "))
    df = df.withColumn(out_col, regexp_replace(col(out_col), r"[^a-z0-9\s]", " "))
    df = df.withColumn(out_col, regexp_replace(col(out_col), r"\s+", " "))
    return df

def build_pipeline(vectorizer="hashing", numFeatures=16384, word2vec_size=100, classifier="lr"):
    tokenizer = Tokenizer(inputCol="clean", outputCol="words")
    stopwords = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    if vectorizer == "hashing":
        if classifier == "nb":
            hashing = HashingTF(inputCol="filtered_words", outputCol="features", numFeatures=numFeatures)
            feat_stages = [hashing]
        else:
            hashing = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=numFeatures)
            idf = IDF(inputCol="rawFeatures", outputCol="features")
            feat_stages = [hashing, idf]
    elif vectorizer == "word2vec":
        if classifier == "nb":
            raise ValueError("NaiveBayes (multinomial) is not compatible with Word2Vec features (signed values).")
        
        w2v = Word2Vec(vectorSize=word2vec_size, minCount=1, inputCol="filtered_words", outputCol="features")
        feat_stages = [w2v]
    else:
        raise ValueError("vectorizer must be 'hashing' or 'word2vec'")

    if classifier == "lr":
        clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    elif classifier == "nb":
        clf = NaiveBayes(featuresCol="features", labelCol="label")
    elif classifier == "gbt":
        clf = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)
    else:
        raise ValueError("classifier must be 'lr', 'nb' or 'gbt'")

    stages = [tokenizer, stopwords] + feat_stages + [clf]
    return Pipeline(stages=stages)

def run_experiments(spark, df):
    df = df.filter(col("sentiment").isNotNull())
    df = df.withColumn("label", ((col("sentiment").cast("integer") + 1) / 2).cast("double"))
    df = clean_text(df, text_col="text", out_col="clean")
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    vectorizers = ["hashing", "word2vec"]
    classifiers = ["lr", "nb", "gbt"]

    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")

    results = []
    for vect in vectorizers:
        for clf in classifiers:
            start = time.time()
            row = {"vectorizer": vect, "classifier": clf, "accuracy": None, "f1": None, "auc": None, "time_sec": None, "error": None}
            try:
                pipeline = build_pipeline(vectorizer=vect, numFeatures=16384, word2vec_size=100, classifier=clf)
                model = pipeline.fit(train)
                preds = model.transform(test)
                acc = eval_acc.evaluate(preds)
                f1 = eval_f1.evaluate(preds)
                auc = None
                try:
                    auc = eval_auc.evaluate(preds)
                except Exception:
                    auc = None
                row.update({"accuracy": float(acc), "f1": float(f1), "auc": (float(auc) if auc is not None else None)})
            except Exception as e:
                row["error"] = str(e)
            row["time_sec"] = round(time.time() - start, 2)
            results.append(row)
            print(f"Finished: vect={vect} clf={clf} acc={row['accuracy']} f1={row['f1']} auc={row['auc']} time={row['time_sec']} err={row['error']}")

    return results

def main():
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    df = spark.read.csv("Data/sentiments.csv", header=True, inferSchema=True)
    print(run_experiments(spark, df))
    spark.stop()

if __name__ == "__main__":
    main()