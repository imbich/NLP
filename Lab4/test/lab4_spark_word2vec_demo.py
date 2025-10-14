from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():
    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("Word2VecDemo") \
        .getOrCreate()
    
    # TIỀN XỬ LÝ
    data_path = r'C:\Users\NGUYEN PHUONG BICH\HOC_TAP\NLP\Data\c4-train.00000-of-01024-30K.json\c4-train.00000-of-01024-30K.json'
    df = spark.read.json(data_path)
    
    # 1. Chọn cột text và chuyển thành lowercase
    df = df.select(lower(col("text")).alias("text"))
    # 2. Xóa dấu câu và ký tự đặc biệt
    df = df.withColumn("text", regexp_replace("text", "[^a-zA-Z0-9\\s]", ""))
    # 3. Tách text thành mảng các từ
    df = df.withColumn("words", split(col("text"), "\\s+"))
    
    # Cấu hình & huấn luyện mô hình Word2Vec
    word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="words", outputCol="model")
    model = word2vec.fit(df)

    # Truy vấn từ tương tự
    synonyms = model.findSynonyms("computer", 5)
    synonyms.show()

    # 6️ Dừng Spark
    spark.stop()

if __name__ == "__main__":
    main()