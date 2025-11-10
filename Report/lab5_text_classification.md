# Báo cáo Lab 5: Text Classification

## 1. Mục tiêu chung
- Xây dựng một quy trình phân loại văn bản hoàn chỉnh, từ văn bản thô đến mô hình học mát đã được huấn luyện, sử dụng các kỹ thuật Tokenization và Vectorization đã học từ những bài thực hành trước.

- So sánh các phương pháp phân loại trên dữ liệu văn bản.

- Đánh giá, cải thiện và tự động hóa thử nghiệm để so sánh hiệu năng.

## 2. Triển khai

**Pipeline**: `Văn bản thô` $\rightarrow$ `Tokenization` $\rightarrow$ `Vectorization` $\rightarrow$ `Mô hình học máy` $\rightarrow$ `Dự đoán`

**Model**: Sử dụng mô hình Logistic Regression - một mô hình tuyến tính đơn giản nhưng hiệu quả cho phân loại nhị phân.

**Đánh giá**: Để đánh giá hiệu suất của mô hình, sử dụng các chỉ số như `Accuracy, Precision, Recall, F1-score.`

### Task 1: Data Preparation (with scikit-learn)

- **Dữ liệu**: Sử dụng một tập dữ liệu nhỏ trong bộ nhớ để đơn giản hóa

<pre>
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
    ]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative
</pre>

- **Vectorize**: Sử dụng `TfidfVectorizer` hoặc `CountVectorizer` để chuyển đổi những `texts` thành đặc trưng số

### Task 2: TextClassifier Implementation
- Mục tiêu: Xây dựng một lớp mô hình phân loại văn bản (TextClassifier) sử dụng Logistic Regression kết hợp với Vectorizer để chuyển đổi dữ liệu văn bản thành dạng đặc trưng số học, phục vụ cho quá trình huấn luyện và dự đoán.

- Được triển khai trong `src/models/text_classifier.py`

#### Thiết kế lớp `TextClassifier`
- Phương thức khởi tạo `__init__(self, vectorizer: Vectorizer)`
    - Nhận một đối tượng `Vectorizer` để xử lý dữ liệu văn bản.
    - Khởi tạo thuộc tính `_model` để lưu mô hình Logistic Regression sau khi huấn luyện.

- Phương thức huấn luyện `fit(self, texts: List[str], labels: List[int]):`
    - Chuyển đổi danh sách văn bản `texts` thành ma trận đặc trưng `X`.
    - Khởi tạo mô hình Logistic Regression với `solver="liblinear"` (phù hợp cho tập dữ liệu nhỏ).
    - Huấn luyện mô hình trên dữ liệu huấn luyện `(X, labels)`.

- Phương thức dự đoán `predict(self, texts: List[str]) -> List[int]:`
    - Dữ liệu đầu vào được vector hóa bằng `transform`.
    - Sử dụng mô hình đã huấn luyện để dự đoán nhãn cho các văn bản mới.
    - Trả về danh sách các nhãn dự đoán.

- Phương thức đánh giá `evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str,float]:` 
    - Sử dụng các hàm đánh giá từ `sklearn.metrics`
    - Trả về bộ từ điển gồm các chỉ số:
        - `Accuracy`: độ chính xác tổng thể.
        - `Precision`: độ chính xác trên từng lớp.
        - `Recall`: độ nhạy của mô hình.
        - `F1-score`: trung bình điều hòa giữa `Precision` và `Recall`.

- Kết quả mong đợi: Sau khi huấn luyện, mô hình có thể:
    - Phân loại văn bản đầu vào thành các nhãn cụ thể.
    - Đưa ra báo cáo đánh giá độ chính xác và chất lượng phân loại.


### Task 3: Evaluation

- Mục tiêu: Thực hiện kiểm thử mô hình phân loại văn bản đã xây dựng bằng cách:
    - Chuẩn bị dữ liệu văn bản và nhãn.
    - Chia dữ liệu thành tập huấn luyện và kiểm thử.
    - Tiến hành huấn luyện mô hình.
    - Dự đoán và đánh giá chất lượng mô hình bằng các chỉ số thống kê.

- Được triển khai trong: `test/lab5_test.py`

- Cách chạy (Đứng tại thu mục `NLP`): `python -m Lab5.test.lab5_test`

- Sử dụng tập dữ liệu nhỏ ở trên, dùng `train_test_split` từ `sklearn.model_selection` với tỷ lệ train-test tương ứng 80-20

- Các thành phần:
    - `RegexTokenizer`: token hóa văn bản.
    - `CountVectorizer, TfidfVectorizer`: để vector háo văn bản sau khi token hóa.

    <pre>
    regexTokenizer = RegexTokenizer()
    countVectorizer = CountVectorizer(tokenizer=regexTokenizer)
    tfidfVectorizer = TfidfVectorizer()
    </pre>

- Huấn luyện và đánh giá mô hình:
    - Với `CountVectorizer`:
        <pre>
        textClassifier_count = TextClassifier(vectorizer=countVectorizer)
        textClassifier_count.fit(X_train, y_train)
        y_pred_count = textClassifier_count.predict(X_test)
        print(f"Count Vectorizer Evaluation\n", textClassifier_count.evaluate(y_test, y_pred_count))
        </pre>
        
        - Kết quả:
        <pre>
        Count Vectorizer Evaluation
        {'accuracy': 0.5, 'precision': 0.5, 'recall': 1.0, 'f1_score': 0.6666666666666666}
        </pre>

        - Nhận xét: 
            - Mô hình với `CountVectorizer` chỉ dự đoán đúng một nửa số mẫu kiểm tra. Điều này cho thấy khả năng tổng thể của mô hình chưa tốt.
            - Nguyên nhân có thể do tập dữ liệu nhỏ, phân bố từ vựng không đồng đều hoặc vector tần suất đơn thuần chưa thể hiện được ngữ nghĩa của câu.
            - Điều này cho thấy phương pháp CountVectorizer chưa đủ mạnh để xử lý bài toán phân loại văn bản có ít dữ liệu huấn luyện.

    - Với `TfidfVectorizer`:
        <pre>
        textClassifier_tfidf = TextClassifier(vectorizer=tfidfVectorizer)
        textClassifier_tfidf.fit(X_train, y_train)
        y_pred_tfidf = textClassifier_tfidf.predict(X_test)
        print(f"TF-IDF Vectorizer Evaluation\n", textClassifier_tfidf.evaluate(y_test, y_pred_tfidf))
        </pre>

        - Kết quả:
        <pre>
        TF-IDF Vectorizer Evaluation
        {'accuracy': 0.5, 'precision': 0.5, 'recall': 1.0, 'f1_score': 0.6666666666666666}
        </pre>

        - Nhận xét:
            - Việc chuyển từ `CountVectorizer` sang `TF-IDF Vectorizer` không làm thay đổi hiệu suất mô hình — các chỉ số đều giữ nguyên.
            - Dữ liệu huấn luyện quá nhỏ hoặc mất cân bằng nên hai phương pháp biểu diễn chưa thể hiện được sự khác biệt.
            - Mô hình có thể đang overfit hoặc không khai thác đủ thông tin ngữ nghĩa.

- File kiểm thử giúp:
    - Kiểm chứng hoạt động của mô hình `TextClassifier`.
    - Đánh giá hiệu quả của mô hình thông qua các chỉ số chuẩn.
    - Cung cấp quy trình đầy đủ từ tiền xử lý $\rightarrow$ huấn luyện $\rightarrow$ dự đoán $\rightarrow$ đánh giá.

### Advanced Example: Sentiment Analysis with PySpark
- Mục tiêu: 
    - Triển khai mô hình phân loại cảm xúc (sentiment classification) trên tập dữ liệu văn bản lớn bằng cách sử dụng Apache Spark.
    - Khác với các ví dụ trước chạy trên bộ nhớ máy đơn lẻ, PySpark cho phép xử lý phân tán dữ liệu ở quy mô lớn, giúp hệ thống vẫn hoạt động hiệu quả khi dữ liệu vượt quá giới hạn bộ nhớ RAM của một máy tính.

- Triển khai:
    - Toàn bộ quy trình được triển khai trong: `test/lab5_spark_sentiment_analysis.py`
    - Cách chạy (Đứng tại thư mục `NLP`): `python -m Lab5.test.lab5_spark_sentiment_analysis`

    - Bước 1: Khởi tạo SparkSession
        <pre>
        spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
        </pre>
        - Đây là điểm khởi đầu của mọi ứng dụng Spark.
        - Cho phép tạo, quản lý và truy cập các DataFrame phân tán trên cluster.
        - `appName` giúp định danh tiến trình trong Spark UI để theo dõi hiệu năng.

    - Bước 2: Tải và tiền xử lý dữ liệu
        <pre>
        df = spark.read.csv("Data/sentiments.csv", header=True, inferSchema=True)
        df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
        initial_row_count = df.count()
        df = df.dropna(subset=["sentiment"])
        </pre>

        - Đọc dữ liệu file `Data/sentiments.csv` chứa hai cột:
            - `text`: câu văn bản.
            - `sentiment`: nhãn cảm xúc gốc (`-1` cho tiêu cực, `1` cho tích cực).
        - Chuẩn hóa nhãn: Chuyển các giá trị `-1` $\rightarrow$ `0` và `1` $\rightarrow$ `1` để phù hợp với đầu vào của mô hình phân loại nhị phân.
        - Xử lý giá trị thiếu: Loại bỏ các hàng có nhãn cảm xúc bị thiếu để đảm bảo tính toàn vẹn của dữ liệu.

    - Bước 3: Xây dựng Pipeline tiền xử lý
        - Spark ML sử dụng mô hình Pipeline bao gồm chuỗi các Transformer (biến đổi dữ liệu) và Estimator (thuật toán huấn luyện).

        <pre>
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        </pre>

        - `tokenizer`: tách văn bản thành các từ riêng lẻ.
        - `stopwordRemover`: loại bỏ các từ dừng phổ biến giúp giảm nhiễu.
        - `hashingTF`: chuyển danh sách từ thành vector đặc trưng cố định kích thước (`numFeatures=10000`)
        - `idf`: giảm trọng số của các từ xuất hiện thường xuyên, làm nổi bật các từ đặc trưng hơn trong toàn bộ tập văn bản.
    
    - Bước 4: Huấn luyện mô hình Logistic Regression
        <pre>
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10,   regParam=0.0001)
        pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

        train, test = df.randomSplit([0.8, 0.2], seed=42)
        model = pipeline.fit(train)
        </pre>

        - Mô hình Logistic Regression là thuật toán phân loại tuyến tính đơn giản nhưng hiệu quả trong bài toán phân loại cảm xúc.
        - `Pipeline` gộp tất cả các bước xử lý thành một chuỗi thống nhất.
        - Khi gọi `fit()`, Spark sẽ:
            1. Tokenize dữ liệu.
            2. Loại bỏ stop words.
            3. Tạo vector đặc trưng.
            4. Tính IDF.
            5. Huấn luyện Logistic Regression trên đầu ra cuối cùng.
        $\Rightarrow$ Giúp đảm bảo tính tái sử dụng và nhất quán khi áp dụng lên dữ liệu mới (train và test).

    - Bước 5: Đánh giá mô hình
        <pre>
        predictions = model.transform(test)
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        accuracy = evaluator_acc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)

        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        </pre>

        - `transform()`: áp dụng toàn bộ pipeline lên tập kiểm thử, tự động thực hiện mọi bước tiền xử lý.
        - `MulticlassClassificationEvaluator` được sử dụng để tính toán các chỉ số đánh giá như `Accuracy, F1-score`.

    - Kết quả: Với train-test tỷ lệ tương ứng là 80-20
        <pre>
        Accuracy: 0.7294860234445446
        F1 Score: 0.7266221530017497
        </pre>
    
    - Nhận xét:
        - Accuracy và F1 tương đương cho thấy mô hình không quá thiên lệch về một lớp lớn; nhưng giá trị ~0.73 chỉ là baseline, vẫn còn nhiều không gian cải thiện.
        - Feature đơn giản (Count/TF-IDF) có thể thiếu thông tin ngữ cảnh.
        - Spark có khả năng mở rộng: pipeline dễ chuyển sang cluster để xử lý tập lớn.
        - Có thể thử các biểu diễn khác: CountVectorizer vs TF-IDF vs pre-trained embeddings (GloVe/FastText) hoặc Word2Vec (nếu có đủ dữ liệu).

### Task 4: Evaluating and Improving Model Performance
- Mục tiêu: đánh giá hiệu năng baseline và các cải tiến đã thử, phân tích kết quả, nêu vấn đề gặp phải và khuyến nghị hướng cải thiện.

- Triển khai:
    - Vectorizer: HashingTF (+IDF) và Word2Vec (mean pooling).
    - Classifier: LogisticRegression (LR), Multinomial NaiveBayes (NB), GBTClassifier (GBT).

- Kết quả:
    - Baseline (Spark pipeline LR, example run): Accuracy = 0.7294860234445446, F1 = 0.7266221530017497
    - Kết quả chính:
    <pre>
    [{'vectorizer': 'hashing', 'classifier': 'lr', 'accuracy': 0.7457168620378719, 'f1': 0.7467363563526117, 'auc': 0.7906086968586972, 'time_sec': 7.85, 'error': None}, 
    {'vectorizer': 'hashing', 'classifier': 'nb', 'accuracy': 0.7718665464382326, 'f1': 0.7577805131002275, 'auc': 0.8314775502275503, 'time_sec': 1.8, 'error': None}, 
    {'vectorizer': 'hashing', 'classifier': 'gbt', 'accuracy': 0.7664562669071235, 'f1': 0.7454600816183472, 'auc': 0.8362349456099448, 'time_sec': 55.36, 'error': None}, 
    {'vectorizer': 'word2vec', 'classifier': 'lr', 'accuracy': 0.6844003606853021, 'f1': 0.6498203235240835, 'auc': 0.7221459096459086, 'time_sec': 4.68, 'error': None}, 
    {'vectorizer': 'word2vec', 'classifier': 'nb', 'accuracy': None, 'f1': None, 'auc': None, 'time_sec': 0.03, 'error': 'NaiveBayes (multinomial) is not compatible with Word2Vec features (signed values).'}, 
    {'vectorizer': 'word2vec', 'classifier': 'gbt', 'accuracy': 0.6591523895401262, 'f1': 0.6383120976501246, 'auc': 0.6838682151182152, 'time_sec': 14.38, 'error': None}]
    </pre>

- Nhận xét:
    - `HashingTF (counts) + Multinomial NB cải thiện so với baseline LR`: NB phù hợp với đặc trưng đếm (multinomial), nhanh và hiệu quả trên dữ liệu nhỏ.
    - GBT tăng AUC nhưng tốn thời gian — phù hợp khi cần tối ưu ROC và có tài nguyên.
    - Kết luận: trên tập dữ liệu hiện tại, biểu diễn sparse counts + NB/LR là lựa chọn thực tế; embeddings tự huấn luyện chỉ vượt trội nếu có corpus lớn hoặc dùng pre-trained embeddings.

## 3. Vấn đề gặp phải

### Dữ liệu quá nhỏ (mô hình không học được đặc trưng)
- Tập dữ liệu huấn luyện không đủ đại diện; không đủ mẫu cho mỗi lớp; nhiều từ hiếm.
- Cách xử lý:
    - Tăng kích thước dữ liệu (thu thập thêm, data augmentation: paraphrase, back-translation). (Sử dụng tập dữ liệu `sentiments.csv`)
    - Dùng stratify=labels trong train_test_split để đảm bảo phân bố lớp giữa train/test.

### CountVectorizer trả về kết quả kém hoặc model dựa đoán 1 lớp
- Tokenizer tùy chỉnh không tương thích với CountVectorizer
- Sau token hóa, nhiều văn bản trở thành rỗng → ma trận đặc trưng chứa nhiều hàng rỗng.
- Dữ liệu nhỏ / từ vựng quá khác biệt → vector hóa theo tần suất không đủ phân biệt.
- Cách xử lý:
    - Dùng TfidfVectorizer thay vì Count nếu cần giảm ảnh hưởng từ quá phổ biến.

### ZeroDivisionWarning
- Mô hình không dự đoán nhãn dương (hoặc âm) → precision cho lớp đó vô nghĩa.
- Cách xử lý: Khi tính metric: dùng tham số `zero_division=0` để tránh exception và ghi rõ cách xử lý.

### Multinomial NaiveBayes không chấp nhận giá trị âm $\rightarrow$ gây exception khi dùng Word2Vec.
- Giải pháp: skip combo Word2Vec+NB; dùng HashingTF/CountVectorizer cho NB hoặc dùng LR/GBT cho Word2Vec.

## 4. Tham khảo
- Spark: https://spark.apache.org/
- Dataset: https://classroom.google.com/u/2/c/ODAyNDI0NjIwODE1/m/ODA0Njg2NzEwMjI4/details (Dữ liệu đã được cung cấp trong Dataset: Twitter Sentiments)
- HashingDF: https://stackoverflow.com/questions/35205865/what-is-the-difference-between-hashingtf-and-countvectorizer-in-spark
- ChatGPT: https://chatgpt.com/

