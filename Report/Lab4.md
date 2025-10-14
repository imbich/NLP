# Báo cáo Lab 4: Word Embeddings

## 1. Mục tiêu

- Hiểu và thực hành sử dụng các mô hình word embedding phổ biến (Word2Vec, Glove hoặc fastText).

- Trực quan hóa không gian vector từ.

- Huấn luyện mô hình Word2Vec trên dữ liệu tiếng Anh thực tế (`UD English-EWT`)

- Đánh giá và sử dụng mô hình embedding tự huấn luyện

## 2. Nội dung thực hiện

### 2.1. Quy trình thực hiện

#### Giảm chiều và trực quan hóa vector

- **Giảm chiều:** Sử dụng các kỹ thuật như PCA hoặc t-SNE để giảm chiều các word vector (từ Word2Vec, GloVe, hoặc fastText) xuống còn 2 hoặc 3 chiều.

- **Trực quan hóa:** Vẽ biểu đồ (scatter plot) để hiển thị các từ trong không gian 2D/3D, qua đó quan sát mối quan hệ ngữ nghĩa giữa chúng.

**Các bước thực hiện:**

1. **Lấy tập từ và vector từ mô hình pre-trained `glove.2024.wikigiga.50d`

2. **Giảm chiều**
   - Sử dụng hàm `reduce_dimensions` trong file `src/representations/reduce_dim.py` để giảm chiều các vector từ về 2D hoặc 3D.
   - Có thể lựa chọn giữa hai phương pháp:
     - **PCA (Principal Component Analysis):** Giảm chiều tuyến tính, giữ lại phương sai lớn nhất.
     - **t-SNE (t-distributed Stochastic Neighbor Embedding):** Giảm chiều phi tuyến, giữ lại cấu trúc lân cận, phù hợp trực quan hóa.

3. **Trực quan hóa:**  
   - Sử dụng các hàm `plot_2d`, `plot_3d` hoặc `plot_embeddings` để vẽ scatter plot các từ trong không gian 2D/3D.
   - Gắn nhãn các điểm để dễ quan sát mối quan hệ giữa các từ.

4. **Chạy demo**
Tại thư mục `Lab4/src/representations` chạy
```
python reduce_dim.py
```
#### Task 1: Setup

1. **Cài đặt `gensim`**

Thêm `gensim` vào file `requirements.txt` và cài đặt băng lệnh 
```
pip install -r requirements.txt
```
Hoặc cài đặt trực tiếp bằng lệnh:
```
pip install gensim
```

2. **Tải mô hình pre-trained**
Sử dụng mô hình pre-trained từ kho dữ liệu của gensim.

```python
import gensim.downloader as api
model = api.load("glove-wiki-gigaword-50")
```
Mô hình sẽ tự động được tải về và lưu trong cache của gensim.

#### Task 2: Word Embedding Exploration

1. **Tạo file word_embedder.py**  
   Đã tạo file `src/representations/word_embedder.py` để xây dựng lớp `WordEmbedder` phục vụ cho việc khám phá embedding.

2. **Cài đặt lớp WordEmbedder**  
   - Hàm khởi tạo `__init__(self, model_name: str)` nhận tên mô hình (ví dụ: `'glove-wiki-gigaword-50'`), sử dụng `gensim.downloader.load` để tải mô hình và lưu vào thuộc tính `self.model`.
   - Phương thức `get_vector(self, word: str)` trả về vector embedding của `word`. Nếu `word` không có trong từ điển (OOV), trả về `None`.
   - Phương thức `get_similarity(self, word1: str, word2: str)` trả về độ tương đồng cosine giữa `word1` và `word2`. Nếu một trong hai từ không có trong từ điển, trả về `None`.
   - Phương thức `get_most_similar(self, word: str, top_n: int = 10)` trả về danh sách `top_n` từ giống nhất với `word`. Nếu `word` không có trong từ điển, trả về `None`.

3. **Hướng dẫn sử dụng và ví dụ kết quả**  
   Ví dụ sử dụng lớp WordEmbedder:
   ```python
   from src.representations.word_embedder import WordEmbedder

   wordEmbedder = WordEmbedder('glove-wiki-gigaword-50')
   print("Vector for the word 'king':", wordEmbedder.get_vector('king'))  # Vector 50 chiều
   print("Similarity between 'king' and 'queen':", wordEmbedder.get_similarity('king', 'queen'))  # Độ tương đồng cosine
   print("10 most similar words to 'computer':", wordEmbedder.get_most_similar('computer'))  # Top 10 từ giống nhất
   ```

   **Kết quả mẫu:**
   - `get_vector('king')`: Trả về một mảng numpy 50 chiều.
   - `get_similarity('king', 'queen')`: ~0.7839 (cao, đúng kỳ vọng).
   - `get_most_similar('computer')`: Trả về các từ như 'computers', 'software', 'technology',...

4. **Nhận xét về độ tương đồng và các từ đồng nghĩa từ model pre-trained**  
    - Mô hình pre-trained `glove-wiki-gigaword-50` cho kết quả độ tương đồng giữa các cặp từ rất hợp lý. Ví dụ, cặp từ `'king'` và `'queen'` có độ tương đồng cosine cao (~0.78), phản ánh đúng mối quan hệ gần gũi về nghĩa.
    
    - Các từ đồng nghĩa hoặc liên quan như `'computer'` trả về các từ gần nhất là `'computers'`, `'software'`, `'technology'`, `'hardware'`,... Điều này cho thấy mô hình đã học được các mối liên hệ ngữ nghĩa phổ biến trong tiếng Anh.

#### Task 3: Document Embedding

1. **Cài đặt phương thức embed_document**  
   Đã bổ sung phương thức `embed_document(self, document: str)` vào lớp `WordEmbedder` trong file `src/representations/word_embedder.py`.  
   - Phương thức này nhận đầu vào là một chuỗi văn bản.
   - Sử dụng Tokenizer (cụ thể là RegexTokenizer từ Lab 1) để tách văn bản thành các token.
   - Lấy vector cho từng token, bỏ qua các từ không có trong từ điển (OOV).
   - Nếu không có từ nào hợp lệ, trả về vector 0.
   - Nếu có, tính trung bình cộng từng chiều của các vector từ để tạo vector văn bản.

2. **Đánh giá và kiểm thử**  
   Đã tạo file `test/lab4_test.py` để kiểm thử các chức năng:
   - Lấy vector cho từ `'king'`.
   - Tính độ tương đồng giữa `'king'` và `'queen'`, giữa `'king'` và `'man'`.
   - Lấy 10 từ giống nhất với `'computer'`.
   - Nhúng câu `"The queen rules the country."` và in ra vector kết quả.

   **Ví dụ mã kiểm thử:**
   ```python
   from Lab4.src.representations.word_embedder import WordEmbedder

    wordEmbedder = WordEmbedder('glove-wiki-gigaword-50')

    print("Vector for the word 'king':", wordEmbedder.get_vector('king'))
    print("Similarity between 'king' and 'queen':", wordEmbedder.get_similarity('king', 'queen'))
    print("Similarity between 'king' and 'man':", wordEmbedder.get_similarity('king', 'man'))
    print("10 most similar words to 'computer':", wordEmbedder.get_most_similar('computer'))
    sentence = "The queen rules the country."
    print(f"Embed the sentence '{sentence}':", wordEmbedder.embed_document(sentence))
   ```

   **Chạy kiểm thử:**
   Tại thư mục gốc `NLP` chạy:
   ```
    python -m Lab4.test.lab4_test
   ```

   **Kết quả mẫu:**
   - Vector for the word `'king'`: mảng numpy 50 chiều.
   - Similarity between `'king'`-`'queen'`: ~0.78.
   - Similarity between `'king'`-`'man'`: ~0.53.
   - 10 từ giống nhất với `'computer'`: `'computers'`, `'software'`, `'technology'`, `'electronic'`, `'internet'`, `'computing'`, `'devices'`, `'digital'`, `'applications'`, `'pc'`.
   - Embed the sentence `'The queen rules the country.'`: mảng numpy 50 chiều đại diện cho câu.

3. **Nhận xét**  
   - Phương pháp trung bình vector từ là một baseline đơn giản nhưng hiệu quả cho biểu diễn văn bản ngắn.
   - Kết quả embedding văn bản phản ánh tốt ý nghĩa tổng thể của câu nếu các từ chính đều nằm trong từ điển.
   - Nếu văn bản chứa nhiều từ OOV, vector kết quả sẽ kém ý nghĩa hơn.

#### Bonus Task: Training a Word2Vec Model from Scratch

Bên cạnh việc sử dụng các mô hình embedding pre-trained, đã thực hiện huấn luyện một mô hình Word2Vec mới trên tập dữ liệu tiếng Anh thực tế để hiểu rõ hơn về quy trình và khả năng tùy biến theo miền dữ liệu cụ thể.

**Các bước thực hiện trong script `test/lab4_embedding_training_demo.py`:**

1. **Đọc dữ liệu hiệu quả:**  
   Script đọc dữ liệu từ file `Data/UD_English-EWT/en_ewt-ud-train.txt` theo từng dòng, giúp tiết kiệm bộ nhớ khi xử lý tập dữ liệu lớn.

2. **Huấn luyện mô hình Word2Vec:**  
   Sử dụng thư viện `gensim`, mô hình Word2Vec được huấn luyện trên toàn bộ dữ liệu đã đọc. Các tham số như kích thước vector, cửa sổ ngữ cảnh, số lần lặp,… được thiết lập phù hợp với bài toán.

3. **Lưu mô hình:**  
   Sau khi huấn luyện xong, mô hình được lưu vào file `results/word2vec_ewt.model` để sử dụng lại mà không cần huấn luyện lại từ đầu.

4. **Trình diễn sử dụng mô hình:**  

    Script minh họa cách sử dụng mô hình vừa huấn luyện để tìm các từ tương tự (most similar) và giải các bài toán phép toán từ vựng (analogy), ví dụ: `'king' - 'man' + 'woman' ≈ 'queen'`.

    **Cách chạy demo:**

    Từ thư mục `Lab4` của dự án, chạy lệnh sau:
    ```
    python test/lab4_embedding_training_demo.py
    ```

    **Kết quả chạy demo**

    ```
    Most similar words to 'king':
   [('shocked', 0.9423750638961792), ('shady', 0.9412024021148682), ('mayko', 0.9380961656570435), ('muscle', 0.9367177486419678), ('repeatedly', 0.9354345202445984)]

    Analogy: king - man + woman = ?
   [('relaxed', 0.9148825407028198), ('birds', 0.9121307730674744), ('dirty', 0.9075636863708496), ('manner', 0.9055753946304321), ('bathroom', 0.9048301577568054)]
   ```

   - Các từ tương tự với `'king'` và kết quả phép toán từ vựng chưa thực sự hợp lý, cho thấy mô hình tự huấn luyện trên tập dữ liệu nhỏ hoặc không chuyên biệt có thể chưa học được các quan hệ ngữ nghĩa phức tạp như mô hình pre-trained.
   - Điều này nhấn mạnh tầm quan trọng của dữ liệu lớn và đa dạng khi huấn luyện embedding từ đầu.


    **Ý nghĩa:**  
    Việc tự huấn luyện mô hình giúp kiểm soát tốt hơn về miền dữ liệu, đồng thời cho phép so sánh trực tiếp với các mô hình pre-trained về chất lượng embedding, đặc biệt với các từ vựng chuyên ngành hoặc ít phổ biến.

#### Advanced Task: Scaling Word2Vec with Apache Spark

Trong thực tế, nhiều bài toán NLP cần xử lý tập dữ liệu văn bản rất lớn, vượt quá khả năng bộ nhớ của một máy đơn lẻ. Khi đó, việc huấn luyện Word2Vec bằng gensim không còn phù hợp. Apache Spark là một framework tính toán phân tán mạnh mẽ, cho phép xử lý và huấn luyện mô hình trên các tập dữ liệu lớn nhờ khả năng phân phối công việc lên nhiều máy.

**Lợi ích khi dùng Spark cho Word2Vec:**
- **Khả năng mở rộng:** Xử lý và huấn luyện trên tập dữ liệu lớn hơn nhiều lần so với RAM của một máy.
- **Tốc độ:** Phân tán công việc giúp tăng tốc quá trình huấn luyện.
- **Tích hợp hệ sinh thái Big Data:** Dễ dàng kết hợp với các pipeline xử lý dữ liệu lớn khác.


**Triển khai với PySpark**

1. **Cài đặt pyspark:**  
   Thêm `pyspark` vào `requirements.txt` và cài đặt:
   ```
   pip install -r requirements.txt
   ```
    hoặc 
    ```
    pip install pyspark
    ```
2. **Huấn luyện Word2Vec với Spark:**
   Đã tạo file huấn luyện `test/lab4_spark_word2vec_demo.py`
3. **Chạy**
Từ thư mục `Lab4`, chạy:
```
python test/lab4_spark_word2vec_demo.py
```

**Kết quả mong đợi**
- Script sẽ đọc dữ liệu JSON `c4-train.00000-of-01024-30K.json`, làm sạch và tách từ bằng Spark.
- Huấn luyện mô hình Word2Vec 100 chiều trên tập dữ liệu lớn.
- Hiển thị 5 từ giống nhất với `"computer"`.

**So sánh giữa model pre-trained và model tự huấn luyện**  
Mô hình pre-trained (như Glove) thường cho kết quả tốt hơn về độ bao phủ từ vựng và chất lượng vector, đặc biệt với các từ phổ biến hoặc có nhiều ngữ cảnh. Mô hình tự huấn luyện trên tập dữ liệu nhỏ (như UD English-EWT) có thể cho kết quả tốt với các từ xuất hiện nhiều trong tập, nhưng thường kém hơn với từ hiếm hoặc các mối quan hệ ngữ nghĩa phức tạp. Ngoài ra, model pre-trained có khả năng tổng quát hóa tốt hơn nhờ được huấn luyện trên tập dữ liệu lớn và đa dạng.