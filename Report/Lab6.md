# Báo cáo Lab 6: PyTorch, RNN/LSTM, PoS

## 1. Mục tiêu chung
- Làm quen với thư viện học sâu.
- Hiểu rõ hạn chế của các mô hình phân loại văn bản truyền thống (Bag-of-Words, Word2Vec trung bình).
- Nắm vững kiến trúc và luồng hoạt động của pipeline sử dụng RNN/LSTM cho bài toán phân loại văn bản.
- Phân tích và đánh giá sức mạnh của mô hình chuỗi trong việc “hiểu” ngữ cảnh của câu.
- Áp dụng các kiến thức lý thuyết về Mạng Nơ-ron
Hồi quy (RNN) đã học để xây dựng một mô hình hoàn chỉnh cho bài toán Part-of-Speech
(POS) Tagging.

## 2. Triển khai

### Part 1: Làm quen với PyTorch

- Mã nguồn nằm trong file : `Lab6/pytorch_introduction.ipynb`

#### Phần 1: Khám phá Tensor

##### Task 1: Tạo Tensor
- Tạo tensor từ list
    ``` python
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"Tensor từ list:\n {x_data}\n")
    ```
    - Kết quả: 
    ```python
    Tensor từ list:
    tensor([[1, 2],
            [3, 4]])
    ```

- Tạo tensor từ NumPy array
    ```python
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f"Tensor từ NumPy array:\n {x_np}\n")
    ```
    - Kết quả: 
    ```python
    # Tạo tensor từ list
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"Tensor từ list:\n {x_data}\n")
    ```

- Tạo tensor với các giá trị ngẫu nhiên hoặc hằng số
    ```python
    x_ones = torch.ones_like(x_data) # tạo tensor gồm các số 1 có cùng shape với x_data
    print(f"Ones Tensor:\n {x_ones}\n")
    x_rand = torch.rand_like(x_data, dtype=torch.float) # tạo tensor ngẫu nhiên
    print(f"Random Tensor:\n {x_rand}\n")

    # In ra shape, dtype, và device của tensor
    print(f"Shape của tensor: {x_rand.shape}")
    print(f"Datatype của tensor: {x_rand.dtype}")
    print(f"Device lưu trữ tensor: {x_rand.device}")
    ```
    - Kết quả:
    ```python
    Ones Tensor:
    tensor([[1, 1],
            [1, 1]])

    Random Tensor:
    tensor([[0.9259, 0.4644],
            [0.0732, 0.7929]])

    Shape của tensor: torch.Size([2, 2])
    Datatype của tensor: torch.float32
    Device lưu trữ tensor: cpu
    ```
##### Task 1.2: Các phép toán trên Tensor
1. Cộng `x_data` với chính nó.
2. Nhân `x_data` với 5.
3. Nhân ma trận `x_data` với `x_data.T`

    ```python
    print(f"Cộng x_data với chính nó: \n{x_data} \n+\n{x_data} \n=", x_data + x_data)
    print(f"Nhân x_data với 5: \n{x_data} \n * 5 \n=", x_data * 5)
    print(f"Nhân x_data với x_data.T: \n{x_data} \n@\n{x_data.T} \n=", x_data @ x_data.T)
    ```

    - Kết quả:
    ```python
    Cộng x_data với chính nó: 
    tensor([[1, 2],
            [3, 4]]) 
    +
    tensor([[1, 2],
            [3, 4]]) 
    = tensor([[2, 4],
            [6, 8]])

    Nhân x_data với 5: 
    tensor([[1, 2],
            [3, 4]]) 
    * 5 
    = tensor([[ 5, 10],
            [15, 20]])

    Nhân x_data với x_data.T: 
    tensor([[1, 2],
            [3, 4]]) 
    @
    tensor([[1, 3],
            [2, 4]]) 
    = tensor([[ 5, 11],
            [11, 25]])
    ```

##### Task 1.3: Indexing và Slicing
Từ tensor `x_data`:
1. Lấy ra hàng đầu tiên.
2. Lấy ra cột thứ hai.
3. Lấy ra giá trị ở hàng thứ hai, cột thứ hai

    ```python
    print(f"Hàng đầu tiên của \n{x_data} là:", x_data[0])
    print(f"Cột thứ hai của \n{x_data} là:", x_data[:,1])
    print(f"Giá trị ở hàng thứ hai, cột thứ hai của \n{x_data} là:", x_data[1,1])
    ```

    - Kết quả:
    ```python
    Hàng đầu tiên của 
    tensor([[1, 2],
            [3, 4]]) là: tensor([1, 2])
    Cột thứ hai của 
    tensor([[1, 2],
            [3, 4]]) là: tensor([2, 4])
    Giá trị ở hàng thứ hai, cột thứ hai của 
    tensor([[1, 2],
            [3, 4]]) là: tensor(4)
    ```

##### Task 1.4: Thay đổi hình dạng Tensor
Sử dụng `reshape` để biến tensor có shape (4,4) thành tensor có shape (16,1)

- Kết quả:
    ```python
    import torch
    x = torch.arange(16).reshape(4,4)
    y = x.reshape(16,1)
    print(y)
    ```

    ```python
    Tensor có shape (4,4):
    tensor([[0.1016, 0.8452, 0.1346, 0.1692],
            [0.4619, 0.6292, 0.1144, 0.1913],
            [0.4614, 0.8410, 0.0188, 0.7149],
            [0.2466, 0.5514, 0.6366, 0.9481]])
    Tensor sau khi reshape (16,1):
    tensor([[0.1016],
            [0.8452],
            [0.1346],
            [0.1692],
            [0.4619],
            [0.6292],
            [0.1144],
            [0.1913],
            [0.4614],
            [0.8410],
            [0.0188],
            [0.7149],
            [0.2466],
            [0.5514],
            [0.6366],
            [0.9481]])
    ```
#### Phần 2: Tự động tính Đạo hàm với `autograd`
##### Task 2.1: Thực hành với `autograd`

```python
# Tạo một tensor và yêu cầu tính đạo hàm cho nó
x = torch.ones(1, requires_grad=True)
print(f"x: {x}")

# Thực hiện một phép toán
y = x + 2
print(f"y: {y}")

# y được tạo ra từ một phép toán có x, nên nó cũng có grad_fn
print(f"grad_fn của y: {y.grad_fn}")

# Thực hiện thêm các phép toán
z = y * y * 3

# Tính đạo hàm của z theo x
z.backward() # tương đương z.backward(torch.tensor(1.))

# Đạo hàm được lưu trong thuộc tính .grad
# Ta có z = 3 * (x+2)^2 => dz/dx = 6 * (x+2). Với x=1, dz/dx = 18
print(f"Đạo hàm của z theo x: {x.grad}")
```
- Kết quả:

    ```python
    x: tensor([1.], requires_grad=True)
    y: tensor([3.], grad_fn=<AddBackward0>)
    grad_fn của y: <AddBackward0 object at 0x000001D3D9FC80D0>
    Đạo hàm của z theo x: tensor([18.])
    ```
- Câu chuyện xảy ra khi gọi `z.backward()` một lần nữa: Sẽ gặp lỗi do graph đã được giải phóng (Xóa khỏi bộ nhớ) sau lần backward đầu tiên để tiết kiệm bộ nhớ, nên khi gọi lần tiếp theo thì các tensor trung gian đã bị xóa và không thể backward được

#### Phần 3: Xây dựng mô hình đầu tiên với `torch.nn`
##### Task 3.1: Lớp `nn.Linear` : Thực hiện phép biến đổi tuyến tính $y=xA^T + b$
```python
# Khởi tạo một lớp Linear biến đổi từ 5 chiều -> 2 chiều
linear_layer = torch.nn.Linear(in_features=5, out_features=2)

# Tạo một tensor đầu vào mẫu
input_tensor = torch.randn(3,5) # 3 mẫu, mỗi mẫu 5 chiều

# Truyền đầu vào qua lớp linear
output = linear_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n {output}")
```
- Kết quả:
```python
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
Output:
 tensor([[-0.9370,  0.3914],
        [-0.9572,  0.4896],
        [-0.7300,  0.1394]], grad_fn=<AddmmBackward0>)
```

##### Task 3.2: Lớp `nn.Embedding`
```python
# Khởi tạo lớp Embedding cho một từ điển 10 từ, mỗi từ biểu diễn bằng vector 3 chiều
embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)

# Tạo một tensor đầu vào chứa các chỉ số của từ (ví dụ: một câu)
# Các chỉ số phải nhỏ hơn 10
input_indices = torch.LongTensor([1, 5, 0, 8])

# Lấy ra các vector embedding tương ứng
embeddings = embedding_layer(input_indices)
print(f"Input shape: {input_indices.shape}")
print(f"Output shape: {embeddings.shape}")
print(f"Embeddings:\n {embeddings}")
```
- Kết quả:
```python
Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Embeddings:
 tensor([[ 9.4391e-01, -1.5359e+00, -3.1712e-01],
        [ 9.6256e-01, -6.3085e-01,  1.1433e-03],
        [ 9.5631e-01, -2.3418e-01, -3.4213e-01],
        [ 1.7969e-01, -3.3344e-01, -2.5940e-01]], grad_fn=<EmbeddingBackward0>)
```

##### Task 3.3: Kết hợp thành một `nn.Module`

```python
from torch import nn
class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        # Định nghĩa các lớp (layer) bạn sẽ dùng
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU() # Hàm kích hoạt
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, indices):
        # Định nghĩa luồng dữ liệu đi qua các lớp
        # 1. Lấy embedding
        embeds = self.embedding(indices)
        # 2. Truyền qua lớp linear và hàm kích hoạt
        hidden = self.activation(self.linear(embeds))
        # 3. Truyền qua lớp output
        output = self.output_layer(hidden)
        return output
    
# Khởi tạo và kiểm tra mô hình
model = MyFirstModel(vocab_size=100, embedding_dim=16, hidden_dim=8, output_dim=2)
input_data = torch.LongTensor([[1, 2, 5, 9]]) # một câu gồm 4 từ
output_data = model(input_data)
print(f"Model output shape: {output_data.shape}")
```
- Kết quả:
    ```python
    Model output shape: torch.Size([1, 4, 2])
    ```

#### Kết luận:
- Làm quen được với các thành phần cơ bản nhất của PyTorch.
- Cách tạo và thao tác với Tensor, hiểu cơ chế tự động tính đạo hàm.
- Xây dựng mạng nơ-ron đơn giản bằng `nn.Module`.

### Part 2: Phân loại văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

#### Mục tiêu: 
Xây dụng, huấn luyện và so sánh hiệu năng giữa các mô hình:
1. TF-IDF + Logistic Regression (Baseline 1).
2. Word2Vec (vector trung bình) + Dense Layer (Baseline 2).
3. Embedding Layer (pre-trained) + LSTM.
4. Embedding Layer (học từ đầu) + LSTM.

#### Phần 2: Thực hành

1. Đọc dữ liệu
    ```python
    df_train = pd.read_csv('hwu/train.csv', header=None, names=['text', 'intent'])
    df_val = pd.read_csv('hwu/val.csv', header=None, names=['text', 'intent'])
    df_test = pd.read_csv('hwu/test.csv', header=None, names=['text', 'intent'])
    ```
2. Tiền xử lý các nhãn (intent) để chuyển chúng thành dạng số.
    - fit LabelEncoder trên toàn bộ tập intent và transform trên các tập train/val/test
    ```python
    from sklearn.preprocessing import LabelEncoder
    intents = (
        df_train['intent'].tolist() +
        df_val['intent'].tolist() +
        df_test['intent'].tolist()
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(intents)
    df_train["intent"] = label_encoder.transform(df_train["intent"])
    df_val["intent"] = label_encoder.transform(df_val["intent"])
    df_test["intent"] = label_encoder.transform(df_test["intent"])
    ```

##### Nhiệm vụ 1: Pipeline TF-IDF + Logistic Regression
- Đây là mô hình baseline, cơ sở so sánh

    ```python
    # 1. Tạo một pipeline với TfidfVectorizer và LogisticRegression
    tfidf_lr_pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000),
        LogisticRegression(max_iter=1000)
    )

    # 2. Huấn luyện pipeline trên tập train
    tfidf_lr_pipeline.fit(df_train["text"], df_train["intent"])

    # 3. Đánh giá trên tập test
    y_pred = tfidf_lr_pipeline.predict(df_test["text"])
    loss_tfidf = log_loss(df_test["intent"], tfidf_lr_pipeline.predict_proba(df_test["text"]), labels=list(range(num_classes)))

    print(classification_report(y_true=df_test["intent"], y_pred=y_pred))
    print("Test loss", loss_tfidf)
    ```

##### Nhiệm vụ 2: Pipeline Word2Vec(Trung bình) + Dense Layer
- Mô hình baseline thứ 2, sử dụng embedding nhưng chưa có khả năng xử lý chuỗi.
    ```python
    # 1. Huấn luyện mô hình Word2Vec trên dữ liệu text của bạn
    sentences = [text.split() for text in df_train['text']]
    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # 2. Viết hàm để chuyển mỗi câu thành vector trung bình
    def sentence_to_avg_vector(text, model):
    # ... (Implement logic)
        words = text.split()
        vectors = []
        vectors = [model.wv[word] if word in model.wv else np.zeros(100) for word in words]   
        avg_vector = np.mean(vectors ,axis=0)
        return avg_vector

    # 3. Tạo dữ liệu train/val/test X_train_avg, X_val_avg, X_test_avg
    X_train_avg = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_train['text']])
    X_val_avg = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_val['text']])
    X_test_avg = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_test['text']])

    y_train = df_train["intent"].values
    y_val   = df_val["intent"].values
    y_test  = df_test["intent"].values

    num_classes = len(set(intents))

    # 4. Xây dựng mô hình Sequential của Keras
    model = Sequential([
        Dense(128, activation='relu', input_shape=(w2v_model.vector_size,)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # 5. Compile, huấn luyện và đánh giá mô hình
    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_avg, y_train,
        validation_data=(X_val_avg, y_val), 
        epochs=100,
        batch_size=32
    )

    y_pred_probs = model.predict(X_test_avg)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    ```

##### Nhiệm vụ 3: Mô hình Nâng cao (Embedding Pre-trained + LSTM)
- Sử dụng `Word2Vec` đã huấn luyện ở nhiệm vụ 2 để khởi tạo trọng số cho `Embedding Layer`
    ```python
    # 1. Tiền xử lý cho mô hình chuỗi
    # a. Tokenizer: Tạo vocab và chuyển text thành chuỗi chỉ số

    tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
    tokenizer.fit_on_texts(df_train["text"])

    # Chuyển text sang sequences
    train_sequences = tokenizer.texts_to_sequences(df_train["text"])
    val_sequences   = tokenizer.texts_to_sequences(df_val["text"])
    test_sequences  = tokenizer.texts_to_sequences(df_test["text"])

    # b. Padding: Đảm bảo các chuỗi có cùng độ dài
    max_len = 50
    X_train_pad = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    X_val_pad   = pad_sequences(val_sequences, maxlen=max_len, padding='post')
    X_test_pad  = pad_sequences(test_sequences, maxlen=max_len, padding='post')

    # 2. Tạo ma trận trọng số cho Embedding Layer từ Word2Vec
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    # 3. Xây dựng mô hình Sequential với LSTM
    lstm_model_pretrained = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix], # Khởi tạo trọng số
            input_length=max_len,
            trainable=False # Đóng băng lớp Embedding
        ),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
        ])

    # 4. Compile, huấn luyện (sử dụng EarlyStopping) và đánh giá
    lstm_model_pretrained.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Nhãn dạng số nguyên
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = lstm_model_pretrained.fit(
        X_train_pad, df_train["intent"].values,
        validation_data=(X_val_pad, df_val["intent"].values),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop]
    )

    y_pred_probs = lstm_model_pretrained.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print("Test loss:", model.evaluate(X_test_avg, y_test, verbose=1)[0])
    print("Test loss:", lstm_model_pretrained.evaluate(X_test_pad, y_test)[0])
    ```

##### Nhiệm vụ 4: Mô hình Nâng cao (Embedding học từ đầu + LSTM)
- Mô hình tự học lớp Embedding. Kiến trúc tương tự nhiệm vụ 3, nhưng Embedding Layer sẽ được học từ đầu.
    ```python
    # Dữ liệu đã được tiền xử lý (tokenized, padded) từ nhiệm vụ 3
    # 1. Xây dựng mô hình
    lstm_model_scratch = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=100, # Chọn một chiều embedding, ví dụ 100
            input_length=max_len
            # Không có weights, trainable=True (mặc định)
        ),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    # 2. Compile, huấn luyện và đánh giá mô hình
    lstm_model_scratch.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = lstm_model_scratch.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop]
    )

    y_pred_probs = lstm_model_scratch.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(classification_report(y_test, y_pred))
    print("Test loss:", lstm_model_scratch.evaluate(X_test_pad, y_test, verbose=1)[0])
    ```

##### Nhiệm vụ 5: Đáng giá, So sánh và Phân tích
1. So sánh định lượng:

    | Pipeline  | F1-score (Macro) | Test Loss |
    |-------|-------|-------|
    | TF-IDF + Logistic Regression  | 0.82 | 1.05 |
    | DWord2Vec (Avg) + Dense       | 0.32 | 2.42 |
    | Embedding (Pre-trained) + LSTM  | 0.04 | 3.59 |
    | Embedding (Scratch) + LSTM  | 0.00 | 4.13 |

- Nhận xét: 
    - TF-IDF + Logistic Regression cho kết quả vượt trội
        - Mô hình tuyến tính cổ điển hoạt động rất hiệu quả với tập dữ liệu này.
        - Lý do: 
            - Dữ liệu có thể không quá phức tạp về ngữ nghĩa $\rightarrow$ TF-IDF đủ để phân tách lớp.
            - Logistic Regression phù hợp khi số mẫu không quá lớn.
            - Không cần học embedding từ dữ liệu $\rightarrow$ ít overfitting hơn.

        $\rightarrow$ Vì vậy, mô hình đơn giản nhưng ổn định lại mang hiệu quả tốt nhất.
    - Word2Vec trung bình (Avg) + Dense mạng nông hoạt động yếu
        - Mặc dù Word2Vec nắm bắt được thông tin ngữ nghĩa, nhưng việc lấy trung bình embedding làm mất toàn bộ cấu trúc của câu:
            - Mọi thứ trở thành một vector duy nhất $\rightarrow$ mất thứ tự từ.
            - Mô hình Dense không đủ mạnh để tái tạo thông tin câu.
            - Nếu embedding Word2Vec huấn luyện trên chính dataset nhỏ $\rightarrow$ bị hạn chế mạnh về chất lượng.
        $\rightarrow$ Đây là hạn chế chung của phương pháp "average embedding".
    - Embedding Pretrained + LSTM cho kết quả rất thấp
        - Embedding pretrained từ Word2Vec được huấn luyện trên chính dữ liệu huấn luyện nhỏ $\rightarrow$ không tạo ra biểu diễn bền vững.
        - LSTM yêu cầu tập dữ liệu lớn hơn nhiều để học được chuỗi.
        - Nhiều từ không có trong Word2Vec $\rightarrow$ rơi vào `<UNK>` $\rightarrow$ mất dữ liệu ngữ nghĩa.
    - Embedding Scratch + LSTM: Tệ nhất
        - Mô hình gần như không học được gì.
        - Embedding khởi tạo ngẫu nhiên + dữ liệu không đủ lớn $\rightarrow$ embedding không hội tụ tốt.
        - Padding nhiều hoặc max_len không tối ưu $\rightarrow$ nhiễu tín hiệu vào LSTM.

2. Phân tích định tính
- Kết quả :

    | Sentence  | True label | TF-IDF + LR | Word2Vec Avg + Dense | Embedding Pre-trained + LSTM | Embedding Scratch + LSTM |
    |-------|-------|-------|-------|-------|-------|
    | can you remind me to not call my mom  | reminder_create | play_podcasts | datetime_query|iot_hue_lightoff|transport_taxi
    | is it going to be sunny or rainy tomorrow | weather_query | iot_wemo_on |alarm_remove|iot_hue_lightchange|transport_taxi
    | find a flight from new york to london but not through paris  | flight_search | cooking_recipe |general_praise|general_repeat|transport_taxi|

    1. Câu : "can you remind me to not call my mom"
        - Đây là câu có phủ định "to not call", yêu cầu mô hình phải hiểu mối quan hệ dài giữa các từ trong câu.
        - Mặc dù LSTM được thiết kế để xử lý chuỗi và nắm bắt dependencies dài, kết quả vẫn không chính xác. Điều này có thể do dữ liệu huấn luyện quá ít hoặc không đủ đa dạng để mô hình học được các cấu trúc phủ định phức tạp.
        - TF-IDF + LR vẫn dự đoán gần đúng ý định liên quan đến task (play/nhắc nhở), nhưng thực ra không thực sự "hiểu" ngữ cảnh, nó chỉ dựa trên tần suất từ khóa.
    2. Câu : "is it going to be sunny or rainy tomorrow"
        - Đây là câu hỏi về thời tiết, có từ khóa "sunny", "rainy", "tomorrow" gợi ý intent.
        - LSTM không dự đoán đúng, có thể do từ vựng liên quan đến thời tiết quá ít trong dữ liệu huấn luyện, nên mô hình không học được mối liên kết giữa từ khóa và intent.
        - TF-IDF + LR đôi khi lại “bắt” từ khóa tốt hơn, nên dự đoán gần hơn.
    3. Câu : "find a flight from new york to london but not through paris"
        - Đây là câu phức tạp với nhiều thông tin về địa điểm, bao gồm cả phủ định "but not through paris".
        - LSTM có khả năng lý thuyết để hiểu chuỗi dài và phủ định, nhưng nếu dữ liệu huấn luyện không có các ví dụ tương tự, nó sẽ không học được pattern này, dẫn đến dự đoán sai.
        - Câu này cho thấy rằng LSTM chỉ thực sự mạnh khi có đủ dữ liệu đa dạng để học các dependencies dài và cấu trúc phức tạp.

3. Nhận xét chung về ưu nhược điểm của từng phương pháp

    1. TF-IDF + Logistic Regression

    - Ưu điểm:
        - Triển khai đơn giản, nhanh, ít yêu cầu về tài nguyên.

        - Hoạt động tốt với dữ liệu nhỏ, đặc biệt khi intent có các từ khóa đặc trưng rõ ràng.

        - Cho kết quả F1-score cao nhất (0.82) trong các mô hình thử nghiệm, test loss thấp nhất (1.05).

    - Nhược điểm:

        - Không hiểu được ngữ cảnh hay thứ tự từ.

        - Khó xử lý các câu phức tạp, phủ định hoặc mối quan hệ dài giữa các từ.

        - Không tận dụng được thông tin ngữ nghĩa từ embedding.

    2. Word2Vec (Average) + Dense\
    - Ưu điểm:

        - Sử dụng embedding giúp nắm bắt ngữ nghĩa từ vựng, có thể so sánh similarity giữa các từ.

        - Mô hình dense đơn giản, dễ huấn luyện.

    - Nhược điểm:

        - Chỉ lấy trung bình vector từ, bỏ qua thứ tự từ và dependencies trong câu.

        - Kết quả F1 thấp (0.32), test loss cao (2.42).

        - Không hiệu quả với các câu phức tạp, vì mất thông tin về cấu trúc chuỗi.

    3. Embedding Pre-trained + LSTM
    - Ưu điểm:

        - Lý thuyết mạnh về xử lý chuỗi và dependencies dài, có thể nắm bắt phủ định, từ nối, thứ tự từ.

        - Sử dụng pre-trained embedding tận dụng ngữ nghĩa từ vựng đã học sẵn.

    - Nhược điểm:

        - Hiệu quả kém trong thực nghiệm (F1 ~0.04, test loss 3.59), do dữ liệu huấn luyện quá nhỏ hoặc không đủ đa dạng.

        - Dễ overfitting, khó tối ưu, cần nhiều tuning hyperparameters và data augmentation.

        - Kết quả dự đoán trên các câu ví dụ không chính xác, chứng tỏ LSTM chưa học được patterns phức tạp.
    
    4. Embedding Scratch + LSTM
    - Ưu điểm:

        - LSTM có khả năng học sequence dependencies trực tiếp từ dữ liệu của dự án.

        - Không phụ thuộc embedding pre-trained, có thể thích nghi với dữ liệu domain riêng.

    - Nhược điểm:

        - F1-score gần 0, test loss cao nhất (4.13).

        - Dữ liệu nhỏ làm cho embedding và LSTM khó học tốt, dẫn đến dự đoán không chính xác.

        - Huấn luyện lâu, khó converged.

### Part 3: Xây dựng mô hình RNN cho bài toán Part-of-Speech Tagging
#### Mục tiêu:
- Áp dụng các kiến thức lý thuyết về Mạng Nơ-ron Hồi quy (RNN) đã học để xây dựng một mô hình hoàn chỉnh cho bài toán Part-of-Speech
(POS) Tagging.

#### Các bước thực hiện
##### Task 1: Tải và tiền xử lý dữ liệu
1. Viết hàm đọc file `.conllu`
```python
def load_conllu(file_path):
    """
    Đọc file .conllu và trả về danh sách các câu.
    Mỗi câu là một list các tuple (word, upos_tag)
    """
    sentences = []
    sentence = []

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":  # kết thúc một câu
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            elif line.startswith("#"):  # comment → bỏ qua
                continue
            else:
                parts = line.split("\t")
                if len(parts) >= 4:
                    word = parts[1]
                    upos = parts[3]
                    sentence.append((word, upos))
        # Thêm câu cuối nếu file không kết thúc bằng line trống
        if sentence:
            sentences.append(sentence)
    return sentences

train_path = "../Data/UD_English-EWT/en_ewt-ud-train.conllu"
dev_path = "../Data/UD_English-EWT/en_ewt-ud-dev.conllu"

train_sentences = load_conllu(train_path)
dev_sentences   = load_conllu(dev_path)
```
2. Xây dựng từ điển
```python
# Từ điển word → index
word_to_ix = {}
for sent in train_sentences:
    for word, tag in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# Thêm token <UNK> cho từ không có trong từ điển
word_to_ix["<UNK>"] = len(word_to_ix)

# Từ điển tag → index
tag_to_ix = {}
for sent in train_sentences:
    for word, tag in sent:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

# In ra kích thước từ điển
print("Kích thước của word_to_ix:", len(word_to_ix))
print("Số nhãn UPOS:", len(tag_to_ix))
```
– `word_to_ix`: Ánh xạ mỗi từ duy nhất sang một chỉ số (index) nguyên. Thêm một token đặc biệt là `<UNK>` cho các từ không có trong từ điển.

– `tag_to_ix`: Ánh xạ mỗi nhãn UPOS duy nhất sang một chỉ số nguyên.

- Kết quả:
```python
Kích thước của word_to_ix: 20201
Số nhãn UPOS: 18
```

##### Task 2: Tạo PyTorch Dataset và DataLoader

1. Tạo lớp `POSDataset`

    ```python
    class POSDataset(Dataset):
        def __init__(self, sentences, word_to_ix, tag_to_ix):
            """
            sentences: Danh sách các câu đã xử lý.
            word_to_ix: Mapping của từ-> index
            tag_to_ix: Mapping của nhãn -> index
            """
    ```
    - Nhận vào danh sách các câu đã xử lý và hai từ điển `word_to_ix,
    tag_to_ix`.

    ```python
    def __len__(self):
        return len(self.sentences)
    ```
    - Trả về tổng số câu trong bộ dữ liệu.

    ```python
    def __getitem__(self, index):
        #...
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)
    ```
    - Nhận vào một index và trả về một cặp tensor: `(sentence_indices, tag_indices)`. Tensor này chứa các chỉ số của từ/nhãn trong câu tương ứng.

2. Tạo `DataLoader`

- Hàm đệm 
    ```python
    def collate_fn(batch):
        """
        batch: danh sách các (sentence_indices, tag_indices)
        Trả về:
            - sentences_padded: tensor (batch_size, max_len)
            - tags_padded: tensor (batch_size, max_len)
            - lengths: danh sách độ dài thực của từng câu
        """
        sentences, tags = zip(*batch)
        lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)
        
        # Pad sequences về cùng chiều dài
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        tags_padded      = pad_sequence(tags, batch_first=True, padding_value=-100)  # -100 cho ignore_index
        
        return sentences_padded, tags_padded, lengths
    ```
    - `collate_fn` để đệm các câu và nhãn trong cùng một batch về cùng độ dài.

    ```python
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    ```

##### Task 3: Xây dựng mô hình RNN
Mô hình sẽ bao gồm 3 lớp chính:
1. `nn.Embedding`: Chuyển đổi chỉ số của từ thành vector.
2. `nn.RNN`: Xử lý chuỗi vector embedding.
3. `nn.Linear`: Ánh xạ output của RNN sang không gian nhãn để dự đoán.

##### Task 4: Huấn luyện Mô hình
- Khởi tạo mô hình, optimizer (ví dụ: torch.optim.Adam), và loss function.
- Sử dụng `nn.CrossEntropyLoss` cho bài toán này.
- Trong mỗi vòng lặp huấn luyện : Thực hiện 5 bước kinh điển: (1) Xóa gradient cũ, (2) Forward pass, (3) Tính loss, (4) Backward pass (lan truyền ngược), (5) Cập nhật trọng số.
- In ra giá trị loss trung bình sau mỗi epoch.

    ```python
    Epoch 1/10, Loss: 1.1714
    Epoch 2/10, Loss: 0.6570
    Epoch 3/10, Loss: 0.4929
    Epoch 4/10, Loss: 0.3915
    Epoch 5/10, Loss: 0.3202
    Epoch 6/10, Loss: 0.2646
    Epoch 7/10, Loss: 0.2199
    Epoch 8/10, Loss: 0.1853
    Epoch 9/10, Loss: 0.1555
    Epoch 10/10, Loss: 0.1310
    ```

##### Task 5: Đánh giá mô hình
Quy trình
- Đặt mô hình ở chế độ đánh giá: model.eval().
- Tắt việc tính toán gradient: with torch.no_grad(): ...
- Lặp qua từng batch trong DataLoader của tập dev.
- Với mỗi batch, lấy dự đoán của mô hình bằng cách áp dụng torch.argmax trên chiều cuối cùng của output.
- So sánh dự đoán với nhãn thật để tính toán độ chính xác (accuracy). 

Hàm đánh giá kết quả của mô hình:
```python
def evaluate(model, data_loader, device="cpu"):
    """
    model: mô hình đã huấn luyện
    data_loader: DataLoader của tập dev/test
    device: 'cuda' hoặc 'cpu'
    
    Trả về: token-level accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            logits = model(sentences, lengths)  # (batch_size, seq_len, num_classes)
            preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            
            # Flatten để so sánh
            preds_flat = preds.view(-1)
            tags_flat = tags.view(-1)
            
            # Chỉ tính các token không phải padding
            mask = tags_flat != -100
            correct += (preds_flat[mask] == tags_flat[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

train_accuracy = evaluate(model, train_loader)
dev_accuracy = evaluate(model, dev_loader)

print(f"Độ chính xác trên Train: {train_accuracy*100:.2f}%")
print(f"Độ chính xác trên Dev: {dev_accuracy*100:.2f}%")
```

Kết quả đánh giá trên Train và trên Dev:
```python
Độ chính xác trên Train: 96.62%
Độ chính xác trên Dev: 87.91%
```

- Hàm `predict_sentence(sentence)` nhận vào một câu mới, xử lý và in ra các cặp (từ, nhãn_dự_đoán)

```python
def predict_sentence(sentence):
    # Chuyển câu thành index theo từ điển word_to_ix
    indices = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in s.split()]
    input_tensor = torch.tensor([indices], dtype=torch.long).to("cpu")  # batch_size=1
    lengths = torch.tensor([len(indices)], dtype=torch.long).to("cpu")

    # Dự đoán
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor, lengths)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    # Chuyển từ index về nhãn
    pred_tags = [(w, list(tag_to_ix.keys())[list(tag_to_ix.values()).index(p)]) for w, p in zip(s.split(), preds)]
    
    return pred_tags

s = "I love NLP"
print(f"Câu: {s}")
print(f"Dự đoán:", predict_sentence(s))
```
- Kết quả: 
```python
Câu: I love NLP
Dự đoán: [('I', 'PRON'), ('love', 'VERB'), ('NLP', 'NOUN')]
```

