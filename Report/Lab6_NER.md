# Báo cáo Lab 6: Nhận dạng thực thể tên (NER)
## 1. Mục tiêu chung
- Áp dụng Mạng Nơ-ron hồi quy (RNN) để xây dựng một mô hình hoàn chỉnh cho bài toán Nhận dạng thực thể tên (Named Entity Recognition - NER)

## 2. Triển khai
- Mã nguồn được đặt trong file : `NLP/Lab6/rnn_for_ner.ipynb`

- Sử dụng bộ dữ liệu `CoNLL 2003`, một trong những bộ dữ liệu benchmark tiêu chuẩn cho bài toán NER.

- Dữ liệu được gán nhãn theo định dạng IOB(Inside, Outside, Beginning). Ví dụ:
    - $\texttt{B-PER}$: Bắt đầu một thực thể Tên người (Person).
    - $\texttt{I-PER}$: Bên trong một thực thể Tên người.
    - $\texttt{B-LOC}$: Bắt đầu một thực thể Địa điểm (Location).
    - $\texttt{I-Loc}$: Bên trong một thực thể Địa điểm.
    - $\texttt{O}$: Không phải là một thực thể (Outside).


### Task 1: Tải và Tiền xử lý Dữ liệu
1. Tải dữ liệu từ Hugging Face
    - Sử dụng hàm `datasets.load_dataset("conll2003")` để tải bộ dữ liệu
    - Trả về `DatasetDict` :
        ```python
        DatasetDict({
            train: Dataset({
                features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
                num_rows: 14041
            })
            validation: Dataset({
                features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
                num_rows: 3250
            })
            test: Dataset({
                features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
                num_rows: 3453
            })
        })

        ```
2. Trích xuất câu và nhãn
    - Kết quả của quá trình trích xuất:
    ```python
        Danh sách nhãn: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC',         'B-MISC', 'I-MISC']
        Sentence: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
        NER IDs: [3, 0, 7, 0, 0, 0, 7, 0, 0]
        NER Labels: ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
    ```

3. Xây dựng từ điển

    - $\texttt{word\_to\_ix}$: Ánh xạ mỗi từ duy nhất sang một chỉ số nguyên.
    - $\texttt{tag\_to\_ix}$: Ánh xạ mỗi nhãn NER duy nhất sang một chỉ số nguyên.
    - Kết quả: 
        ```python
        Số lượng từ trong word_to_ix: 23625
        Số lượng nhãn trong tag_to_ix: 9
        Ví dụ trong word_to_ix: [('<PAD>', 0), ('<UNK>', 1), ('EU', 2), ('rejects', 3), ('German', 4), ('call', 5), ('to', 6), ('boycott', 7), ('British', 8), ('lamb', 9)]
        tag_to_ix: {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        ```

### Task 2: Tạo PyTorch Dataset và DataLoader
1. Tạo lớp $\texttt{NERDataset}$: Kế thừa $\texttt{torch.utils.data.Dataset}$
2. Tạo $\texttt{DataLoader}$
- Khởi tạo $\texttt{DataLoader}$ cho cả train và validation
- Hàm $\texttt{collate\_fn}$ để đệm các câu và nhãn trong cùng một batch về cùng độ dài.

### Task 3: Xây dựng mô hình RNN
- Mô hình bao gồm 3 lớp chính:
    - $\texttt{nn.Embedding}$: Chuyển đổi chỉ số của từ thành vector
    - $\texttt{nn.LSTM}$ (Bi-LSTM): Xử lý chuỗi vector embedding.
    - $\texttt{nn.Linear}$: Ánh xạ output của LSTM sang không gian nhãn để dự đoán.

- Khởi tạo mô hình với $\texttt{vocab\_size=len(word\_to\_ix), embedding\_dim=100, hidden\_dim=128, output\_size=len(tag\_to\_ix), padding\_idx=word\_to\_ix["<PAD>"]}$

### Task 4: Huấn luyện mô hình
1. Khởi tạo
    ```python
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=padding_tag_id)
    ```
2. Vòng lặp huấn luyện
    ```python
    Epoch 1/5 - Loss trung bình: 0.5822
    Epoch 2/5 - Loss trung bình: 0.2606
    Epoch 3/5 - Loss trung bình: 0.1477
    Epoch 4/5 - Loss trung bình: 0.0862
    Epoch 5/5 - Loss trung bình: 0.0480
    ```
### Task 5: Đánh giá Mô hình
- Kết quả đánh giá trên Test và Val
    ```python
    ============================== Trên tập Test ==============================
    Accuracy (token-level): 0.9286
    Precision: 0.7149, Recall: 0.6050, F1-score: 0.6554

    Báo cáo chi tiết theo thực thể:
                precision    recall  f1-score   support

            LOC       0.83      0.70      0.76      1668
            MISC       0.71      0.57      0.63       702
            ORG       0.74      0.49      0.59      1661
            PER       0.61      0.64      0.62      1617

    micro avg       0.71      0.60      0.66      5648
    macro avg       0.72      0.60      0.65      5648
    weighted avg       0.72      0.60      0.65      5648

    ============================== Trên tập Val ==============================
    Accuracy (token-level): 0.9514
    Precision: 0.8103, Recall: 0.7129, F1-score: 0.7585

    Báo cáo chi tiết theo thực thể:
                precision    recall  f1-score   support

            LOC       0.89      0.78      0.83      1837
            MISC       0.88      0.71      0.79       922
            ORG       0.76      0.61      0.68      1341
            PER       0.74      0.72      0.73      1842

    micro avg       0.81      0.71      0.76      5942
    macro avg       0.82      0.71      0.76      5942
    weighted avg       0.81      0.71      0.76      5942

    Độ chính xác trên tập validation: 0.9514
    ```

- Ví dụ dự đoán với câu mới (sử dụng hàm $\texttt{predict\_sentence(sentence)}$):
    - Câu : **"VNU University is located in Hanoi"**
    - Dự đoán :
        ```python
        VNU	B-ORG
        University	I-ORG
        is	O
        located	O
        in	O
        Hanoi	B-LOC
        ```

