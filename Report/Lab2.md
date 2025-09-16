# Báo cáo Lab 1: Count Vectorization

## 1. Mục tiêu

- Hiểu và cài đặt mô hình Bag-of-words cơ bản thông qua $\texttt{CountVectorizer}$.

- Tái sử dụng $\texttt{Tokenizer}$ từ Lab1 để tiền xử lí văn bản trước khi biến đổi thành vector số.

## 2. Nội dung thực hiện

- Task 1: Xây dựng Interface cho Vectorizer

    - Tạo abstract class $\texttt{Vectorizer}$ trong $\texttt{src/core/interfaces.py}$.

    - Định nghĩa 3 phương thức trừu tượng:

        - $\texttt{fit(self, corpus: list[str])}$ $\rightarrow$ học từ vựng từ tập văn bản.

        - $\texttt{transform(self, documents: list[str])}$ $\rightarrow$ $\texttt{list[list[int]]}$ $\rightarrow$ biến đổi văn bản thành vector đếm.

        - $\texttt{fit_transform(self, corpus: list[str])}$ $\rightarrow$ $\texttt{list[list[int]]}$ $\rightarrow$ kết hợp cả $\texttt{fit}$ và $\texttt{transform}$.

- Task 2: Cài đặt $\texttt{CountVectorizer}$

    - Tạo file $\texttt{src/representations/count_vectorizer.py}$

    - Xây dựng class $\texttt{CountVectorizer}$ kế thừa $\texttt{Vectorizer}$:

        - Hàm khởi tạo nhận một $\texttt{Tokenizer}$

        - Thuộc tính $\texttt{vocabulary_}$ (dict) để ánh xạ từ $\rightarrow$ chỉ số.

    - Cài đặt các phương thức:

        - $\texttt{fit}$: duyệt qua toàn bộ corpus, tokenize từng văn bản và xây dựng $\texttt{vocabulary_}$.

        - $\texttt{transform}$: tạo vector đếm cho mỗi văn bản dựa trên $\texttt{vocabulary_}.

        - $\texttt{fit_transform}$: gọi lần lượt $\texttt{fit}$ và $\texttt{transform}$.

- Task 3: Evaluation

    - Tạo file $\texttt{test/lab2_test.py}$

    - Import $\texttt{RegexTokenizer}$ từ Lab1 và $\texttt{CountVectorizer}$ từ Lab 2.

    - Thử nghiệm trên corpus mẫu:
    
    <pre> 
    ```python 
    corpus = [ 
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI." ] 
    ``` 
    </pre>

    - Kết quả in ra:

        - Vocabulary (từ điển ánh xạ từ $\rightarrow$ số).

        - Document-term matrix (biểu diễn vector cho từng câu).

         <pre> 
        ```python
        ==========With simple corpus:==========
       
        The learned vocabulary: {'.': 0, 'AI': 1, 'I': 2, 'NLP': 3, 'a': 4, 'is': 5, 'love': 6, 'of': 7, 'programming': 8, 'subfield': 9}
        Resulting document-term matrix: [[1, 0, 1, 1, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1, 0, 1]]
        ```</pre>

## 3. Kết quả đạt được

- Hiểu rõ cơ chế của Bag-of-words và cách biến văn bản thành vector số.

- Tái sử dụng $\texttt{RegexTokenizer}$ từ Lab1 $\rightarrow$$ cho thấy tính module hóa trong thiết kế.

- Hoàn thiện pipeline cơ bản cho NLP:

    - Bước 1: Tokenization (Lab1).

    - Bước 2: Vectorization (Lab2).

## 4. Đóng góp cá nhân

- Viết test để kiểm chứng hoạt động của $\texttt{CountVectorizer}$ với dữ liệu thực hiện $\texttt{UD_English-EWT}$

<pre>
```python
==========With UD_English-EWT corpus:==========
The learned vocabulary: {'!': 0, ',': 1, '-': 2, '.': 3, '2': 4, '3': 5, ':': 6, 'Abdullah': 7, 'Al': 8, 'American': 9, 'Ani': 10, 'Baghdad': 11, 'DPA': 12, 'Interior': 13, 'Iraqi': 14, 'Ministry': 15, 'MoI': 16, 'Qaim': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'by': 35, 'causing': 36, 'cells': 37, 'cleric': 38, 'come': 39, 'for': 40, 'forces': 41, 'had': 42, 'in': 43, 'killed': 44, 'killing': 45, 'mosque': 46, 'near': 47, 'of': 48, 'officials': 49, 'operating': 50, 'preacher':': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'by': 35, 'causing': 36, 'cells': 37, 'cleric': 38, 'come': 39, 'for': 40, 'forces': 41, 'had': 42, 'in': 43, '': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'b': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'b': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, ': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'b': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, ': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, ': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, ': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'by': 35, 'causing': 36, 'cells': 37, 'cleric': 38, 'come': 39, 'for': 40, 'forces': 41, 'had': 42, 'in': 43, 'killed': 44, 'killing': 45, 'mosque': 46, 'near': 47, 'of': 48, 'officials': 49, 'operating': 50, 'preacher': 51, 'respected': 52, 'run': 53, 'terrorist': 54, 'that': 55, 'the': 56, 'them': 57, 'they': 58, 'to': 59, 'town': 60, 'trouble': 61, 'up': 62, 'us': 63, 'were': 64, 'will': 65, 'years': 66}
Resulting document-term matrix: [[0, 1, 2, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
```</pre>

- Đảm bảo code tuân theo cấu trúc project, dễ mở rộng cho các vectorizer khác (TF-IDF, Word2Vec,...)




