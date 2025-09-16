# Báo cáo Lab 1: Text Tokenization

## Mục tiêu:

- Hiểu và triển khai bước tiền xử lý cơ bản trong NLP: tokenization.

- Viết 2 bộ tokenizer:

1. $\texttt{SimpleTokenizer}$: dựa trên split + xử lí dấu câu.

2. $\texttt{RegexTokenizer}$: sử dụng regex để tách token mạnh mẽ hơn.

- Kiểm thử tokenizers trên câu ví dụ và dữ liệu thực hiện (UD_English-EWT dataset).

## Triển khai

### 1. Interface $\texttt{Tokenizer}$

- Được định nghĩa trong $\texttt{src/core/interfaces.py}$.

- Là abstract base class với phương thức $\texttt{tokenize(self, text: str) -> list[str]}$.

### 2. `SimpleTokenizer (src/preprocessing/simple\_tokenizer.py)`
- Kế thừa từ $\texttt{Tokenizer}$.
- Cách hoạt động:
    
    - Chuyển text về lowercase.
    
    - Split theo whitespace.

    - Xử lí dấu câu $\texttt{".",",","?","!"}$ tách riêng khỏi từ.

- Ví dụ: $\texttt{"Hello, world!"}$ $\rightarrow$ $\texttt{["hello", ",", "world", "!"]}$.

### 3. `RegexTokenizer (src/preprocessing/regex_tokenizer.py)`

- Cũng kế thừa từ $\texttt{Tokenizer}$

- Dùng regex `\w+|[^\w\s]` để tách token:

    - `\w` bắt từ (bao gồm chữ và số).

    - `^\w\s` bắt các ký tự đặc biệt không phải chữ-số-khoảng trắng.

- Ưu điểm: gọn hơn, bao phủ nhiều trường hợp hơn so với split thủ công.

### 4. Kiểm thử ($\texttt{main.py}$)

- Test trên các câu mẫu:

    - "Hello, world! This is a test."

    - "NLP is fascinating... isn't it?"

    - "Let's see how it handles 123 numbers and punctuation!"

- Test trên dữ liệu thực: `{UD_English-EWT}` (500 ký tự đầu tiên).

- So sánh kết quả giữa 2 tokenizer.

## Kết quả quan sát

1. $\texttt{SimpleTokenizer}$ hoạt động tốt cho trường hợp dấu cơ bản, nhưng khó xử lí khi chuỗi phức tập (ví dụ: $\texttt{isn't}$ $\rightarrow$ $\texttt{["isn't"]}$ thay vì $\texttt{["isn", "'", "t"]}$).

2. $\texttt{RegexTokenizer}$ tách token tốt hơn, nhận diện từ + dấu câu rõ ràng, xử lí số (123) và từ viết tắt dễ hơn.

3. Khi áp dụng lên `UD_English-EWT`:

- $\texttt{SimpleTokenizer}$: tách từ + dấu câu nhưng đôi khi giữ nguyên dấu ' trong từ ghép.

- $\texttt{RegexTokenizer}$: output đa dạng và chính xác hơn, ví dụ $\texttt{"don't"}$ $\rightarrow$ $\texttt{["don", "'", "t"]}$.

<pre>
```
Simple Tokenizer Results:
Hello, world! This is a test. -> ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
NLP is fascinating... isn't it? -> ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
Let's see how it handles 123 numbers and punctuation! -> ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
===============================================
Regex Tokenizer Results:
Hello, world! This is a test. -> ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
NLP is fascinating... isn't it? -> ['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
Let's see how it handles 123 numbers and punctuation! -> ['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the'] 
```
</pre>


## Đóng góp và học được

- Hiểu rõ nguyên lí tách token và tầm quan trọng trong NLP pipeline.

- Viết được interface chung cho tokenizer $\rightarrow$ dễ mở rộng thêm tokenizer khác.

- Triển khai và so sánh 2 cách tiếp cận:

    - Rule-based đơn giản (SimpleTokenizer).

    - Regex-based (tổng quát và mạnh mẽ hơn).

- Thấy rõ sự khác biệt khi chạy trên dữ liệu thực tế (UD_English-EWT).

- Xây dựng project có cấu trúc module rõ ràng, dễ import và mở rộng cho các lab sau.