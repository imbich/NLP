import os
import sys

current_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

core_path = os.path.abspath(os.path.join(current_dir, "..", "core"))

sys.path.append(core_path)
from interfaces import Vectorizer, Tokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = dict[str, int]()
    
    def fit(self, corpus: list[str]):
        '''
        - Khởi tạo một tập hợp rỗng để giữ các token duy nhất.
        - Lặp qua từng tài liệu trong kho dữ liệu.
        - Với mỗi tài liệu, sử dụng tokenizer đã được cung cấp để lấy ra danh sách các token.
        - Thêm tất cả các token vào tập hợp có được một từ vựng duy nhất.
        - Sau khi đã xử lí hết tài liệu, tạo một từ điển vocabulary_ ánh xạ mỗi token với một chỉ số nguyên duy nhất.
        '''
        vocab_set = set()
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            vocab_set.update(tokens)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocab_set))}

    def transform(self, documents: list[str]) -> list[list[int]]:
        '''
        - Với mỗi tài liệu trong danh sách đầu vào:
            - Tạo một vector 0 có độ dài bằng với kích thước của từ điển vocabulary_.
            - Với mỗi token, nếu tồn tại trong vocabulary_ thì tăng số đếm tại chỉ số tương ứng trong vector 0.
        - Trả về danh sách các vector kết quả.
        '''
        if not self.vocabulary_:
            raise ValueError("Vocabulary is empty. Please fit the vectorizer before transforming data.")
        
        vectors = []
        for document in documents:
            tokens = self.tokenizer.tokenize(document)
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    vector[index] += 1
            vectors.append(vector)
        return vectors
    
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.vocabulary_, self.transform(corpus)