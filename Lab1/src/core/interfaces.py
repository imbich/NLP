from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    
    def tokenize(self, text:str) -> List:

        '''Tách một chuỗi đầu vào thành 1 list các token'''
        pass

class Vectorizer(ABC):

    def fit(self, corpus: list[str]):
        '''Học từ vựng từ danh sách tài liệu'''
        pass

    def transform(self, documents: list[str]) -> list[list[int]]:
        '''Chuyển đổi một danh sách các tài liệu thành danh sách các vector đếm dựa trên từ vựng đã học'''
        pass

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass
