from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    
    def tokenize(self, text:str) -> List:

        '''Tách một chuỗi đầu vào thành 1 list các token'''
        pass

class Vectorizer(ABC):

    def fit(self, corpus: list[str]):
        '''
        Learns the vocabulary from a list of ducuments (corpus).
        '''
        pass

    def transform(self, documents: list[str]) -> list[list[int]]:
        '''
        Transform a list of documents into a list of count vectors based on the learned vocabulary.
        '''
        pass

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        '''
        A convernience method that performs fit and then transform on the same data.
        '''
        pass
