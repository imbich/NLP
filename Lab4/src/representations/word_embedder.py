import gensim
import numpy as np
import gensim.downloader as api
from Lab2.src.preprocessing.regex_tokenizer import RegexTokenizer

class WordEmbedder:
    def __init__(self, model_name: str='glove-wiki-gigaword-50'):
        
        self.model_name = model_name
        self.model = api.load(self.model_name)

    def get_vector(self, word: str):
        if word is None:
            return None
        
        for candidate in (word, word.lower()):
            if candidate in self.model:
                return self.model[candidate]
            try:
                vec = self.model[candidate]
                return np.asanyarray(vec, dtype=float)
            except KeyError:
                continue    
        return None
    
    def get_similarity(self, word1: str, word2: str):
        if word1 is None or word2 is None:
            return None
        try:
            return self.model.similarity(word1, word2)
        except KeyError:
            return None
        
    def get_most_similar(self, word1: str, top_n: int=10):
        if word1 is None:
            return None
        try:
            return self.model.most_similar(word1, topn=top_n)
        except KeyError:
            return None
        
    def embed_document(self, document: str):
        tokenizer = RegexTokenizer()
        vectors = []
        dim = getattr(self.model, "vector_size", None)

        for token in tokenizer.tokenize(document):
            vector = self.get_vector(token)
            if vector is None:
                vector = [0] * dim
            vectors.append(vector)

        return np.mean(np.vstack(vectors), axis=0)
