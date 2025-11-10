from typing import List, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self._model = None

    def fit (self, texts: List[str], labels: List[int]) -> None:
        """Huấn luyện mô hình phân loại văn bản."""
        X = self.vectorizer.fit_transform(texts)
        if type(X) is tuple:
            X = X[1]
        self._model = LogisticRegression(random_state=0, max_iter=1000, solver="liblinear")
        self._model.fit(X, labels)
    
    def predict(self, texts: List[str]) -> List[int]:
        """Dự đoán nhãn cho các văn bản mới."""
        if self._model is None:
            raise ValueError("Model has not been trained yet.")
        X = self.vectorizer.transform(texts)
        return self._model.predict(X).tolist()
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Đánh giá hiệu suất của mô hình."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }