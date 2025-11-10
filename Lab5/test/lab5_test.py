
from Lab2.src.representations.count_vectorizer import CountVectorizer
from Lab2.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab5.src.models.text_classifier import TextClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
    ]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
print(X_train)
regexTokenizer = RegexTokenizer()
countVectorizer = CountVectorizer(tokenizer=regexTokenizer)
textClassifier_count = TextClassifier(vectorizer=countVectorizer)
textClassifier_count.fit(X_train, y_train)

y_pred_count = textClassifier_count.predict(X_test)
print(f"Count Vectorizer Evaluation\n", textClassifier_count.evaluate(y_test, y_pred_count))

tfidfVectorizer = TfidfVectorizer()
textClassifier_tfidf = TextClassifier(vectorizer=tfidfVectorizer)
textClassifier_tfidf.fit(X_train, y_train)

y_pred_tfidf = textClassifier_tfidf.predict(X_test)
print("="*100)
print(f"TF-IDF Vectorizer Evaluation\n", textClassifier_tfidf.evaluate(y_test, y_pred_tfidf))