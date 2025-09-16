import os
import sys

current_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))
sys.path.append(src_path)

from preprocessing.regex_tokenizer import RegexTokenizer
from representations.count_vectorizer import CountVectorizer


if __name__ == "__main__":
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    count_vectorizer = CountVectorizer(tokenizer=RegexTokenizer())
    vocabulary, vectors = count_vectorizer.fit_transform(corpus)

    print('==========With simple corpus:==========')
    print('The learned vocabulary:', vocabulary)
    print('Resulting document-term matrix:',vectors)

    print("\n===============================================\n")
    with open(r"..\..\UD_English-EWT\en_ewt-ud-train.txt", 'r', encoding='utf-8') as f:
        corpus2 = f.readlines()
    
    vocabulary, vectors = count_vectorizer.fit_transform(corpus2[:5])

    print('==========With UD_English-EWT corpus:==========')
    print('The learned vocabulary:', vocabulary)
    print('Resulting document-term matrix:',vectors)