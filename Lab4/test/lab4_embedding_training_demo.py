import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# ========== 1. Đọc dữ liệu ==========

DATA_PATH = r"C:\Users\NGUYEN PHUONG BICH\HOC_TAP\NLP\Data\UD_English-EWT\en_ewt-ud-train.txt"
RESULT_PATH = r"C:\Users\NGUYEN PHUONG BICH\HOC_TAP\NLP\Lab4\results\word2vec_ewt.model"

def stream_sentences(file_path):
    """Đọc từng dòng và tách token để tiết kiệm bộ nhớ."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = simple_preprocess(line)
            if tokens:
                yield tokens

# ========== 2. Huấn luyện mô hình Word2Vec ==========
def train_word2vec(sentences):
    print("Training Word2Vec model ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,     # Kích thước embedding
        window=10,            # Kích thước ngữ cảnh
        min_count=5,         # Bỏ từ xuất hiện ít hơn 2 lần
        workers=4,           # Số luồng CPU
        sg=1,                # sg=1: Skip-gram, sg=0: CBOW
        epochs=10             # Số vòng lặp huấn luyện
    )
    print("Training complete!")
    return model

# ========== 3. Lưu mô hình ==========

def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path):
    print("Loading model ...")
    model = Word2Vec.load(model_path)
    print("Complete ...")
    return model

# ========== 4. Demo sử dụng mô hình ==========

def demo_usage(model):
    try:
        print("\nMost similar words to 'king':")
        print(model.wv.most_similar("king", topn=5))

        print("\nAnalogy: king - man + woman = ?")
        result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
        print(result)
    except KeyError:
        print("Một số từ không có trong từ vựng.")

# ========== 5. Chạy toàn bộ pipeline ==========
if __name__ == "__main__":
    sentences = list(stream_sentences(DATA_PATH))
    print(f"Loaded {len(sentences)} sentences.")
    model = train_word2vec(sentences)
    save_model(model, RESULT_PATH)

    # Tải lại model
    model = load_model(RESULT_PATH)
    demo_usage(model)
