# Xử lý Ngôn ngữ Tự nhiên và Học Sâu

Dự án lưu trữ các bài thực hành (Lab) môn Xử lý Ngôn ngữ Tự nhiên và Học Sâu.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Các bài Lab](#các-bài-lab)
- [Cài đặt](#cài-đặt)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)

## Giới thiệu

Repository này chứa các bài thực hành về Xử lý Ngôn ngữ Tự nhiên (NLP) và Học Sâu (Deep Learning), bao gồm các chủ đề từ cơ bản đến nâng cao như text preprocessing, word embeddings, RNN, và Transformers.

## Cấu trúc dự án

```
NLP/
├── Lab2/                       
│   ├── src/
│   │   ├── core/
│   │   ├── preprocessing/
│   │   ├── representations/
│   │   └── ...
│   ├── test/
│   └── main.py
├── Lab4/                      
│   ├── src/
│   ├── test/
│   └── results/
├── Lab5/                       
│   ├── src/
│   ├── test/
│   └── lab5.ipynb
├── Lab6/                      
│   ├── pytorch_introduction.ipynb
│   ├── rnn_for_ner.ipynb
│   ├── rnn_for_pos_tagging.ipynb
│   ├── rnn_token_classification.ipynb
│   ├── rnns_text_classifiaction.ipynb
│   └── hwu/
├── Lab7/                       
│   ├── transformer_introduction.ipynb
│   └── reports/
├── Lab8/                       
├── Data/                       
│   └── UD_English-EWT/
├── Report/                     # Báo cáo các bài Lab
├── .venv/                      
├── requirements.txt            # Dependencies
├── .gitignore
└── README.md
```

## Các bài Lab

### Lab 2: Text Preprocessing & Representations
- Tiền xử lý văn bản (tokenization, normalization, stemming, lemmatization)
- Biểu diễn văn bản (Bag of Words, TF-IDF, Word Embeddings)
- Cấu trúc source code modular với các module core, preprocessing, representations

### Lab 4: WordEmbedding
- Hiểu và thực hành sử dụng các mô hình word embedding phổ biến (Word2Vec, Glove hoặc fastText).

- Kết quả thử nghiệm được lưu trong thư mục `results/`

### Lab 5: Text Classifier
- Xây dựng một quy trình phân loại văn bản hoàn chỉnh, từ văn bản thô đến mô hình học mát đã được huấn luyện, sử dụng các kỹ thuật Tokenization và Vectorization đã học từ những bài thực hành trước.

### Lab 6: Recurrent Neural Networks
- **PyTorch Introduction**: Giới thiệu framework PyTorch
- **RNN for NER**: Named Entity Recognition sử dụng RNN
- **RNN for POS Tagging**: Part-of-Speech Tagging
- **RNN Token Classification**: Phân loại token với RNN
- **Text Classification**: Phân loại văn bản với RNN
- Dataset: HWU (Hwu64) cho intent classification

### Lab 7: Transformers
- Giới thiệu kiến trúc Transformer
- Self-attention mechanism
- Applications của Transformers trong NLP

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd NLP
```

### 2. Tạo virtual environment

```bash
python -m venv .venv
```

### 3. Kích hoạt virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 5. Tải spaCy language model

```bash
python -m spacy download en_core_web_sm
```

## Yêu cầu hệ thống

- Python 3.8+
- pip
- Git

## Công nghệ sử dụng

### Thư viện NLP
- **spaCy** (3.8.7): NLP pipeline và language models
- **Gensim** (4.3.3): Topic modeling và word embeddings
- **Transformers** (4.56.1): State-of-the-art NLP models
- **Tokenizers** (0.22.0): Fast tokenization

### Deep Learning Frameworks
- **PyTorch**: Deep learning framework chính
- **PySpark** (4.0.1): Distributed computing

### Data Processing
- **NumPy** (1.26.4): Numerical computing
- **Pandas** (via pydantic): Data manipulation
- **SciPy** (1.13.1): Scientific computing

### Development Tools
- **Jupyter**: Interactive notebooks
- **IPython**: Enhanced Python shell
- **debugpy**: Debugging support

### Utilities
- **tqdm**: Progress bars
- **Rich**: Terminal formatting
- **Click**: Command-line interfaces

---

**Ghi chú**: Dự án được phát triển cho mục đích học tập môn Xử lý Ngôn ngữ Tự nhiên và Học Sâu.