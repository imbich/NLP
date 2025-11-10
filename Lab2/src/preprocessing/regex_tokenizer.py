import os
import sys
import re
# Lấy đường dẫn thư mục cha của notebook
current_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
data_path = os.path.abspath(os.path.join(current_dir, "..", "core"))

# Thêm vào sys.path
sys.path.append(data_path)
from interfaces import Tokenizer

class RegexTokenizer(Tokenizer):

    def tokenize(self, text: str):
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return tokens