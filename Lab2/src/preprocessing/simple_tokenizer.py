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

class SimpleTokenizer(Tokenizer):

    def tokenize(self, text: str):
        '''
        - Chuyển văn bản về chữ thường.
        - Phân tách văn bản thành các token dựa trên khoảng trắng.
        - Xử lý các dấu câu cơ bản (ví dụ: . , ? !) bằng cách tách chúng ra khỏi từ.
        '''
        result = []
        tokens = text.lower().strip().split()
        for token in tokens:
            if re.search(r'[.,!?;]', token):
                token = re.sub(r'([.,!?;])', r' \1 ', token)
            result.extend(token.split())

        return result
    