from typing import List
import re

class Tokenizer:
    
    def tokenize(self, text:str)-> List:
        '''Tách một chuỗi đầu vào thành 1 list các token'''
        
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return tokens

    