from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

if __name__ == "__main__":
    text1 = "Hello, world! This is a test."
    text2 = "NLP is fascinating... isn't it?"
    text3 = "Let's see how it handles 123 numbers and punctuation!"

    simple_tokenizer = SimpleTokenizer()
    print("Simple Tokenizer Results:")
    print(f'{text1} -> {simple_tokenizer.tokenize(text1)}')
    print(f'{text2} -> {simple_tokenizer.tokenize(text2)}')
    print(f'{text3} -> {simple_tokenizer.tokenize(text3)}')
    print('===============================================')
    
    regex_tokenizer = RegexTokenizer()
    print("Regex Tokenizer Results:")
    print(f'{text1} -> {regex_tokenizer.tokenize(text1)}')
    print(f'{text2} -> {regex_tokenizer.tokenize(text2)}')
    print(f'{text3} -> {regex_tokenizer.tokenize(text3)}')

    # ... (your tokenizer imports and instantiations) ...
    dataset_path = r"..\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    # Take a small portion of the text for demonstration
    sample_text = raw_text[:500] # First 500 characters
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")