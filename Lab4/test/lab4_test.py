from Lab4.src.representations.word_embedder import WordEmbedder

wordEmbedder = WordEmbedder('glove-wiki-gigaword-50')

print("="*50)
print("Vector for the word 'king':", wordEmbedder.get_vector('king'))
print("="*50)
print("Similarity between 'king' and 'queen':", wordEmbedder.get_similarity('king', 'queen'))
print("Similarity between 'king' and 'man':", wordEmbedder.get_similarity('king', 'man'))
print("="*50)
print("10 most similar words to 'computer':", wordEmbedder.get_most_similar('computer'))
print("="*50)
sentence = "The queen rules the country."
print(f"Embed the sentence '{sentence}':", wordEmbedder.embed_document(sentence))