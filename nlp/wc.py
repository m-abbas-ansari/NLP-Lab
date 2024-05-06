import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Sample corpus
corpus = """
    This is a sample corpus for generating a word context matrix. 
    The purpose is to demonstrate how to create a matrix based on word context in a text.
    The matrix will show the frequency of words within a specified window size.
"""

# Tokenize the corpus
tokens = word_tokenize(corpus.lower())

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [
    token for token in tokens if token.isalnum() and token not in stop_words]

# Define window size
window_size = 2

# Create word-context matrix
word_context_matrix = {}

for i in range(len(filtered_tokens)):
    word = filtered_tokens[i]
    if word not in word_context_matrix:
        word_context_matrix[word] = {}

    for j in range(max(0, i - window_size), min(len(filtered_tokens), i + window_size + 1)):
        if i != j:
            context_word = filtered_tokens[j]
            if context_word not in word_context_matrix[word]:
                word_context_matrix[word][context_word] = 1
            else:
                word_context_matrix[word][context_word] += 1

# Convert word-context matrix to a numpy matrix
words = list(word_context_matrix.keys())
word_context_array = np.zeros((len(words), len(words)), dtype=int)

for i, word in enumerate(words):
    for j, context_word in enumerate(words):
        if context_word in word_context_matrix[word]:
            word_context_array[i, j] = word_context_matrix[word][context_word]

print("Word-Context Matrix:")
print("    " + " ".join(words))
for i, row in enumerate(word_context_array):
    print(words[i].ljust(10) + " ".join(map(lambda x: str(x).rjust(5), row)))

# find similarity between words using cosine similarity without using any library


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


print("\nCosine Similarity Matrix:")
print("    " + " ".join(words))
for i, row in enumerate(word_context_array):
    similarity_row = []
    for j in range(len(words)):
        similarity = cosine_similarity(
            word_context_array[i], word_context_array[j])
        similarity_row.append(f"{similarity:.2f}".rjust(5))
    print(words[i].ljust(10) + " ".join(similarity_row))
