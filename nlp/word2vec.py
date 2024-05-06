import nltk
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
import string

import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK data 
nltk.download('movie_reviews')

# Load the movie reviews from NLTK's IMDb dataset
reviews = [movie_reviews.words(fileid) for fileid in movie_reviews.fileids()]

# Preprocess the text
stop_words = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)
filtered_reviews = []

for review in reviews:
    words = [word.lower() for word in review if word.lower() not in stop_words and word.lower() not in punctuation]
    filtered_reviews.append(words)

# Train Word2Vec model
model = Word2Vec(filtered_reviews, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector for a specific word (example)
word = "movie"
word_vector = model.wv[word]

print(f"Word Vector for '{word}':\n{word_vector}")



# Sample data for demonstration
texts = ["this movie is great", "awesome film", "worst movie ever", "I enjoyed it a lot"]

# Corresponding labels for each text
labels = ["positive", "positive", "negative", "positive"]

# Tokenize the texts and train Word2Vec model
tokenized_texts = [text.split() for text in texts]
word2vec_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Convert texts to sequences of word indices
word_index = {word: index + 1 for index, word in enumerate(word2vec_model.wv.index_to_key)}
X = [[word_index[word] for word in seq] for seq in tokenized_texts]
X = pad_sequences(X, maxlen=max(len(seq) for seq in tokenized_texts), padding='post')

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Sample input text for testing
input_text = "I loved this movie!"

tokenized_input = input_text.split()

# Convert the tokenized input text to word indices, handling out-of-vocabulary words
input_indices = [word_index[word] if word in word_index else 0 for word in tokenized_input]

# Pad the input sequence to match the maximum sequence length
padded_input = pad_sequences([input_indices], maxlen=X.shape[1], padding='post')

# Make predictions using the trained model
prediction = model.predict(padded_input)
predicted_label = "positive" if prediction[0][0] > 0.5 else "negative"

print(f"Input Text: {input_text}")
print(f"Predicted Label: {predicted_label}")


