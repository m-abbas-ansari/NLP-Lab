from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Input text for which you want to generate embeddings
input_text = "I love natural language processing."

# Tokenize the input text
tokenized_text = tokenizer.tokenize(input_text)
tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert token IDs to tensor
input_tensor = torch.tensor([input_ids])

# Get the BERT model output
with torch.no_grad():
    outputs = model(input_tensor)

# Extract the embeddings from the BERT model outputs
hidden_states = outputs[2]
word_embeddings = hidden_states[-1]  # Last layer hidden states for each token

# Average the token embeddings to get the sentence embedding
sentence_embedding = torch.mean(word_embeddings, dim=1).squeeze()

print("Word Embeddings:")
print(word_embeddings)
print("Sentence Embedding:")
print(sentence_embedding)


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split

# Sample data for text classification
texts = ["I love natural language processing.", "This movie is great!", "I don't like this product.", "The weather today is nice."]
labels = [1, 1, 0, 1]  # Binary labels (1: positive, 0: negative)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 classes for binary classification

# Tokenize the input texts
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Split data into training and testing sets
input_ids_train, input_ids_test, labels_train, labels_test = \
    train_test_split(tokenized_texts['input_ids'], labels, test_size=0.2, random_state=42)
attention_masks_train, attention_masks_test = \
    train_test_split(tokenized_texts['attention_mask'], test_size=0.2, random_state=42)

# Create DataLoader for training and testing data
train_data = TensorDataset(input_ids_train, attention_masks_train, torch.tensor(labels_train))
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

test_data = TensorDataset(input_ids_test, attention_masks_test, torch.tensor(labels_test))
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

# Set optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
model.train()
for epoch in range(3):  # Example: 3 epochs
    total_loss = 0
    for batch in train_loader:
        input_ids_batch, attention_masks_batch, labels_batch = batch
        optimizer.zero_grad()
        outputs = model(input_ids_batch, attention_mask=attention_masks_batch, labels=labels_batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}")

# Evaluation on test data
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids_batch, attention_masks_batch, labels_batch = batch
        outputs = model(input_ids_batch, attention_mask=attention_masks_batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(labels_batch.tolist())

# Calculate accuracy
accuracy = torch.sum(torch.tensor(predictions) == torch.tensor(true_labels)).item() / len(true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# Sample input text for testing
input_text = "I love this movie!"

# Tokenize the input text
tokenized_input = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# Make predictions using the trained model
model.eval()
with torch.no_grad():
    outputs = model(**tokenized_input)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

predicted_label = "positive" if prediction == 1 else "negative"

print(f"Input Text: {input_text}")

print(f"Predicted Label: {predicted_label}")
