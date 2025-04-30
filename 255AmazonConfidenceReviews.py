
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import re
import unicodedata
import time
import datetime
from tqdm import tqdm
from collections import Counter
import csv

# Fix random seed for reproducibility
seed = 3541
torch.manual_seed(seed)
np.random.seed(seed)

# Helper functions
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:
        return 'Date today: %s' % datetime.date.today()

# Simplified text preprocessing functions
def preprocess_text(text):
    # Handle None or non-string values
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags (simplified)
    text = re.sub(r'<.*?>', '', text)
    
    # Replace newlines, tabs with spaces
    text = re.sub(r'[\n\t\r]', ' ', text)
    
    # Remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def simple_tokenize(text):
    """Split text into words by spaces"""
    return text.split()

class SimpleVocab:
    """Simple vocabulary builder"""
    def __init__(self, texts, min_freq=2):
        print(f"Building vocabulary with minimum frequency {min_freq}...")
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = simple_tokenize(text)
            word_counts.update(tokens)
        
        # Add words that meet the minimum frequency
        idx = 2  # Start after special tokens
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
        self.size = len(self.word2idx)
        print(f"Vocabulary built with {self.size} unique words")
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        tokens = simple_tokenize(text)
        return [self.word2idx.get(word, 1) for word in tokens]  # 1 is <unk>
    
    def pad_sequence(self, sequence, max_len):
        """Pad or truncate sequence to max_len"""
        if len(sequence) < max_len:
            return sequence + [0] * (max_len - len(sequence))  # 0 is <pad>
        else:
            return sequence[:max_len]

# 1. Load dataset with custom CSV parsing
print(f"===== STARTING BINARY CLASSIFICATION MODEL =====")
print(f"{date_time(1)}")
print("Loading datasets...")

def load_csv_no_headers(file_path, num_samples=None):
    """
    Load CSV without headers, where:
    - Column 0: class label
    - Column 1: review title (optional - we'll merge with review text)
    - Column 2: review text
    """
    labels = []
    texts = []
    
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if num_samples is not None and i >= num_samples:
                break
                
            if len(row) >= 3:  # Ensure we have at least 3 columns
                # Use the label from first column
                labels.append(row[0])
                
                # Combine title and text (columns 1 and 2)
                title = row[1] if len(row) > 1 else ""
                text = row[2] if len(row) > 2 else ""
                full_text = title + " " + text
                texts.append(full_text)
            elif len(row) == 2:  # If only 2 columns
                labels.append(row[0])
                texts.append(row[1])
    
    print(f"Loaded {len(texts)} examples from {file_path}")
    return labels, texts


# Update these paths to match your actual files
# train_path = '/Users/shan/Desktop/cmpe255/train.csv'
# test_path = '/Users/shan/Desktop/cmpe255/test.csv'
train_path = '/Users/richardph911/Downloads/archive/train.csv' 
test_path = '/Users/richardph911/Downloads/archive/test.csv' 

# Load the data
print(f"Loading training data from {train_path}...")
y_train, X_train = load_csv_no_headers(train_path, num_samples=100000)
print(f"Loading test data from {test_path}...")
y_test, X_test = load_csv_no_headers(test_path, num_samples=50000)

# Use a portion of training data as validation
val_size = 50000
print(f"Splitting {val_size} examples for validation...")
y_val = y_train[:val_size]
X_val = X_train[:val_size]
y_train = y_train[val_size:val_size+100000]
X_train = X_train[val_size:val_size+100000]

print(f"Final data split:")
print(f"  - Training: {len(X_train)} examples")
print(f"  - Validation: {len(X_val)} examples")
print(f"  - Testing: {len(X_test)} examples")

# Show a sample of the data to verify
print("\nSample text from dataset:")
print(X_train[0][:200] + "..." if len(str(X_train[0])) > 200 else X_train[0])
print("\nSample label from dataset:")
print(y_train[0])

# 2. Preprocess text
print("\nPreprocessing text...")
print("Processing training texts...")
X_train_processed = [preprocess_text(text) for text in tqdm(X_train)]
print("Processing validation texts...")
X_val_processed = [preprocess_text(text) for text in tqdm(X_val)]
print("Processing test texts...")
X_test_processed = [preprocess_text(text) for text in tqdm(X_test)]
print("Text preprocessing complete")

# 3. Build vocabulary and convert to sequences
print("\nBuilding vocabulary...")
MAX_SEQUENCE_LENGTH = 220
EMBED_SIZE = 300
BATCH_SIZE = 128

# Build vocabulary from training data
vocab = SimpleVocab(X_train_processed, min_freq=2)
print(f"Vocabulary size = {vocab.size}")

# Convert texts to sequences
print("Converting texts to sequences...")
print("Processing training sequences...")
X_train_seq = [vocab.text_to_sequence(text) for text in X_train_processed]
print("Processing validation sequences...")
X_val_seq = [vocab.text_to_sequence(text) for text in X_val_processed]
print("Processing test sequences...")
X_test_seq = [vocab.text_to_sequence(text) for text in X_test_processed]

# Pad sequences
print("Padding sequences to length:", MAX_SEQUENCE_LENGTH)
X_train_padded = [vocab.pad_sequence(seq, MAX_SEQUENCE_LENGTH) for seq in X_train_seq]
X_val_padded = [vocab.pad_sequence(seq, MAX_SEQUENCE_LENGTH) for seq in X_val_seq]
X_test_padded = [vocab.pad_sequence(seq, MAX_SEQUENCE_LENGTH) for seq in X_test_seq]
print("Sequence padding complete")

# Convert to PyTorch tensors
print("Converting to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_padded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_padded, dtype=torch.long)

# Encode labels
print("Encoding labels...")
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

y_train_tensor = torch.tensor(y_train, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

# Create dataset and dataloader
print("Creating PyTorch datasets and dataloaders...")
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("Data preparation complete")

# 4. Print sequence length statistics instead of visualizing
print("\nAnalyzing sequence lengths:")
train_lens = [len(s) for s in X_train_seq]
train_avg_len = sum(train_lens) / len(train_lens)
train_max_len = max(train_lens)
train_min_len = min(train_lens)
print(f"Training data sequence lengths - Avg: {train_avg_len:.1f}, Min: {train_min_len}, Max: {train_max_len}")

test_lens = [len(s) for s in X_test_seq]
test_avg_len = sum(test_lens) / len(test_lens)
test_max_len = max(test_lens)
test_min_len = min(test_lens)
print(f"Test data sequence lengths - Avg: {test_avg_len:.1f}, Min: {test_min_len}, Max: {test_max_len}")

# 5. Define CNN model in PyTorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length, dropout_rate=0.1):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers with appropriate padding to maintain dimensions
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the correct size after convolutions and pooling
        # Each pooling layer halves the sequence length
        # Need to calculate exact dimensions to avoid shape mismatch
        # First, add padding calculation for convolutions
        length_after_conv1 = max_seq_length + 2*1 - 4 + 1  # padding=1, kernel=4
        length_after_pool1 = length_after_conv1 // 2
        
        length_after_conv2 = length_after_pool1 + 2*1 - 4 + 1
        length_after_pool2 = length_after_conv2 // 2
        
        length_after_conv3 = length_after_pool2 + 2*1 - 4 + 1
        length_after_pool3 = length_after_conv3 // 2
        
        # Final size is channels * final_length
        fc_input_size = 128 * length_after_pool3
        print(f"Calculated fc_input_size: {fc_input_size}")
        
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Embedding Layer
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # First Conv Block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second Conv Block
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third Conv Block
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Debug print to check the actual flattened size
        if not hasattr(self, 'printed_shape'):
            print(f"Actual flattened size: {x.size(1)}")
            self.printed_shape = True
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return torch.sigmoid(x)

# 6. Initialize model, loss function, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TextCNN(vocab.size, EMBED_SIZE, MAX_SEQUENCE_LENGTH).to(device)
print("Model architecture:")
print(model)

# Binary Cross Entropy
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Training loop with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    counter = 0
    
    print(f"Starting training with {num_epochs} max epochs, patience={patience}")
    print(f"Training on {len(train_loader.dataset)} examples, validating on {len(val_loader.dataset)} examples")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        print(f"Training...")
        for inputs, labels in train_loader:
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"  Processing batch {batch_count}/{len(train_loader)}")
                
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        print(f"Validating...")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            counter += 1
            print(f"Validation loss did not improve. Counter: {counter}/{patience}")
            if counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses, train_accs, val_accs

print("\nStarting training...")
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer)

# 8. Print performance summary instead of plotting
print("\nTraining Performance Summary:")
best_train_loss = min(train_losses)
best_val_loss = min(val_losses)
best_train_acc = max(train_accs)
best_val_acc = max(val_accs)

print(f"Best Training Loss: {best_train_loss:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Training Accuracy: {best_train_acc:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# 9. Evaluate model on test set
def evaluate_model(model, test_loader):
    print(f"Evaluating model on {len(test_loader.dataset)} test examples...")
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    batch_count = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_count += 1
            if batch_count % 20 == 0:
                print(f"  Processing test batch {batch_count}/{len(test_loader)}")
                
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())
    
    # Calculate accuracyls

    all_preds = torch.cat(all_preds).squeeze()
    all_labels = torch.cat(all_labels).squeeze()
    
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")
    # accuracy = accuracy_score(all_labels, all_preds)
    # print(f"Test Accuracy: {accuracy:.4f}")
    
    # F1 score
    f1 = f1_score(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Confusion matrix (just print the values, no visualization)
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    return accuracy

print("\nEvaluating model on test set...")
test_accuracy = evaluate_model(model, test_loader)

# 10. Save model
model_filename = 'Binary_Classification_PyTorch_CNN.pth'
torch.save(model.state_dict(), model_filename)
print(f"\nModel saved as {model_filename}")
print(f"\n===== COMPLETED AT {date_time(1)} =====")



# Expected output with max-epoch = 1 rather than 30:  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
#               precision    recall  f1-score   support

#          0.0       0.49      1.00      0.66     24626
#          1.0       0.00      0.00      0.00     25374

#     accuracy                           0.49     50000
#    macro avg       0.25      0.50      0.33     50000
# weighted avg       0.24      0.49      0.33     50000


# Confusion Matrix:
# [[24626     0]
#  [25374     0]]

# ===== Example Predictions =====

# Review: Waste of time and money Pretty useless. Should more properly be titled "Great Family Resorts", as that is what it principly addresses. Has some information about seperate activities (at the resorts) f...
# Sentiment: Negative (Confidence: 0.00) 
# True Label: Negative
# --------------------------------------------------------------------------------

# Review: Mariah has proven herself to be an industry tramp....... 2005 has shown the true side of mariah, and has shown her greed and hatred for her fans. She knows that they will buy anything that has her nam...
# Sentiment: Negative (Confidence: 0.00)
# True Label: Negative
# --------------------------------------------------------------------------------

# Review: Total rip off The Three stooges( Collectors Edition)The box says "7DVDs over 11 hours" what they dont say is that there are only 4 classic stooges shorts and only on the first DVD. The rest of the set...
# Sentiment: Negative (Confidence: 0.00)
# True Label: Negative
# --------------------------------------------------------------------------------

# Review: Medwyn Goodall I find Medwyn Goodall's music mesmerizing! I LOVE his Druid, his Merlin series, Grail Quest, King Authur. Guinevere, Clan -- all of his Celtic CD's... I'm a writer of fantasy, and no ot...
# Sentiment: Negative (Confidence: 0.00)
# True Label: Positive
# --------------------------------------------------------------------------------

# Review: Intersex is NOT transgender The author of this book incorrectly includes infants born with genetic conditions under the transgender term. Intersex conditions are those in which an infant is born with ...
# Sentiment: Negative (Confidence: 0.00)
# True Label: Negative
# --------------------------------------------------------------------------------

# Model saved as Binary_Classification_PyTorch_CNN.pth

# ===== COMPLETED AT Timestamp: 2025-04-30 00:12:03 =====