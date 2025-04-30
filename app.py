import streamlit as st
import torch
from utils import preprocess_text, SimpleVocab
import torch.nn as nn
import torch.nn.functional as F

# Configuration
MODEL_PATH = "Binary_Classification_PyTorch_CNN.pth"
VOCAB_SIZE = 43793
EMBED_SIZE = 300
MAX_SEQUENCE_LENGTH = 220

# Define the model architecture (must match training)
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128 * 26, 256)  # Calculated based on MAX_SEQUENCE_LENGTH
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(VOCAB_SIZE, EMBED_SIZE, MAX_SEQUENCE_LENGTH).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def predict(text, model, vocab, device):
    processed = preprocess_text(text)
    sequence = vocab.text_to_sequence(processed)
    padded = vocab.pad_sequence(sequence, MAX_SEQUENCE_LENGTH)
    tensor = torch.tensor([padded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        pred = output.item()
        return {
            'sentiment': 'Positive' if pred > 0.5 else 'Negative',
            'confidence': round((pred if pred > 0.5 else 1 - pred) * 100, 2),
            'score': pred
        }

# UI Setup
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis")

# Load resources
model = load_model()
device = next(model.parameters()).device
vocab = SimpleVocab([], min_freq=2)  # Should load your actual vocab

# Input/Output
text = st.text_area("Enter text:", height=150, key="input_text")

if st.button("Analyze"):
    if text.strip():
        result = predict(text, model, vocab, device)
        if result['sentiment'] == "Positive":
            st.success(f"Positive ({result['confidence']}% confidence)")
        else:
            st.error(f"Negative ({result['confidence']}% confidence)")
        st.progress(result['confidence'] / 100)
    else:
        st.warning("Please enter some text")