import streamlit as st
import torch
from utils import preprocess_text, SimpleVocab, TextCNN

# Constants
MODEL_PATH = "Binary_Classification_PyTorch_CNN.pth"
VOCAB_SIZE = 20000
EMBED_SIZE = 300
MAX_SEQUENCE_LENGTH = 220

# App config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(VOCAB_SIZE, EMBED_SIZE, MAX_SEQUENCE_LENGTH).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def predict(text):
    processed = preprocess_text(text)
    sequence = vocab.text_to_sequence(processed)
    padded = vocab.pad_sequence(sequence, MAX_SEQUENCE_LENGTH)
    tensor = torch.tensor([padded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        pred = output.item()
        sentiment = "Positive" if pred > 0.5 else "Negative"
        confidence = pred if sentiment == "Positive" else 1 - pred
    
    return {
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2),
        'score': float(pred)
    }

# Load model and vocab
model = load_model()
device = next(model.parameters()).device
vocab = SimpleVocab([], min_freq=2)

# UI
st.title("Sentiment Analysis")
text = st.text_area("Enter text:", height=150)

if st.button("Analyze"):
    if text.strip():
        result = predict(text)
        if result['sentiment'] == "Positive":
            st.success(f"😊 Positive ({result['confidence']}%)")
        else:
            st.error(f"😞 Negative ({result['confidence']}%)")
        st.progress(result['confidence'] / 100)
    else:
        st.warning("Please enter some text")

if st.button("Example"):
    st.session_state.example = "This watch is 5/5. I love it. It is very comfortable to wear"
    st.experimental_rerun()

if 'example' in st.session_state:
    text = st.session_state.example