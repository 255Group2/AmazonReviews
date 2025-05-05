import pickle

SENTIMENT_MODEL_PATH = "./models/best_sentiment_model.pkl"
SUMMARY_MODEL_PATH = "./improved_model.pkl"
from SummaryModel import ImprovedReviewSummarizer

def load_summary_model():
    try:
        # Ensure the class is available before loading
        print(f"Loading model from {SUMMARY_MODEL_PATH}")
        
        # Use the class method to load
        model_bundle = ImprovedReviewSummarizer.load("improved_model.pkl")
        print("Model loaded successfully")
        return model_bundle
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model instance")
        # Fallback to creating a new instance
        return ImprovedReviewSummarizer()

# Load sentiment analysis model
def load_sentiment_model():
    with open(SENTIMENT_MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
    return model_bundle["model"], model_bundle["vectorizer"]

def preprocess_text(text):
    # Add any preprocessing used during training, e.g., lowercasing, cleaning
    return text.lower()

def predict_review(text, title, model, vectorizer):
    combined = preprocess_text(title + ' ' + text)
    try:
        processed = preprocess_text(text)
    
        # Transform using vectorizer
        features = vectorizer.transform([processed])
        
        # Get prediction and probability
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Convert to sentiment label
        result = "Positive" if prediction == 1 else "Negative"
        return result, prediction, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None
