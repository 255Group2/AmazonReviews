# app.py
import joblib
import streamlit as st
from utils import preprocess_text

# Function to load the model package
def load_model_package(path="best_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model package: {e}")
        return None

# Load it
model_package = load_model_package()

if model_package is None:
    st.stop()

model = model_package['model']
vectorizer = model_package['vectorizer']
metadata = model_package.get('metadata', {})

# Optional debug
st.write("Model metadata:", metadata)

# Prediction
user_input = st.text_input("Enter a review:")
if user_input:
    processed = preprocess_text(user_input)
    X_new = vectorizer.transform([processed])
    prediction = model.predict(X_new)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
    st.write("Processed:", processed)
    st.write("Prediction:", sentiment)

# import streamlit as st
# import pickle
# import pandas as pd
# from datetime import datetime
# from utils import preprocess_text
# # Cache the model loading for better performance
# @st.cache_resource
# def load_model_package():
#     try:
#         with open('best_model.pkl', 'rb') as f:
#             package = pickle.load(f)
#             return package
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# # Load the model package
# model_package = load_model_package()

# if model_package is None:
#     st.stop()  # Stop the app if model failed to load

# # Extract components
# model = model_package['model']
# vectorizer = model_package['vectorizer']
# metadata = model_package.get('metadata', {})

# # Streamlit UI
# st.title("Sentiment Analysis")
# st.write("Analyze text for positive/negative sentiment")

# # Display model metadata
# with st.expander("Model Info"):
#     st.write(f"**Model type:** {type(model).__name__}")
#     st.write(f"**Created on:** {metadata.get('created_at', 'Unknown')}")
#     st.write(f"**Classes:** {', '.join(metadata.get('classes', []))}")

# # User input
# text_input = st.text_area(
#     "Enter text to analyze:", 
#     "This product works great and I'm very satisfied!",
#     height=150
# )

# if st.button("Analyze", type="primary"):
#     if not text_input.strip():
#         st.warning("Please enter some text to analyze")
#     else:
#         with st.spinner("Processing..."):
#             try:
#                 # Vectorize the input
#                 processed = preprocess_text(text_input)
#                 X = vectorizer.transform([text_input])
                
#                 # Predict
#                 prediction = model.predict(X)[0]
#                 sentiment = metadata.get('classes', ['negative', 'positive'])[prediction]
                
#                 # Get probabilities if available
#                 if hasattr(model, 'predict_proba'):
#                     proba = model.predict_proba(X)[0]
#                     confidence = proba[prediction]
#                 else:
#                     proba = None
#                     confidence = "N/A"
                
#                 # Display results
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Prediction")
#                     if sentiment.lower() == 'positive':
#                         st.success(f"👍 {sentiment}")
#                     else:
#                         st.error(f"👎 {sentiment}")
                
#                 with col2:
#                     st.subheader("Confidence")
#                     if proba is not None:
#                         st.progress(confidence)
#                         st.write(f"{confidence*100:.1f}% confident")
#                     else:
#                         st.info("Confidence scores not available")
                
#                 # Show detailed probabilities if available
#                 if proba is not None:
#                     with st.expander("Detailed probabilities"):
#                         prob_df = pd.DataFrame({
#                             'Class': metadata.get('classes', ['negative', 'positive']),
#                             'Probability': proba
#                         })
#                         st.bar_chart(prob_df.set_index('Class'))
                        
#             except Exception as e:
#                 st.error(f"Prediction error: {str(e)}")

# # Example texts
# st.subheader("Try these examples:")
# examples = [
#     "This is the best product I've ever bought!",
#     "Terrible experience, would never buy again.",
#     "It's okay but could be better.",
#     "The service was good but delivery was late."
# ]

# cols = st.columns(len(examples))
# for i, example in enumerate(examples):
#     if cols[i].button(example, key=f"ex_{i}"):
#         st.session_state.text_input = example