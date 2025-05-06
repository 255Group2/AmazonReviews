#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

nltk_sw = ['d', 'm', 'o', 's', 't', 'y', 'll', 're', 've', 'ma', "that'll", 'ain', "she's", "it's", "you're", "you've", "you'll", "youd", 'isn', "isn't", 'aren', "aren't", 'wasn', "wasn't", 'weren', "weren't", 'don', "don't", 'doesn', "doesn't", 'didn', "didn't", 'hasn', "hasn't", 'haven', "haven't", 'hadn', "hadn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", "shouldve", 'won', "won't", 'wouldn', "wouldn't", 'couldn', "couldn't", 'i', 'me', 'my', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'himself', 'herself', 'itself', 'myself', 'yourself', 'yourselves', 'ourselves', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'had', 'has', 'have', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'again', 'further', 'then', 'once', 'can', 'will', 'just', 'should', 'now']
added_sw = ["he's", "he'd", "she'd", "he'll", "she'll", "you'll", "they'd", "could've", "would've", 'could', 'would', "i'm", 'im', "thatll", "shes", "youre", "youve", "youll", "youd", "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt", "hasnt", "havent", "hadnt", "mightnt", "mustnt", "neednt", "shant", "shouldnt", "shouldve", "wont", "wouldnt", "couldnt", 'a','b','c','e','f','g','h','i','j','k','l','n','p','q','r','u','v','w','x','z','lol']
stop_words = added_sw + nltk_sw
punc = ''',.;:?!'\"()[]{}<>|\\/@#^&*_~=+\\n\\t'''


# In[32]:


import json
def load_data(file_path, chunk_size=100000, target_rows=288000):
    try:
        print(f"Attempting to load JSON file from: {file_path}")
        all_data = []
        total_rows = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    all_data.append(json.loads(line.strip()))
                    total_rows += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line {i + 1}: {e}")
                    continue
        
        print(f"Total rows loaded: {total_rows}")
        
        # Randomly sample to target_rows to approximate 5M non-zero values
        if total_rows > target_rows:
            all_data = random.sample(all_data, target_rows)
            print(f"Sampled down to {target_rows} rows to target 5M non-zero values")
        else:
            print("Dataset smaller than target, using all rows")
        
        df = pd.DataFrame(all_data)
        # print("JSON file loaded successfully. Columns:", df.columns.tolist())
        
        # print("\nFirst 100 rows of the DataFrame:")
        # print(df.head(100))
        
        if 'overall' not in df.columns:
            raise KeyError("'overall' column not found in the dataset")
        if 'reviewText' not in df.columns:
            raise KeyError("'reviewText' column not found in the dataset")
        if 'asin' not in df.columns:
            raise KeyError("'asin' column not found in the dataset")

        label_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
       
        df['label'] = df['overall'].map(label_map)
        
        if df['label'].isnull().any():
            raise ValueError("Some 'overall' values couldn't be mapped to labels. Invalid 'overall' values found.")

        df['title'] = df.get('reviewerName', '').astype(str)
        df['text'] = df['reviewText'].astype(str).fillna('')
        df['asin'] = df['asin'].astype(str)
        
        split_result = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        if len(split_result) != 2:
            raise ValueError(f"train_test_split returned {len(split_result)} values, expected 2")
        train_df, test_df = split_result
        
        print("Dataset loaded successfully!")
        print(f"Training dataset shape: {train_df.shape}")
        print(f"Test dataset shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return None, None


# In[59]:


def preprocess_text(text, title=None):
    if not isinstance(text, str):
        text = str(text)
    
    if title is not None and isinstance(title, str):
        text = title + ' ' + text
    
    negation_phrases = {
        r'\bno\s+good\b': 'no_good',
        r'\bnot\s+good\b': 'not_good',
        r'\bnot\s+funny\b': 'not_funny',
        r'\bnot\s+worth\b': 'not_worth',
        r'\bnot\s+bad\b': 'not_bad',
        r'\bnot\s+great\b': 'not_great',
        r'\bnot\s+a\s+fan\b': 'not_a_fan',
        r'\bnot\s+impressed\b': 'not_impressed',
        r'\bnot\s+satisfied\b': 'not_satisfied',
        r'\bwaste\s+of\s+money\b': 'waste_of_money',
    }
    
    for pattern, replacement in negation_phrases.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Step 2: Handle contractions (convert to underscores)
    contraction_map = {
        "won't": "will_not", "can't": "can_not", "n't": "_not",
        "'re": "_are", "'s": "_is", "'d": "_would", 
        "'ll": "_will", "'ve": "_have", "'m": "_am"
    }
    for contraction, expansion in contraction_map.items():
        text = text.replace(contraction, expansion)
    
    # Step 3: Clean unwanted elements
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\d+', '', text)
    text = re.sub(f'[{re.escape(punc)}]', '', text)  # Remove punctuation
    
    # Step 4: Additional negation handling for remaining cases
    text = re.sub(r'\b(not|no)\s+(\w+)', r'\1_\2', text.lower())
    
    # Step 5: Tokenize and filter
    tokens = word_tokenize(text)
    tokens = [
        t for t in tokens 
        if t not in stop_words 
        and len(t) > 1 
        and not t.isnumeric()
    ]
    
    # Step 6: Reconstruct without splitting negation tokens
    return ' '.join(tokens)  # Keep underscores in tokens like "no_good"


# In[60]:


# Vectorize text using TF-IDF
def vectorize_text(train_text, test_text):
    try:
        print("Vectorizing text...")
        # important_phrases = ['no_good', 'not_good', 'not_funny', 'not_worth', 'not_work', 'not_recommend']
        # vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), vocabulary=important_phrases + None)

        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words=None)
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)
        total_non_zeros = X_train.nnz + X_test.nnz
        print(f"Total non-zero values in TF-IDF matrices: {total_non_zeros}")
        return X_train, X_test, vectorizer
    except Exception as e:
        print(f"Error in vectorize_text: {e}")
        return None, None, None


# ###Predict review

# In[61]:


def predict_review(review, title, model, vectorizer):
    review = str(review) if review is not None else ''
    processed_text = preprocess_text(review, title=title)
    X_new = vectorizer.transform([processed_text])
    prediction = model.predict(X_new)[0]
    
    # Get confidence score
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X_new)[0][1] if prediction == 1 else model.predict_proba(X_new)[0][0]
    
    sentiment = "Good (Positive)" if prediction == 1 else "Bad (Negative)"
    result = f"Predicted result: {sentiment}"
    if confidence is not None:
        result += f"\nConfidence Score: {confidence:.4f}"
    
    return result, prediction, confidence


# In[65]:


if __name__ == "__main__":
    json_path = '/Users/richardph911/Desktop/Office_Products.json'
    train_df, test_df = load_data(json_path)
    if train_df is not None and test_df is not None:
        train_df['reviews'] = train_df['title'].astype(str).fillna('') + ' ' + train_df['text'].astype(str).fillna('')
        test_df['reviews'] = test_df['title'].astype(str).fillna('') + ' ' + test_df['text'].astype(str).fillna('')
        train_df['text'] = train_df['reviews'].apply(preprocess_text)
        test_df['text'] = test_df['reviews'].apply(preprocess_text)
        train_df = train_df.drop(columns=['reviews'])
        test_df = test_df.drop(columns=['reviews'])
        # Vectorize vector
        X_train_full, X_test, vectorizer = vectorize_text(train_df['text'], test_df['text'])
        y_train_full = train_df['label'].values
        y_test = test_df['label'].values
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        accuracies = []
        model_names = list(models.keys())
        trained_models = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            y_test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_test_pred)
            accuracies.append(accuracy)
            print(f"{name} Test Performance:")
            print(classification_report(y_test, y_test_pred))

        best_model_name = model_names[np.argmax(accuracies)]
        best_model = trained_models[best_model_name]
        best_accuracy = max(accuracies)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=model_names, y=accuracies, palette='viridis')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Comparison of Different Models', fontsize=14)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black',
                        xytext=(0, 6), textcoords='offset points')
        plt.tight_layout()
        plt.savefig('model_comparison.png')

        from collections import defaultdict
        grouped_reviews = defaultdict(lambda: {"positive": [], "negative": [], "confidence_scores": []})

        reviews_to_predict = test_df.to_dict('records')
        for review_data in reviews_to_predict:
            try:
                asin = review_data["asin"]
                title = review_data["title"]
                text = review_data["text"]
                result, prediction, confidence = predict_review(text, title, best_model, vectorizer)
                sentiment = "positive" if prediction == 1 else "negative"
                review_info = {
                    "title": title,
                    "text": text,
                    "result": result,
                    "confidence": confidence
                }
                grouped_reviews[asin][sentiment].append(review_info)
                grouped_reviews[asin]["confidence_scores"].append(confidence)
            except Exception as e:
                print(f"Error processing review for ASIN {asin}: {e}")

        for asin, data in grouped_reviews.items():
            print(f"\nProduct ASIN: {asin}")
            positive_count = len(data["positive"])
            negative_count = len(data["negative"])
            print(f"Positive Reviews: {positive_count}")
            print(f"Negative Reviews: {negative_count}")
            print(f"Best Model Accuracy: {best_accuracy:.4f}")
            confidence_scores = [f"{score:.4f}" for score in data["confidence_scores"]]
            print(f"Review Confidence Scores: {', '.join(confidence_scores)}")
            print("-" * 50)
    else:
        print("Failed to load data")


# In[ ]:





# In[63]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Assuming you have these defined:
# model = your trained model
# vectorizer = your trained TfidfVectorizer
# accuracy = your model's accuracy

# Create a package with everything needed
model_package = {
    'model': best_model,
    'vectorizer': vectorizer,
    'metadata': {
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classes': ['negative', 'positive']  # Update with your classes
    }
}

# Save to single file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)


# In[64]:


test_cases = [
    "no good",
    "not good",
    "not worth it",
    "terrible experience",
    "absolutely amazing",
    "very good",
    "not bad",
    "not great",
    "not a fan",
    "waste of money"
]

for text in test_cases:
    result, _, _ = predict_review(text, "", model, vectorizer)
    print(f"Input: '{text}' --> {result}")


# In[ ]:




