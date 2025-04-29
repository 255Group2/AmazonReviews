#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# In[ ]:


# Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, names=['label', 'title', 'text'])
        print("Dataset loaded successfully!")
        df['label'] = df['label'].map({1: 0, 2: 1})  # 1->0 (negative), 2->1 (positive)
        X = df['text'].astype(str).values
        y = df['label'].values
        return df, X, y
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None


# In[ ]:


# Text preprocessing
def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, flags=re.IGNORECASE)
    # Tokenize
    tokens = text.lower().split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# In[ ]:





# In[ ]:




