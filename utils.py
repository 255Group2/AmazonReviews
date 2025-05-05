
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