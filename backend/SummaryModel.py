import sys
sys.path.append('../src')
import pickle
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import math
import joblib

class ImprovedReviewSummarizer:
    """
    Enhanced ML-based review summarizer with improved phrase extraction and coherent summary generation
    """
    
    def __init__(self, model_path="./models/improved_model.pkl"):
        self.model_path = model_path
        
        # Initialize core components
        self.idf = {}
        self.stemmer = PorterStemmer()
        
        # Load NLTK resources
        self._load_nltk_resources()
        
        # Initialize ML models with improved parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # Increased features
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,  # Better filtering
            max_df=0.9,
            preprocessor=self._preprocess_text
        )
        
        self.lda_model = LatentDirichletAllocation(n_components=8, random_state=42)
        self.kmeans_model = KMeans(n_clusters=6, random_state=42)
        
        # Initialize sentiment analyzer
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        
        # Expanded and better organized sentiment indicators
        self.negative_indicators = {
            'quality': ['poor quality', 'low quality', 'cheap', 'flimsy', 'break easily', 
                       'badly made', 'poor construction', 'feels cheap', 'not durable'],
            'performance': ['slow', 'buggy', 'crashes', 'not responsive', 'poor performance', 
                          'doesn\'t work', 'stopped working', 'battery drain', 'overheats'],
            'usability': ['hard to use', 'difficult', 'complicated', 'confusing interface', 
                         'poor design', 'not intuitive', 'uncomfortable'],
            'value': ['overpriced', 'not worth', 'expensive', 'waste of money', 'too costly', 
                     'better alternatives'],
            'service': ['poor support', 'bad customer service', 'unhelpful staff', 'no response', 
                       'terrible warranty', 'bad service'],
            'reliability': ['unreliable', 'inconsistent', 'malfunction', 'defective', 
                          'keeps breaking', 'technical issues']
        }
        
        self.positive_indicators = {
            'quality': ['high quality', 'excellent build', 'well made', 'durable', 'premium feel', 
                       'solid construction', 'good materials', 'sturdy'],
            'performance': ['fast', 'efficient', 'works great', 'responsive', 'reliable performance', 
                          'exceeds expectations', 'flawless', 'powerful'],
            'usability': ['easy to use', 'user friendly', 'intuitive', 'simple setup', 'plug and play', 
                         'comfortable', 'ergonomic', 'convenient'],
            'value': ['great value', 'worth the price', 'affordable', 'good deal', 'reasonably priced', 
                     'value for money', 'budget friendly'],
            'design': ['sleek design', 'beautiful', 'modern look', 'attractive', 'elegant', 
                      'stylish', 'aesthetic'],
            'satisfaction': ['love it', 'highly recommend', 'exceeded expectations', 'impressed', 
                           'perfect', 'exactly what wanted', 'amazing'],
            'service': ['excellent support', 'helpful customer service', 'quick response', 
                       'friendly staff', 'great warranty', 'professional service']
        }
        
        # Phrase templates for better sentence construction
        self.phrase_templates = {
            'positive': [
                "Users praise the {category}, particularly the {aspect}",
                "Many customers appreciate the {category}, especially the {aspect}",
                "The {category} receives positive feedback, with emphasis on {aspect}"
            ],
            'negative': [
                "Common complaints focus on {category}, particularly the {aspect}",
                "Users report issues with {category}, especially regarding {aspect}",
                "Several reviews mention concerns about {category}, particularly {aspect}"
            ]
        }
    
    def _load_nltk_resources(self):
        """Load required NLTK resources"""
        resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'vader_lexicon']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def _preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not", "'ll": " will",
            "'ve": " have", "'re": " are", "'d": " would", "'m": " am",
            "don't": "do not", "doesn't": "does not", "didn't": "did not"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs, special characters but keep useful punctuation
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s\.,!?-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, dataframe):
        """Train the model with improved feature extraction"""
        print("Training Enhanced ML Review Summarizer...")
        
        # Preprocess all reviews
        all_reviews = []
        vocabulary = set()
        doc_f = defaultdict(lambda: 0)
        
        for i, row in dataframe.iterrows():
            # Preprocess the text
            preprocessed_text = self._preprocess_text(row['all_reviews'])
            all_reviews.append(preprocessed_text)
            
            # Tokenize and count for IDF
            tokens = word_tokenize(preprocessed_text)
            vocabulary.update(tokens)
            
            unique_words = set(tokens)
            for word in unique_words:
                doc_f[word] += 1
        
        # Calculate IDF
        DOC_COUNT = len(dataframe)
        for word in vocabulary:
            self.idf[word] = math.log10(DOC_COUNT / float(doc_f[word]))
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(all_reviews)
        
        print(f"Model trained on {DOC_COUNT} documents")
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"IDF dictionary size: {len(self.idf)}")
        
        return self
    
    def extract_meaningful_phrases(self, reviews):
        """Extract coherent and meaningful phrases from reviews"""
        phrase_scores = []
        
        for review_idx, review in enumerate(reviews):
            preprocessed_review = self._preprocess_text(review)
            sentences = sent_tokenize(preprocessed_review)
            
            for sentence in sentences:
                # Get sentence sentiment
                sent_score = self.sia.polarity_scores(sentence)
                
                # Extract meaningful phrases (1-4 words)
                words = word_tokenize(sentence)
                
                # Get phrases of different lengths
                for n in range(1, 5):
                    for i in range(len(words) - n + 1):
                        phrase = ' '.join(words[i:i+n])
                        
                        # Skip if contains too many common words
                        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 
                                       'for', 'of', 'as', 'by', 'that', 'this', 'it', 'is', 'are', 
                                       'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}
                        
                        phrase_words = phrase.split()
                        common_count = sum(1 for word in phrase_words if word in common_words)
                        
                        if common_count / len(phrase_words) > 0.5:
                            continue
                        
                        # Calculate phrase score
                        phrase_score = 0
                        for word in phrase_words:
                            if word in self.idf:
                                phrase_score += self.idf[word]
                        
                        phrase_score /= len(phrase_words)
                        
                        # Add sentiment boost
                        phrase_score *= (1 + abs(sent_score['compound']))
                        
                        phrase_scores.append((phrase, phrase_score, review_idx, sent_score['compound']))
        
        return phrase_scores
    
    def categorize_phrases(self, phrases):
        """Categorize phrases into sentiment categories with better organization"""
        categorized = {
            'positive': defaultdict(list),
            'negative': defaultdict(list)
        }
        
        for phrase, score, review_idx, sentiment in phrases:
            # Check negative indicators first
            for category, indicators in self.negative_indicators.items():
                for indicator in indicators:
                    if indicator in phrase and sentiment < 0:
                        categorized['negative'][category].append((phrase, score, sentiment))
                        break
            else:
                # Check positive indicators
                for category, indicators in self.positive_indicators.items():
                    for indicator in indicators:
                        if indicator in phrase and sentiment > 0:
                            categorized['positive'][category].append((phrase, score, sentiment))
                            break
        
        return categorized
    
    def clean_and_deduplicate_phrases(self, phrases):
        """Clean and deduplicate phrases to avoid repetition"""
        cleaned_phrases = []
        seen_content = set()
        
        # Sort by score
        phrases.sort(key=lambda x: x[1], reverse=True)
        
        for phrase, score, sentiment in phrases:
            # Clean the phrase
            cleaned = re.sub(r'\s+', ' ', phrase).strip()
            
            # Skip if too short
            if len(cleaned) < 3:
                continue
            
            # Check for semantic similarity with already added phrases
            is_similar = False
            for seen_phrase in seen_content:
                # Check word overlap
                phrase_words = set(cleaned.split())
                seen_words = set(seen_phrase.split())
                
                overlap = len(phrase_words.intersection(seen_words))
                if overlap / min(len(phrase_words), len(seen_words)) > 0.7:
                    is_similar = True
                    break
            
            if not is_similar:
                cleaned_phrases.append(cleaned)
                seen_content.add(cleaned)
        
        return cleaned_phrases[:5]  # Return top 5 unique phrases
    
    def build_coherent_summary(self, categorized_phrases, overall_sentiment):
        """Build a coherent summary using sentence templates"""
        summary_parts = []
        
        # Determine sentiment distribution
        positive_count = sum(len(phrases) for phrases in categorized_phrases['positive'].values())
        negative_count = sum(len(phrases) for phrases in categorized_phrases['negative'].values())
        total_feedback = positive_count + negative_count
        
        if total_feedback == 0:
            return "No significant feedback found in reviews."
        
        positive_ratio = positive_count / total_feedback
        
        # Add opening sentence
        if positive_ratio > 0.7:
            summary_parts.append("This product receives overwhelmingly positive feedback.")
        elif positive_ratio < 0.3:
            summary_parts.append("Customer reviews highlight significant concerns with this product.")
        else:
            summary_parts.append("Customer feedback presents a balanced view of this product.")
        
        # Add positive aspects
        if categorized_phrases['positive']:
            pos_sentences = []
            for category, phrases in categorized_phrases['positive'].items():
                if phrases:
                    cleaned_phrases = self.clean_and_deduplicate_phrases(phrases)
                    if cleaned_phrases:
                        template = random.choice(self.phrase_templates['positive'])
                        sentence = template.format(category=category, aspect=', '.join(cleaned_phrases[:2]))
                        pos_sentences.append(sentence)
            
            if pos_sentences:
                summary_parts.extend(pos_sentences[:2])
        
        # Add negative aspects
        if categorized_phrases['negative']:
            neg_sentences = []
            for category, phrases in categorized_phrases['negative'].items():
                if phrases:
                    cleaned_phrases = self.clean_and_deduplicate_phrases(phrases)
                    if cleaned_phrases:
                        template = random.choice(self.phrase_templates['negative'])
                        sentence = template.format(category=category, aspect=', '.join(cleaned_phrases[:2]))
                        neg_sentences.append(sentence)
            
            if neg_sentences:
                if positive_ratio > 0.5:
                    summary_parts.append("However, " + neg_sentences[0][0].lower() + neg_sentences[0][1:])
                else:
                    summary_parts.extend(neg_sentences[:2])
        
        # Add conclusion
        if 0.4 <= positive_ratio <= 0.6:
            summary_parts.append("Overall, potential buyers should carefully weigh these mixed reviews against their specific needs.")
        
        return ' '.join(summary_parts)
    
    def generate_summary(self, reviews):
        """Generate comprehensive and coherent summary"""
        if not reviews:
            return "No reviews available."
        
        # Extract meaningful phrases
        phrases = self.extract_meaningful_phrases(reviews)
        
        # Categorize phrases
        categorized_phrases = self.categorize_phrases(phrases)
        
        # Calculate overall sentiment
        review_sentiments = []
        for review in reviews:
            sentiment_score = self.sia.polarity_scores(review)['compound']
            review_sentiments.append(sentiment_score)
        
        overall_sentiment = np.mean(review_sentiments)
        
        # Build coherent summary
        summary = self.build_coherent_summary(categorized_phrases, overall_sentiment)
        
        return summary
    
    def save(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_path
        
        model_data = {
            'idf': self.idf,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'negative_indicators': self.negative_indicators,
            'positive_indicators': self.positive_indicators,
            'phrase_templates': self.phrase_templates
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path="./models/improved_model.pkl"):
        """Load the trained model"""
        model = cls(model_path=path)
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            model.idf = model_data['idf']
            model.tfidf_vectorizer = model_data['tfidf_vectorizer']
            model.negative_indicators = model_data['negative_indicators']
            model.positive_indicators = model_data['positive_indicators']
            model.phrase_templates = model_data.get('phrase_templates', model.phrase_templates)
            
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating and training new model instance...")
            
            import pandas as pd
            data_file = "data/asin_numreviews_allreview.csv"
            df = pd.read_csv(data_file)
            model.train(df)
            model.save(path)
            print(f"Newly trained model saved to {path}")
        
        return model

__all__ = ['ImprovedReviewSummarizer']