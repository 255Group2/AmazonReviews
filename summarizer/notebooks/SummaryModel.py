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

class PolishedMRLSummarizer:
    """
    Complete ML-based review summarizer with preprocessing, IDF usage, and comprehensive sentiment analysis
    """
    
    def __init__(self, model_path="polished_ml_model.pkl"):
        self.model_path = model_path
        
        # Initialize core components
        self.idf = {}
        self.stemmer = PorterStemmer()
        
        # Load NLTK resources
        self._load_nltk_resources()
        
        # Initialize ML models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            preprocessor=self._preprocess_text
        )
        
        self.lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        self.kmeans_model = KMeans(n_clusters=4, random_state=42)
        
        # Initialize sentiment analyzer
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        
        # Comprehensive sentiment indicators (expanded)
        self.negative_indicators = {
            'quality': ['cheap', 'poor', 'bad', 'terrible', 'awful', 'low quality', 'flimsy', 'fragile', 
                       'bad quality', 'feels cheap', 'cheaply made', 'poor quality', 'shoddy', 'poor construction'],
            'performance': ['slow', 'sluggish', 'crash', 'crashes', 'frozen', 'hangs', 'hang', 'bug', 'bugs', 'buggy', 
                          'glitch', 'glitchy', 'lag', 'lags', 'delayed', 'unresponsive', 'non-responsive', 'failed', 'fails'],
            'battery': ['drain', 'drains', 'draining', 'battery life', 'battery issues', 'battery problem', 'short battery', 
                       'battery dead', 'battery died', 'battery drains', 'battery poor', 'battery depletes'],
            'service': ['poor service', 'unhelpful', 'bad support', 'useless', 'rude', 'terrible support', 
                       'no help', 'waste of time', 'no response', 'customer service', 'awful service'],
            'usability': ['difficult', 'hard', 'frustrating', 'annoying', 'complicated', 'confusing', 'hard to use', 
                         'difficult to use', 'not intuitive', 'user unfriendly', 'complex', 'clunky'],
            'defects': ['broken', 'defective', 'stopped working', 'doesn\'t work', 'not working', 'malfunction', 
                       'issue', 'issues', 'problem', 'problems', 'fault', 'faulty', 'dead on arrival'],
            'design': ['ugly', 'bulky', 'heavy', 'poorly designed', 'bad design', 'awkward', 'uncomfortable', 
                      'hard to hold', 'too big', 'too small', 'doesn\'t fit']
        }
        
        self.positive_indicators = {
            'quality': ['excellent', 'great', 'premium', 'high quality', 'well made', 'solid', 'durable', 
                       'quality build', 'well built', 'sturdy', 'robust', 'reliable', 'high quality', 'premium quality'],
            'performance': ['fast', 'quick', 'smooth', 'efficient', 'responsive', 'reliable', 'works great', 
                          'performs well', 'excellent performance', 'speedy', 'powerful', 'consistent', 'stable'],
            'usability': ['easy', 'simple', 'intuitive', 'user-friendly', 'straightforward', 'easy to use', 
                         'simple to use', 'hassle-free', 'user friendly', 'plug and play', 'easy setup', 'quick setup'],
            'design': ['beautiful', 'sleek', 'modern', 'stylish', 'attractive', 'nice design', 'well designed', 
                      'elegant', 'sophisticated', 'good looking', 'aesthetic', 'pretty', 'gorgeous'],
            'value': ['worth', 'value', 'good deal', 'affordable', 'great value', 'good price', 'reasonably priced', 
                     'money well spent', 'value for money', 'worth the price', 'budget friendly', 'economical'],
            'battery': ['battery life', 'great battery', 'excellent battery', 'long battery', 'battery lasts', 
                       'good battery', 'amazing battery', 'battery excellent', 'all day battery'],
            'service': ['helpful', 'excellent support', 'great service', 'friendly', 'quick response', 'responsive', 
                       'good customer service', 'helpful support', 'supportive', 'good support'],
            'satisfaction': ['love', 'amazing', 'fantastic', 'wonderful', 'perfect', 'exceeded expectations', 
                           'impressed', 'happy', 'satisfied', 'pleased']
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
        """Comprehensive text preprocessing"""
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
        
        # Remove URLs, special characters but keep useful punctuation for context
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s\.,!?-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, dataframe):
        """Train the model by computing IDF values and fitting TF-IDF"""
        print("Training Complete ML Review Summarizer...")
        
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
    
    def extract_phrases_with_tfidf(self, reviews):
        """Extract phrases using TF-IDF scores"""
        # Preprocess reviews
        preprocessed_reviews = [self._preprocess_text(review) for review in reviews]
        
        # Transform with fitted TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(preprocessed_reviews)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for each phrase
        phrase_scores = []
        for i, review in enumerate(preprocessed_reviews):
            # Get TF-IDF scores for this review
            review_scores = tfidf_matrix[i].toarray()[0]
            
            # Extract phrases with high scores
            for j, score in enumerate(review_scores):
                if score > 0.1:  # Threshold for important phrases
                    phrase = feature_names[j]
                    # Check if phrase appears in review
                    if phrase in review:
                        phrase_scores.append((phrase, score, i))
        
        # Sort by TF-IDF score
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        
        return phrase_scores
    
    def extract_sentiment_phrases(self, reviews):
        """Extract phrases with sentiment context using IDF weights"""
        all_sentiment_phrases = {
            'positive': defaultdict(list),
            'negative': defaultdict(list)
        }
        
        # Process each review
        for review_idx, review in enumerate(reviews):
            preprocessed_review = self._preprocess_text(review)
            sentences = sent_tokenize(preprocessed_review)
            
            for sentence in sentences:
                # Get sentence sentiment
                sent_sentiment = self.sia.polarity_scores(sentence)['compound']
                
                # Extract words and their IDF
                words = word_tokenize(sentence)
                word_importance = []
                
                for word in words:
                    if word in self.idf:
                        importance = self.idf[word]
                        word_importance.append((word, importance))
                
                # Sort words by IDF importance
                word_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Extract meaningful phrases from sentence
                for i in range(len(words) - 1):
                    # Get 2-3 word combinations
                    for n in range(2, 4):
                        if i + n <= len(words):
                            phrase = ' '.join(words[i:i+n])
                            
                            # Only keep phrases with important words (high IDF)
                            phrase_words = word_tokenize(phrase)
                            avg_idf = sum(self.idf.get(w, 0) for w in phrase_words) / len(phrase_words)
                            
                            if avg_idf > 0.5:  # Threshold for importance
                                # Categorize phrase
                                for category, indicators in self.negative_indicators.items():
                                    if any(indicator in phrase for indicator in indicators):
                                        all_sentiment_phrases['negative'][category].append((phrase, sent_sentiment, review_idx))
                                        break
                                else:
                                    for category, indicators in self.positive_indicators.items():
                                        if any(indicator in phrase for indicator in indicators):
                                            all_sentiment_phrases['positive'][category].append((phrase, sent_sentiment, review_idx))
                                            break
        
        return all_sentiment_phrases
    
    def clean_and_rank_phrases(self, phrases):
        """Clean, deduplicate and rank phrases"""
        cleaned_phrases = []
        seen_words = set()
        
        # Sort by TF-IDF score (if available) or sentiment intensity
        phrases.sort(key=lambda x: x[1] if isinstance(x[1], float) else abs(x[1]), reverse=True)
        
        for phrase_data in phrases:
            phrase = phrase_data[0]
            
            # Clean the phrase
            cleaned = re.sub(r'\s+', ' ', phrase).strip()
            
            # Skip if too short or purely numeric
            if len(cleaned) < 4 or cleaned.isdigit():
                continue
            
            # Check for redundancy
            phrase_words = set(word_tokenize(cleaned.lower()))
            if not phrase_words.issubset(seen_words):
                cleaned_phrases.append(cleaned)
                seen_words.update(phrase_words)
        
        return cleaned_phrases[:5]  # Return top 5 unique phrases
    
    def generate_summary(self, reviews):
        """Generate comprehensive and balanced summary using all ML features"""
        if not reviews:
            return "No reviews available."
        
        # Extract TF-IDF weighted phrases
        tfidf_phrases = self.extract_phrases_with_tfidf(reviews)
        
        # Extract sentiment-categorized phrases with IDF weighting
        sentiment_phrases = self.extract_sentiment_phrases(reviews)
        
        # Calculate overall sentiment
        review_sentiments = []
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            sentiment_score = self.sia.polarity_scores(review)['compound']
            review_sentiments.append(sentiment_score)
            
            if sentiment_score > 0.05:
                positive_count += 1
            elif sentiment_score < -0.05:
                negative_count += 1
        
        overall_sentiment = np.mean(review_sentiments)
        
        # Determine review type
        if positive_count / len(reviews) > 0.7:
            review_type = 'mostly_positive'
        elif negative_count / len(reviews) > 0.7:
            review_type = 'mostly_negative'
        else:
            review_type = 'mixed'
        
        # Build summary parts
        summary_parts = []
        
        # Add opening sentence
        openers = {
            'mostly_positive': "Customers overwhelmingly praise this product",
            'mostly_negative': "Reviews highlight significant concerns about this product",
            'mixed': "Customer feedback is mixed for this product"
        }
        summary_parts.append(openers[review_type] + ",")
        
        # Add top positive aspects
        if sentiment_phrases['positive']:
            positive_sentences = []
            for category, category_phrases in sentiment_phrases['positive'].items():
                cleaned_phrases = self.clean_and_rank_phrases(category_phrases)
                if cleaned_phrases:
                    if category == 'satisfaction':
                        positive_sentences.append(f"with many reporting {', '.join(cleaned_phrases[:3])}")
                    else:
                        positive_sentences.append(f"noting the {category}: {', '.join(cleaned_phrases[:3])}")
            
            if positive_sentences:
                summary_parts.append(' '.join(positive_sentences[:2]) + ".")
        
        # Add negative aspects
        if sentiment_phrases['negative']:
            negative_sentences = []
            for category, category_phrases in sentiment_phrases['negative'].items():
                cleaned_phrases = self.clean_and_rank_phrases(category_phrases)
                if cleaned_phrases:
                    negative_sentences.append(f"{category} concerns including {', '.join(cleaned_phrases[:3])}")
            
            if negative_sentences:
                summary_parts.append(f"However, some users mention {'; '.join(negative_sentences[:2])}.")
        
        # Add top TF-IDF phrases for additional context
        top_tfidf_phrases = []
        for phrase, score, _ in tfidf_phrases[:5]:
            if len(phrase) > 3 and not any(phrase in sp for sp in summary_parts):
                top_tfidf_phrases.append(phrase)
        
        if top_tfidf_phrases:
            summary_parts.append(f"Key themes include {', '.join(top_tfidf_phrases[:3])}.")
        
        # Add balanced conclusion
        if review_type == 'mixed':
            conclusions = [
                "Overall, potential buyers should weigh these strengths against the mentioned concerns.",
                "The product shows promise but may not suit all user needs.",
                "Consider these mixed experiences when making a purchase decision."
            ]
            summary_parts.append(random.choice(conclusions))
        
        # Join all parts
        final_summary = ' '.join(summary_parts)
        
        return final_summary
    
    def save(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_path
        
        model_data = {
            'idf': self.idf,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'negative_indicators': self.negative_indicators,
            'positive_indicators': self.positive_indicators
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path="polished_ml_model.pkl"):
        """Load the trained model"""
        model = cls(model_path=path)
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            model.idf = model_data['idf']
            model.tfidf_vectorizer = model_data['tfidf_vectorizer']
            model.negative_indicators = model_data['negative_indicators']
            model.positive_indicators = model_data['positive_indicators']
            
            print(f"Model loaded from {path}")
            print(f"IDF dictionary size: {len(model.idf)}")
        except FileNotFoundError:
            print("No saved model found. Using default initialization.")
        
        return model

__all__ = ['PolishedMRLSummarizer']
