import re
import unicodedata
from collections import Counter

def preprocess_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def simple_tokenize(text):
    return text.split()

class SimpleVocab:
    def __init__(self, texts=None, min_freq=2):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        
        if texts:
            word_counts = Counter()
            for text in texts:
                tokens = simple_tokenize(text)
                word_counts.update(tokens)
            
            idx = 2
            for word, count in word_counts.items():
                if count >= min_freq:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
        
        self.size = len(self.word2idx)
    
    def text_to_sequence(self, text):
        tokens = simple_tokenize(text)
        return [self.word2idx.get(word, 1) for word in tokens]
    
    def pad_sequence(self, sequence, max_len):
        return sequence[:max_len] if len(sequence) >= max_len else sequence + [0] * (max_len - len(sequence))