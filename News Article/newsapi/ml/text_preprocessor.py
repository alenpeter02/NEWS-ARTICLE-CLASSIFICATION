import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        
    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'(\w)-(\w)', r'\1 \2', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    def tokenize_text(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]
    
    def apply_stemming(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]
    
    def apply_lemmatization(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def preprocess_pipeline(self, text, use_stemming=True, use_lemmatization=True):
        text = self.clean_text(text)
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.apply_stemming(tokens)
        if use_lemmatization:
            tokens = self.apply_lemmatization(tokens)
        return ' '.join(tokens)
