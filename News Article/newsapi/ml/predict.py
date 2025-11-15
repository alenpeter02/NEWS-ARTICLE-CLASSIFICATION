import os
import pickle
from scipy.sparse import hstack
from .text_preprocessor import TextPreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, 'ml')

model = pickle.load(open(os.path.join(ML_DIR, 'model.pkl'), 'rb'))
vectorizers = pickle.load(open(os.path.join(ML_DIR, 'vectorizers.pkl'), 'rb'))
preprocessor = TextPreprocessor()

def preprocess_input(headline, description, keywords):
    h = preprocessor.preprocess_pipeline(headline)
    d = preprocessor.preprocess_pipeline(description)
    k = preprocessor.preprocess_pipeline(keywords)
    return h, d, k

def vectorize(h, d, k):
    h_vec = vectorizers["headline"].transform([h])
    d_vec = vectorizers["short_description"].transform([d])
    k_vec = vectorizers["keywords"].transform([k])
    return hstack([h_vec, d_vec, k_vec])

def predict_category(headline, description, keywords):
    h, d, k = preprocess_input(headline, description, keywords)
    features = vectorize(h, d, k)

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    return pred, dict(zip(model.classes_, probs))
