# News Article Classification (Django + Machine Learning API)

This project implements a complete News Article Classification system using Natural Language Processing (NLP), Machine Learning, Django, and Django REST Framework.  
The system predicts the category of a news article based on its headline, short description, and keywords.  
It includes a trained ML model and a Django API to serve real-time predictions.

---

## Features

### Machine Learning
- Text preprocessing (cleaning, stopword removal, stemming, lemmatization)
- TF-IDF vectorization for feature extraction
- Handling class imbalance using SMOTE
- Model training using Logistic Regression, Naive Bayes, and SVM
- Evaluation using accuracy, precision, recall, and F1-score

### Django REST API
- `/api/predict/` endpoint for predictions
- DRF serializers for input validation
- Integration of ML model, TF-IDF vectorizers, and preprocessing pipeline
- JSON-based request and response format

---

## Project Structure

newsproject/
newsproject/
settings.py
urls.py
newsapi/
serializers.py
views.py
ml/
model.pkl
vectorizers.pkl
preprocessor.pkl
text_preprocessor.py
predict.py
README.md
requirements.txt


## Installation

### 1. Clone the repository

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>


### 2. Create a virtual environment


python -m venv env
env\Scripts\activate     # Windows
source env/bin/activate  # Linux/Mac


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply database migrations

python manage.py migrate

### 5. Start the server

python manage.py runserver

Server runs at:

```
http://127.0.0.1:8000/
```

## API Usage

### Endpoint

```
POST /api/predict/
```

### Request Body (JSON)

```json
{
    "headline": "Stock Market Surges",
    "short_description": "Tech stocks lead strong rebound",
    "keywords": "market, finance, stocks"
}
```

### Response Format (JSON)

```json
{
    "predicted_category": "BUSINESS",
    "confidence_scores": {
        "BUSINESS": 0.73,
        "TECH": 0.14,
        "WORLD NEWS": 0.06
    }
}
```

---

## Machine Learning Pipeline

### Text Preprocessing

The preprocessing includes:

* Removing HTML tags, URLs, punctuation, and digits
* Converting text to lowercase
* Tokenization
* Stopword removal
* Stemming using PorterStemmer
* Lemmatization

### Feature Engineering

TF-IDF vectorizers are trained separately on:

* headline
* short_description
* keywords

The resulting matrices are combined using `scipy.sparse.hstack`.

### Model Training

Multiple algorithms were trained and evaluated.
Logistic Regression performed best based on overall metrics.

### Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) was used to balance the dataset.

---

## Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC curves

Plots were generated using Matplotlib and Seaborn.

---

## Testing the API

You can test the API using Thunder Client, Postman, or cURL.

Example request:

POST http://127.0.0.1:8000/api/predict/


## Technologies Used

* Python
* Django
* Django REST Framework
* scikit-learn
* imbalanced-learn
* NLTK
* pandas
* numpy
* matplotlib
* seaborn

<img width="1468" height="369" alt="Screenshot 2025-11-16 033731" src="https://github.com/user-attachments/assets/077e06da-22aa-4394-8140-13219c5fca56" />

<img width="1152" height="400" alt="image" src="https://github.com/user-attachments/assets/d4105360-be6c-49a4-bb64-19907da78f17" />

