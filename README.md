# News Article Classification Model

This project builds a **text classification model** that automatically predicts the **category of a news article** (for example: politics, sports, technology, or business) based on its textual content.  
It demonstrates the complete workflow — from **data preprocessing** and **feature extraction** to **model training**, **evaluation**, and **visualization**.

---

## Project Overview

The aim of this project is to apply **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify news articles efficiently and accurately.

### Key Objectives
- Clean and preprocess raw text data  
- Extract features using **TF-IDF Vectorization**  
- Handle class imbalance using **SMOTE**  
- Train multiple ML models and compare performance  
- Evaluate models using precision, recall, F1-score, and accuracy  
- Visualize confusion matrices and performance metrics

---

## Concepts Covered
- Text preprocessing (tokenization, stopword removal, stemming, lemmatization)
- Feature engineering with TF-IDF
- Model selection and hyperparameter tuning using GridSearchCV
- Handling class imbalance with SMOTE
- Model evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)

---

## Libraries Used

- **pandas** — Data manipulation  
- **numpy** — Numerical computation  
- **matplotlib**, **seaborn** — Data visualization  
- **nltk** — Natural Language Toolkit for text preprocessing  
- **scikit-learn** — ML algorithms and evaluation metrics  
- **imbalanced-learn (imblearn)** — SMOTE oversampling  
- **tqdm** — Progress bar for loops  

---

## Project Workflow

1. **Data Collection**  
   Load and inspect the dataset of news articles.

2. **Data Preprocessing**  
   - Remove special characters, digits, and punctuation  
   - Convert text to lowercase  
   - Tokenize and remove stopwords  
   - Apply stemming or lemmatization

3. **Feature Extraction**  
   Convert processed text into numerical vectors using **TF-IDF Vectorizer**.

4. **Model Training**  
   Train multiple algorithms such as:
   - Logistic Regression  
   - Naive Bayes  
   - Support Vector Machine (SVM)

5. **Handling Imbalance**  
   Use **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes.

6. **Evaluation**  
   Evaluate models using:
   - `classification_report`
   - `confusion_matrix`
   - `accuracy_score`
   - ROC and AUC metrics

7. **Visualization**  
   Plot confusion matrices and ROC curves using `matplotlib` and `seaborn`.


## Installation & Usage

### Clone the Repository
```bash
git clone https://github.com/alenpeter/news-article-classifier.git
cd news-article-classifier
