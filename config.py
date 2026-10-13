"""
Configuration settings for Fake News Detection Web Application.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Original Data directory and paths (Kaggle True & Fake CSVs)
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'original_data')
TRUE_CSV_PATH = os.path.join(ORIGINAL_DATA_DIR, 'True.csv')
FAKE_CSV_PATH = os.path.join(ORIGINAL_DATA_DIR, 'Fake.csv')

# Combined & curated dataset path
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_PATH = os.path.join(DATA_DIR, 'news_dataset.csv')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_model.joblib')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
METRICS_PATH = os.path.join(MODELS_DIR, 'metadata.json')
COMPARISON_PATH = os.path.join(MODELS_DIR, 'model_comparison.json')

# Database path
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'predictions.db')

# Application Settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'fake-news-detector-secret-key-2026')
MAX_INPUT_CHARS = 25000
MIN_INPUT_CHARS = 10
PORT = int(os.environ.get('PORT', 5000))
HOST = os.environ.get('HOST', '127.0.0.1')
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# Class mapping
# Label 0: Real News, Label 1: Fake News
CLASS_NAMES = {
    0: 'Likely Real News',
    1: 'Likely Fake News'
}

CLASS_SLUGS = {
    0: 'real',
    1: 'fake'
}
