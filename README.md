# Fake News Detection Using NLP

**Internship Domain:** Data Science & Machine Learning  
**Project Type:** Interactive Web Application & NLP Classification System  
**System Title:** TruthLens.NLP  

---

## 1. Project Abstract & Methodology

In an era of rapid information dissemination across social media and digital networks, automated misinformation detection has become critical. This project presents **TruthLens.NLP**, an end-to-end Machine Learning and Natural Language Processing application designed to classify news articles and headlines into **Likely Real News** and **Likely Fake News**.

### Core Methodology
1. **Curated & Leakage-Free Dataset:** Directly processes the CSV files in `original_data/` (`Fake.csv` and `True.csv` totaling 44,898 raw records, 38,783 curated articles). Crucially, the pipeline filters out publisher datelines (such as wire stamps `[CITY] (Reuters) -`) and deduplicates articles, preventing classifiers from exploiting trivial metadata shortcuts.
2. **Text Preprocessing & Feature Extraction:** Raw text is normalized, lowercased, stripped of URLs/HTML tags, filtered for English stop words, and vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** with 5,000 features and sublinear term-frequency scaling.
3. **Multi-Model Benchmark & Stratified Validation:** Three candidate classifiers were evaluated on an 80/20 stratified split:
   - **Logistic Regression (Primary Model):** **98.25% Test Accuracy**, **98.23% F1-Score**, with naturally calibrated class probabilities.
   - **Linear Support Vector Classifier (SGD log loss):** 97.60% Test Accuracy.
   - **Multinomial Naive Bayes (Baseline):** 93.53% Test Accuracy.
4. **Local Feature Attribution (Explainable AI):** Calculates the product of document TF-IDF values and model coefficients ($w_i \cdot x_i$) to highlight specific tokens influencing the prediction.
5. **Interactive Web Interface:** A responsive Flask application featuring a Home Dashboard, Real-Time Detection Console, Model Analytics Dashboard (powered by Chart.js), and SQLite prediction audit logs.

> **Responsible AI Notice:** Predictions are probabilistic estimates based on writing style, rhetoric, and vocabulary patterns. The system does not cross-reference live external fact-checking databases.

---

## 2. Technology Stack

- **Frontend:** HTML5, CSS3 (Modern Dark Theme), JavaScript (ES6+), Bootstrap 5.3, Chart.js 4.4, Font Awesome 6.
- **Backend:** Python 3.11 with Flask 3.1.
- **Machine Learning & NLP:** Scikit-Learn 1.9, TF-IDF Vectorizer, NLTK-compatible text cleaning.
- **Data Processing:** Python standard `csv`, `numpy`.
- **Model Storage & Serialization:** Joblib.
- **Database / Logging:** SQLite 3 (`database/predictions.db`).
- **Testing:** Python `unittest` framework (21 automated tests).

---

## 3. Project Directory Structure

```
FakeNewsDetection_2/
├── app.py                     # Flask web application and REST API endpoints
├── train_model.py             # Model training, feature extraction, and serialization pipeline
├── evaluate_model.py          # Independent evaluation and data leakage verification script
├── config.py                  # Global application configuration and paths
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive documentation and viva guide
│
├── original_data/
│   ├── Fake.csv               # 23,481 raw fake news articles (62.8 MB)
│   └── True.csv               # 21,417 raw real news articles (53.6 MB)
│
├── data/
│   ├── README.md              # Dataset schema and leakage prevention notes
│   └── news_dataset.csv       # Synchronized 38,783 curated articles (Real & Fake)
│
├── models/
│   ├── README.md              # Model performance comparison and confusion matrix
│   ├── fake_news_model.joblib # Serialized Logistic Regression model
│   ├── tfidf_vectorizer.joblib# Serialized TF-IDF vectorizer (5,000 features)
│   ├── metadata.json          # Actual test metrics and vocabulary statistics
│   └── model_comparison.json  # Comparative metrics across 3 classifiers
│
├── database/
│   ├── db.py                  # SQLite database interface for prediction history
│   └── predictions.db         # Local SQLite database (auto-generated)
│
├── utils/
│   ├── __init__.py            # Package initializer
│   ├── preprocessor.py        # Text cleaning, shortcut stripping, and linguistic statistics
│   └── explainer.py           # Local feature attribution (token explainability)
│
├── templates/
│   ├── base.html              # Base layout with navbar, footer, and styling
│   ├── index.html             # Landing page with summary metrics and workflow
│   ├── detect.html            # Interactive detection page with sample buttons
│   ├── dashboard.html         # Performance analytics and Chart.js visualizations
│   └── about.html             # Detailed ML workflow and viva questions guide
│
├── static/
│   ├── css/
│   │   └── style.css          # Custom dark navy/charcoal responsive theme
│   └── js/
│       ├── app.js             # Global application utilities
│       ├── detect.js          # Detection console frontend controller
│       └── dashboard.js       # Chart.js charts and history table controller
│
└── tests/
    ├── __init__.py            # Test package initializer
    ├── test_app.py            # Unit tests for Flask endpoints and API validation
    └── test_model.py          # Unit tests for preprocessing, models, and explainability
```

---

## 4. Setup & Execution Instructions

### Step 1: Clone or Navigate to the Workspace
```bash
cd "d:\Intern pro\FakeNewsDetection_2"
```

### Step 2: Set Up Python Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Command Prompt:
.venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Models
*Note: Pre-trained models are already included in `models/`. To retrain from scratch:*
```bash
python train_model.py
```
*Output Summary:*
```
[*] Training samples: 31,026, Test samples: 7,757
  > Logistic Regression       | Acc: 98.25% | F1: 98.23%
  > Linear SVM                | Acc: 97.60% | F1: 97.58%
  > Multinomial Naive Bayes   | Acc: 93.53% | F1: 93.48%
[*] Saved artifacts to models/
```

### Step 5: Run Independent Model Evaluation & Leakage Audit
```bash
python evaluate_model.py
```

### Step 6: Run Unit Tests
```bash
python -m unittest discover tests
```
*Result: 21 tests pass with 0 errors.*

### Step 7: Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
**http://127.0.0.1:5000**

---

## 5. API Reference

### 1. `GET /api/health`
Checks server status and model availability.
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "primary_model": "Logistic Regression",
    "dataset_samples": 5764,
    "test_accuracy": 97.75
  }
  ```

### 2. `POST /api/predict`
Analyzes news text and returns prediction and explainability.
- **Request Body:**
  ```json
  {
    "headline": "Senate passes landmark clean energy legislation on Wednesday",
    "article": "Lawmakers finalized negotiations following bipartisan hearings..."
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "prediction": "Likely Real News",
    "predicted_class": 0,
    "confidence": 98.4,
    "probabilities": { "real": 98.4, "fake": 1.6 },
    "analysis_summary": "The text exhibits formal journalistic attribution...",
    "explainability": {
      "real_indicators": [{ "token": "legislation", "weight": 0.42 }],
      "fake_indicators": []
    },
    "linguistics": { "word_count": 82, "lexical_diversity": 74.2 },
    "disclaimer": "Notice: This prediction is a statistical estimate..."
  }
  ```

### 3. `GET /api/metrics`
Returns dataset metrics and classifier benchmark comparisons.

### 4. `GET /api/history`
Returns logged predictions from SQLite with optional `?filter=real|fake` and `?search=keyword`.

### 5. `POST /api/history/clear`
Clears stored prediction logs.

---

## 6. Viva-Voce & Evaluation Q&A

**Q1: What is the significance of removing publisher datelines like `(Reuters)`?**  
*Answer:* In raw Kaggle datasets, almost 99% of real articles start with `[CITY] (Reuters) -`. Without preprocessing, the model simply memorizes the string "reuters" rather than analyzing article content. By systematically stripping publisher signatures, we ensure the model learns true journalistic syntax and sensationalism cues.

**Q2: What is the mathematical basis of TF-IDF?**  
*Answer:* $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$. Sublinear scaling ($1 + \log(\text{TF})$) dampens repeated buzzwords, ensuring balanced feature representation.

**Q3: How does the system explain its predictions?**  
*Answer:* We compute local linear feature attributions ($w_i \cdot x_i$) where $x_i$ is the document's TF-IDF weight and $w_i$ is the model's learned coefficient. Positive weights indicate words associated with misinformation, while negative weights indicate journalistic style.

**Q4: Why was Logistic Regression selected over Naive Bayes?**  
*Answer:* Logistic Regression achieved superior accuracy and F1-score (97.75% vs 94.19%), avoids the conditional independence assumption of Naive Bayes, and yields well-calibrated probabilities via the sigmoid link function.

---

## 7. License & Academic Attribution
Developed as an educational Data Science internship capstone project. Open for demonstration, learning, and academic evaluations.
