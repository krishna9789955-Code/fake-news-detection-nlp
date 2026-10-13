# Machine Learning Models Documentation

## 1. Serialized Model Artifacts

| File | Description | Tool / Format |
| :--- | :--- | :--- |
| `fake_news_model.joblib` | Primary trained classifier (L2-Regularized Logistic Regression) | `joblib` |
| `tfidf_vectorizer.joblib` | Trained TF-IDF feature extraction pipeline (5,000 features) | `joblib` |
| `metadata.json` | Training timestamp, vocabulary size, split counts, test metrics, and confusion matrix | `json` |
| `model_comparison.json` | Benchmark metrics comparing Logistic Regression, Naive Bayes, and Linear SVM | `json` |

---

## 2. Multi-Model Benchmark Comparison (Trained on original_data)

All models were evaluated on the **held-out test split of 7,757 articles** (Stratified 20% of 38,783 curated articles from `original_data/True.csv` and `original_data/Fake.csv`, `random_state=42`):

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **98.25%** | **98.31%** | **98.17%** | **98.23%** | ~0.20 s |
| **Linear SVM (SGD log loss)** | 97.60% | 97.63% | 97.52% | 97.58% | ~0.23 s |
| **Multinomial Naive Bayes** | 93.53% | 93.68% | 93.28% | 93.48% | ~0.03 s |

### Why Logistic Regression is the Primary Model:
1. **Highest Accuracy & Balanced F1:** 98.25% Accuracy and 98.23% F1-score across 7,757 test articles.
2. **Calibrated Probabilities:** Produces genuine posterior class probabilities via the logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
3. **Local Feature Attribution:** The model's linear coefficient vector ($\beta$) allows direct explainability ($w_i \cdot x_i$) showing exactly which words pushed toward Real News or Fake News.
4. **Generalization:** L2 Ridge regularization prevents overfitting on rare tokens.

---

## 3. Confusion Matrix Breakdown (Test Set, N = 7,757)

| Metric | Predicted Real | Predicted Fake | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Real News** | **4,145** (True Real) | **34** (False Fake) | 4,179 |
| **Actual Fake News** | **102** (False Real) | **3,476** (True Fake) | 3,578 |

- **True Positive / True Negative Rate:** 98.25%
- **Total Classification Errors:** 136 out of 7,757 articles (1.75% error rate)

---

## 4. Retraining the Models
To retrain and regenerate all model artifacts:
```bash
python train_model.py
```
To run independent diagnostic validation:
```bash
python evaluate_model.py
```
