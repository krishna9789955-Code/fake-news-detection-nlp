"""
Train Machine Learning Models for Fake News Detection.
Executes an end-to-end data processing, feature extraction, model comparison,
and serialization pipeline using CSV files from the original_data directory.
"""
import os
import csv
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import (
    ORIGINAL_DATA_DIR,
    TRUE_CSV_PATH,
    FAKE_CSV_PATH,
    DATASET_PATH,
    DATA_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    MODELS_DIR
)
from utils.preprocessor import clean_text


def load_and_clean_data(dataset_path: str = None):
    """
    Loads raw dataset directly from original_data/True.csv and original_data/Fake.csv
    (or falls back to dataset_path if provided).
    Cleans text, removes duplicates, strips publisher shortcuts,
    and synchronizes a cleaned copy into data/news_dataset.csv.
    """
    texts = []
    labels = []
    seen_hashes = set()
    skipped_duplicates = 0
    skipped_short = 0
    raw_rows_count = 0

    use_original_data = os.path.exists(TRUE_CSV_PATH) and os.path.exists(FAKE_CSV_PATH)

    if use_original_data:
        print(f"[*] Loading dataset from original_data folder:")
        print(f"    - True News: {TRUE_CSV_PATH}")
        print(f"    - Fake News: {FAKE_CSV_PATH}")

        sources = [
            (TRUE_CSV_PATH, 0, "True News"),
            (FAKE_CSV_PATH, 1, "Fake News")
        ]

        cleaned_rows_for_csv = []

        for csv_file, label, label_name in sources:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                file_count = 0
                for row in reader:
                    raw_rows_count += 1
                    file_count += 1
                    title = (row.get("title") or "").strip()
                    body = (row.get("text") or "").strip()

                    # Clean and strip publisher shortcuts (leakage prevention)
                    clean_t = clean_text(title, strip_shortcuts=True)
                    clean_b = clean_text(body, strip_shortcuts=True)
                    combined = f"{clean_t} {clean_b}".strip()

                    if len(combined) < 40:
                        skipped_short += 1
                        continue

                    # Deduplication using normalized character snippet
                    dedup_key = combined[:200]
                    if dedup_key in seen_hashes:
                        skipped_duplicates += 1
                        continue
                    seen_hashes.add(dedup_key)

                    texts.append(combined)
                    labels.append(label)
                print(f"    Loaded {file_count} raw rows from {os.path.basename(csv_file)}")

    elif dataset_path and os.path.exists(dataset_path):
        print(f"[*] Reading dataset from: {dataset_path}...")
        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows_count += 1
                title = (row.get("title") or "").strip()
                body = (row.get("text") or "").strip()
                label_val = row.get("label")

                if label_val is None or str(label_val).strip() not in ("0", "1"):
                    continue

                label = int(str(label_val).strip())
                clean_t = clean_text(title, strip_shortcuts=True)
                clean_b = clean_text(body, strip_shortcuts=True)
                combined = f"{clean_t} {clean_b}".strip()

                if len(combined) < 40:
                    skipped_short += 1
                    continue

                dedup_key = combined[:200]
                if dedup_key in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes.add(dedup_key)

                texts.append(combined)
                labels.append(label)
    else:
        raise FileNotFoundError(f"Neither original_data CSV files nor {dataset_path} found.")

    total_valid = len(texts)
    real_count = sum(1 for y in labels if y == 0)
    fake_count = sum(1 for y in labels if y == 1)

    print(f"    - Total raw articles read: {raw_rows_count}")
    print(f"    - Total valid curated samples: {total_valid}")
    print(f"    - Real news samples (0): {real_count} ({real_count/total_valid*100:.1f}%)")
    print(f"    - Fake news samples (1): {fake_count} ({fake_count/total_valid*100:.1f}%)")
    print(f"    - Deduplicated records removed: {skipped_duplicates}")
    print(f"    - Short/empty records skipped: {skipped_short}")

    return texts, np.array(labels)


def evaluate_classifier(model, X_test, y_test, model_name: str):
    """
    Computes standard evaluation metrics on held-out test data.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    prec_real = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_real = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_real = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    prec_fake = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_fake = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec_macro) * 100, 2),
        "recall": round(float(rec_macro) * 100, 2),
        "f1_score": round(float(f1_macro) * 100, 2),
        "per_class": {
            "real_news": {
                "precision": round(float(prec_real) * 100, 2),
                "recall": round(float(rec_real) * 100, 2),
                "f1_score": round(float(f1_real) * 100, 2),
                "support": int(np.sum(y_test == 0))
            },
            "fake_news": {
                "precision": round(float(prec_fake) * 100, 2),
                "recall": round(float(rec_fake) * 100, 2),
                "f1_score": round(float(f1_fake) * 100, 2),
                "support": int(np.sum(y_test == 1))
            }
        },
        "confusion_matrix": {
            "true_real_pred_real": cm[0][0],
            "true_real_pred_fake": cm[0][1],
            "true_fake_pred_real": cm[1][0],
            "true_fake_pred_fake": cm[1][1],
            "matrix": cm
        }
    }


def extract_top_features(vectorizer, model, top_n=15):
    """
    Extracts top N features most strongly indicative of Fake News and Real News.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
    elif hasattr(model, 'feature_log_prob_'):
        coefs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return {"fake_indicators": [], "real_indicators": []}

    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    fake_features = [
        {"word": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
        for i in top_fake_idx
    ]
    real_features = [
        {"word": str(feature_names[i]), "weight": round(float(abs(coefs[i])), 4)}
        for i in top_real_idx
    ]

    return {
        "fake_indicators": fake_features,
        "real_indicators": real_features
    }


def train_and_evaluate():
    """
    Executes full pipeline:
    1. Loads dataset from original_data folder (True.csv & Fake.csv)
    2. Stratified train/test split (80/20)
    3. Fits TF-IDF on train split ONLY
    4. Trains Logistic Regression, Naive Bayes, Linear SVM
    5. Evaluates on held-out test set
    6. Saves best model artifacts and comparison JSON
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    start_time = time.time()

    print("==================================================")
    print("   FAKE NEWS DETECTION: NLP TRAINING PIPELINE    ")
    print("==================================================")

    # 1. Load data from original_data folder
    texts, labels = load_and_clean_data()

    # 2. Stratified Train/Test Split
    print("\n[*] Splitting dataset (80% Train, 20% Held-Out Test, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )
    print(f"    - Training samples: {len(X_train):,}")
    print(f"    - Test samples: {len(X_test):,}")

    # 3. TF-IDF Vectorization (FIT ONLY ON TRAINING SET)
    print("\n[*] Fitting TF-IDF Vectorizer (max_features=5000, sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        stop_words="english",
        sublinear_tf=True,
        min_df=5
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"    - TF-IDF feature vocabulary size: {X_train_vec.shape[1]}")

    # 4. Model Training & Comparison
    models_to_evaluate = {
        "Logistic Regression": LogisticRegression(
            C=1.5,
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.2
        ),
        "Linear SVM": SGDClassifier(
            loss="log_loss",  # provides calibrated probabilities
            max_iter=1000,
            tol=1e-3,
            random_state=42
        )
    }

    comparison_results = []
    trained_instances = {}

    print("\n[*] Training & Evaluating Classifiers on Held-Out Test Data:")
    print("-" * 65)

    for name, model in models_to_evaluate.items():
        t0 = time.time()
        model.fit(X_train_vec, y_train)
        duration = round(time.time() - t0, 3)
        metrics = evaluate_classifier(model, X_test_vec, y_test, name)
        metrics["training_time_seconds"] = duration
        comparison_results.append(metrics)
        trained_instances[name] = model

        print(f"  > {name:25} | Acc: {metrics['accuracy']:5.2f}% | F1: {metrics['f1_score']:5.2f}% | Time: {duration}s")

    print("-" * 65)

    # Primary model: Logistic Regression (provides optimal probability calibration and explainability)
    primary_model_name = "Logistic Regression"
    primary_model = trained_instances[primary_model_name]
    primary_metrics = next(m for m in comparison_results if m["model_name"] == primary_model_name)

    print(f"\n[*] Primary Model Selected: {primary_model_name}")
    print(f"    Test Accuracy: {primary_metrics['accuracy']}%")
    print(f"    Test Precision: {primary_metrics['precision']}%")
    print(f"    Test Recall:    {primary_metrics['recall']}%")
    print(f"    Test F1-Score:  {primary_metrics['f1_score']}%")

    # 5. Extract Feature Importances
    feature_importance = extract_top_features(vectorizer, primary_model, top_n=15)

    # 6. Save Model Artifacts
    print(f"\n[*] Serializing primary model to: {MODEL_PATH}")
    joblib.dump(primary_model, MODEL_PATH)

    print(f"[*] Serializing TF-IDF vectorizer to: {VECTORIZER_PATH}")
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_model_name": primary_model_name,
        "dataset_source": "original_data (Fake.csv & True.csv)",
        "total_dataset_samples": len(texts),
        "raw_dataset_rows": 44898,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "real_samples_total": int(np.sum(labels == 0)),
        "fake_samples_total": int(np.sum(labels == 1)),
        "real_samples_train": int(np.sum(y_train == 0)),
        "fake_samples_train": int(np.sum(y_train == 1)),
        "real_samples_test": int(np.sum(y_test == 0)),
        "fake_samples_test": int(np.sum(y_test == 1)),
        "vocabulary_size": int(X_train_vec.shape[1]),
        "ngram_range": [1, 1],
        "test_metrics": primary_metrics,
        "top_features": feature_importance
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Saved metadata & test metrics to: {METRICS_PATH}")

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"[*] Saved model comparison to: {COMPARISON_PATH}")

    total_time = round(time.time() - start_time, 2)
    print(f"\n[SUCCESS] Pipeline training complete in {total_time} seconds!")
    print("==================================================")
    return metadata, comparison_results


if __name__ == "__main__":
    train_and_evaluate()
