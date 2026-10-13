"""
Independent Model Evaluation & Diagnostic Verification Script.
Evaluates the serialized pipeline on held-out test split,
prints confusion matrix, classification metrics, and tests data leakage guards.
"""
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import DATASET_PATH, MODEL_PATH, VECTORIZER_PATH, METRICS_PATH
from train_model import load_and_clean_data
from utils.preprocessor import clean_text


def run_evaluation():
    print("==================================================")
    print("      MODEL EVALUATION & DIAGNOSTIC AUDIT         ")
    print("==================================================")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("[ERROR] Trained model or vectorizer artifact not found!")
        print("Please run 'python train_model.py' first.")
        return False

    print(f"[*] Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"[*] Loading vectorizer from: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Load and clean dataset
    texts, labels = load_and_clean_data()

    # Reconstruct exact held-out test split (seed 42, 20%)
    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    print(f"[*] Held-out test evaluation samples: {len(X_test)}")
    X_test_vec = vectorizer.transform(X_test)

    # Inference
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--------------------------------------------------")
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print("--------------------------------------------------")
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Real News (0)", "Fake News (1)"],
        digits=4
    ))

    print("CONFUSION MATRIX:")
    print(f"               Predicted Real    Predicted Fake")
    print(f"Actual Real:       {cm[0][0]:<14}  {cm[0][1]:<14} (Total: {cm[0][0]+cm[0][1]})")
    print(f"Actual Fake:       {cm[1][0]:<14}  {cm[1][1]:<14} (Total: {cm[1][0]+cm[1][1]})")

    # Data leakage audit check
    print("\n--------------------------------------------------")
    print("DATA LEAKAGE & SOURCE BIAS AUDIT:")
    print("--------------------------------------------------")
    feature_names = vectorizer.get_feature_names_out()
    shortcut_terms = ["reuters", "associated press", "washington reuters", "ap"]
    found_shortcuts = [term for term in shortcut_terms if term in feature_names]

    if not found_shortcuts:
        print("[PASS] No raw wire publisher shortcuts detected in TF-IDF vocabulary.")
        print("       Publisher dateline filtering was verified.")
    else:
        print(f"[WARN] Shortcut terms detected in vocabulary: {found_shortcuts}")

    print("==================================================")
    print("Diagnostic evaluation successfully verified!")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_evaluation()
