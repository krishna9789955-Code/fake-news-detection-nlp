"""
Flask Web Application for Fake News Detection Using NLP.
Provides UI views, real-time prediction API, performance metrics, and history tracking.
"""
import os
import json
import joblib
from flask import Flask, render_template, request, jsonify

from config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    MAX_INPUT_CHARS,
    MIN_INPUT_CHARS,
    CLASS_NAMES,
    PORT,
    HOST,
    DEBUG,
    SECRET_KEY
)
from utils.preprocessor import clean_text, get_linguistic_stats
from utils.explainer import explain_prediction
from database.db import (
    init_db,
    record_prediction,
    get_history,
    get_history_stats,
    clear_history
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JSON_SORT_KEYS'] = False

# Ensure database is initialized
init_db()

# Lazy-loaded or pre-loaded ML artifacts
MODEL = None
VECTORIZER = None
METADATA = None
COMPARISON = None


def load_artifacts():
    """Loads serialized model, vectorizer, and evaluation metrics if available."""
    global MODEL, VECTORIZER, METADATA, COMPARISON
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
        if os.path.exists(VECTORIZER_PATH):
            VECTORIZER = joblib.load(VECTORIZER_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                METADATA = json.load(f)
        if os.path.exists(COMPARISON_PATH):
            with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
                COMPARISON = json.load(f)
    except Exception as e:
        app.logger.error(f"Error loading model artifacts: {e}")


# Load artifacts at startup
load_artifacts()


# -------------------------------------------------------------
# Frontend Page Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    """Home / Landing Dashboard."""
    db_stats = get_history_stats()
    metrics = METADATA if METADATA else {}
    return render_template(
        "index.html",
        page="home",
        metrics=metrics,
        db_stats=db_stats
    )


@app.route("/detect")
def detect():
    """Interactive Fake News Analysis Page."""
    return render_template("detect.html", page="detect")


@app.route("/dashboard")
def dashboard():
    """Model Analytics and Performance Dashboard."""
    metrics = METADATA if METADATA else {}
    return render_template("dashboard.html", page="dashboard", metrics=metrics)


@app.route("/about")
def about():
    """About Model, NLP Pipeline & Viva Voce Guide."""
    metrics = METADATA if METADATA else {}
    return render_template("about.html", page="about", metrics=metrics)


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint to inspect server and model availability."""
    model_loaded = MODEL is not None and VECTORIZER is not None
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "primary_model": METADATA.get("primary_model_name", "Logistic Regression") if METADATA else "None",
        "dataset_samples": METADATA.get("total_dataset_samples", 0) if METADATA else 0,
        "test_accuracy": METADATA.get("test_metrics", {}).get("accuracy", 0.0) if METADATA else 0.0
    }), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Analyzes supplied news headline and/or article body.
    Returns prediction, calibrated confidence, feature contributions, and linguistic stats.
    """
    # Verify model is ready
    if MODEL is None or VECTORIZER is None:
        load_artifacts()
        if MODEL is None or VECTORIZER is None:
            return jsonify({
                "success": False,
                "status": "model_unavailable",
                "error": "Machine learning model artifacts are not currently loaded. Please run 'python train_model.py'."
            }), 503

    # Parse JSON payload
    data = request.get_json(silent=True) or {}
    headline = (data.get("headline") or "").strip()
    article = (data.get("article") or "").strip()

    # Determine input mode
    if headline and article:
        input_mode = "combined"
        raw_text = f"{headline}\n\n{article}"
    elif headline:
        input_mode = "headline"
        raw_text = headline
    elif article:
        input_mode = "article"
        raw_text = article
    else:
        return jsonify({
            "success": False,
            "status": "unable_to_assess",
            "error": "Please provide either a news headline, an article text, or both."
        }), 400

    # Input length validation
    if len(raw_text) > MAX_INPUT_CHARS:
        return jsonify({
            "success": False,
            "status": "input_too_long",
            "error": f"Input exceeds maximum allowed limit of {MAX_INPUT_CHARS:,} characters."
        }), 400

    if len(raw_text) < MIN_INPUT_CHARS:
        return jsonify({
            "success": False,
            "status": "unable_to_assess",
            "error": "Input text is too short to perform reliable NLP feature analysis (minimum 10 characters)."
        }), 400

    # Linguistic analysis on raw text
    linguistic_stats = get_linguistic_stats(raw_text)

    # Consistent preprocessing with training pipeline
    cleaned_text = clean_text(raw_text, strip_shortcuts=True)

    # Check if any usable tokens remain
    if not cleaned_text or linguistic_stats["processed_tokens"] < 2:
        return jsonify({
            "success": False,
            "status": "unable_to_assess",
            "error": "The input contains only symbols, numbers, or stopwords with insufficient contextual vocabulary.",
            "linguistics": linguistic_stats
        }), 400

    try:
        # TF-IDF Feature Transformation
        features = VECTORIZER.transform([cleaned_text])

        # Prediction and probabilities
        pred_class = int(MODEL.predict(features)[0])
        
        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(features)[0]
            real_prob = round(float(probabilities[0]) * 100, 2)
            fake_prob = round(float(probabilities[1]) * 100, 2)
            confidence = round(float(max(probabilities)) * 100, 1)
        else:
            # Fallback for models without calibrated predict_proba
            confidence = 90.0
            real_prob = 100.0 if pred_class == 0 else 0.0
            fake_prob = 100.0 if pred_class == 1 else 0.0

        prediction_label = CLASS_NAMES.get(pred_class, "Unknown")

        # Explainability feature contributions
        explanation = explain_prediction(cleaned_text, MODEL, VECTORIZER, top_k=6)

        # Top indicator word for history record
        top_indicator = ""
        if pred_class == 1 and explanation["fake_indicators"]:
            top_indicator = explanation["fake_indicators"][0]["token"]
        elif pred_class == 0 and explanation["real_indicators"]:
            top_indicator = explanation["real_indicators"][0]["token"]

        # Record prediction into SQLite
        display_title = headline if headline else (article[:90] + "..." if len(article) > 90 else article)
        record_prediction(
            title=display_title,
            snippet=article if article else headline,
            input_mode=input_mode,
            prediction=prediction_label,
            predicted_class=pred_class,
            confidence=confidence,
            real_prob=real_prob,
            fake_prob=fake_prob,
            top_indicator=top_indicator
        )

        # Contextual explanation summary
        if pred_class == 1:
            analysis_summary = (
                f"The text exhibits linguistic patterns, sensational tone, or rhetorical markers "
                f"frequently found in fabricated or unverified reports (confidence: {confidence}%)."
            )
        else:
            analysis_summary = (
                f"The text exhibits formal journalistic attribution, objective phrasing, and editorial "
                f"syntax characteristic of verified news reports (confidence: {confidence}%)."
            )

        return jsonify({
            "success": True,
            "status": "success",
            "prediction": prediction_label,
            "predicted_class": pred_class,
            "class_slug": "fake" if pred_class == 1 else "real",
            "confidence": confidence,
            "probabilities": {
                "real": real_prob,
                "fake": fake_prob
            },
            "analysis_summary": analysis_summary,
            "explainability": explanation,
            "linguistics": linguistic_stats,
            "input_mode": input_mode,
            "disclaimer": (
                "Notice: This prediction is a statistical estimate produced by a trained TF-IDF & "
                "Logistic Regression NLP classifier. It analyzes writing style and vocabulary patterns, "
                "not live external truth-verification. Always verify crucial news with trusted official sources."
            )
        }), 200

    except Exception as e:
        app.logger.error(f"Prediction execution error: {e}")
        return jsonify({
            "success": False,
            "status": "error",
            "error": "An internal error occurred during NLP analysis. Please check input text format."
        }), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns actual dataset evaluation metrics and model comparison data."""
    if not METADATA:
        load_artifacts()
    
    db_stats = get_history_stats()
    return jsonify({
        "metadata": METADATA or {},
        "comparison": COMPARISON or [],
        "db_stats": db_stats
    }), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """Retrieves recorded prediction history with optional filter and search."""
    limit = request.args.get("limit", default=50, type=int)
    filter_class = request.args.get("filter", default=None, type=str)
    search = request.args.get("search", default=None, type=str)
    
    # Cap limit to 100 items
    limit = min(max(1, limit), 100)
    
    history_items = get_history(limit=limit, filter_class=filter_class, search=search)
    stats = get_history_stats()
    
    return jsonify({
        "success": True,
        "count": len(history_items),
        "history": history_items,
        "stats": stats
    }), 200


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears SQLite prediction history."""
    clear_history()
    return jsonify({
        "success": True,
        "message": "Prediction history successfully cleared."
    }), 200


# -------------------------------------------------------------
# Global Error Handlers
# -------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template("index.html", page="home", metrics=METADATA or {}, db_stats=get_history_stats()), 404


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": "Internal server error"}), 500
    return render_template("index.html", page="home", metrics=METADATA or {}, db_stats=get_history_stats()), 500


if __name__ == "__main__":
    print(f"[*] Starting Fake News Detection Server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
