"""
Model explainability module.
Extracts top positive and negative feature contributions for an input text
using the model's internal weights and the document's TF-IDF representations.
"""
import numpy as np


def explain_prediction(text_clean: str, model, vectorizer, top_k: int = 6) -> dict:
    """
    Computes feature contributions for the given preprocessed text.
    
    Returns:
    {
        'fake_indicators': list of {'token': str, 'weight': float, 'direction': 'fake'},
        'real_indicators': list of {'token': str, 'weight': float, 'direction': 'real'},
        'total_tokens_evaluated': int
    }
    """
    if not text_clean or not model or not vectorizer:
        return {'fake_indicators': [], 'real_indicators': [], 'total_tokens_evaluated': 0}
    
    try:
        # Transform the single document into TF-IDF vector
        tfidf_vec = vectorizer.transform([text_clean])
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        # Non-zero feature indices in this document
        row_indices = tfidf_vec.nonzero()[1]
        if len(row_indices) == 0:
            return {'fake_indicators': [], 'real_indicators': [], 'total_tokens_evaluated': 0}
        
        tfidf_values = tfidf_vec.data
        
        # Determine coefficients depending on model type
        if hasattr(model, 'coef_'):
            # Logistic Regression or Linear SVM: shape (1, n_features)
            coefs = model.coef_[0]
            # Contribution = w_i * x_i
            contributions = coefs[row_indices] * tfidf_values
        elif hasattr(model, 'feature_log_prob_'):
            # Multinomial Naive Bayes: log P(w|Fake) - log P(w|Real)
            log_prob_fake = model.feature_log_prob_[1]
            log_prob_real = model.feature_log_prob_[0]
            log_ratio = log_prob_fake - log_prob_real
            contributions = log_ratio[row_indices] * tfidf_values
        else:
            return {'fake_indicators': [], 'real_indicators': [], 'total_tokens_evaluated': 0}
        
        present_features = feature_names[row_indices]
        
        # Sort contributions
        # Positive values push towards Fake (Class 1)
        # Negative values push towards Real (Class 0)
        sorted_pairs = list(zip(present_features, contributions))
        
        # Fake indicators: highest positive contributions
        fake_tokens = [
            {'token': str(feat), 'weight': round(float(val), 4), 'direction': 'fake'}
            for feat, val in sorted(sorted_pairs, key=lambda x: x[1], reverse=True)
            if val > 0.001
        ][:top_k]
        
        # Real indicators: most negative contributions (pushing towards 0 = Real)
        real_tokens = [
            {'token': str(feat), 'weight': round(float(abs(val)), 4), 'direction': 'real'}
            for feat, val in sorted(sorted_pairs, key=lambda x: x[1])
            if val < -0.001
        ][:top_k]
        
        return {
            'fake_indicators': fake_tokens,
            'real_indicators': real_tokens,
            'total_tokens_evaluated': len(row_indices)
        }
    except Exception as e:
        print(f"Explainability computation notice: {e}")
        return {'fake_indicators': [], 'real_indicators': [], 'total_tokens_evaluated': 0}
