"""
Unit tests for text preprocessing, leakage safeguards, and machine learning pipeline.
"""
import unittest
import os
import joblib
from config import MODEL_PATH, VECTORIZER_PATH, METRICS_PATH
from utils.preprocessor import clean_text, tokenize, get_linguistic_stats
from utils.explainer import explain_prediction


class TestModelPipeline(unittest.TestCase):

    def test_clean_text_basic(self):
        raw = "   <p>Hello World! Visit https://example.com for MORE info...</p>   "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "hello world visit for more info")

    def test_strip_publisher_shortcuts(self):
        text_with_shortcut = "GENEVA (Reuters) - The committee met on Wednesday to discuss global policies."
        cleaned = clean_text(text_with_shortcut, strip_shortcuts=True)
        # Should strip the dateline prefix and reuters mention
        self.assertNotIn("reuters", cleaned)
        self.assertIn("committee", cleaned)
        self.assertIn("wednesday", cleaned)

    def test_tokenize_and_stopwords(self):
        text = "This is a comprehensive and structured test of the linguistic tokenizer."
        tokens = tokenize(text, remove_stopwords=True)
        self.assertIn("comprehensive", tokens)
        self.assertIn("structured", tokens)
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)

    def test_linguistic_statistics(self):
        text = "The president addressed the nation. He emphasized economic growth."
        stats = get_linguistic_stats(text)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertGreater(stats["word_count"], 5)
        self.assertGreater(stats["char_count"], 20)
        self.assertGreater(stats["lexical_diversity"], 50.0)

    def test_empty_and_null_inputs(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        stats = get_linguistic_stats("")
        self.assertEqual(stats["word_count"], 0)
        self.assertEqual(stats["char_count"], 0)

    def test_model_artifacts_exist(self):
        self.assertTrue(os.path.exists(MODEL_PATH), "Primary model artifact should exist")
        self.assertTrue(os.path.exists(VECTORIZER_PATH), "TF-IDF vectorizer artifact should exist")
        self.assertTrue(os.path.exists(METRICS_PATH), "Metadata metrics file should exist")

    def test_model_inference_pipeline(self):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        sample_real = clean_text("The foreign minister said in a statement on Tuesday that parliament approved the fiscal bill.")
        vec_real = vectorizer.transform([sample_real])
        pred_real = model.predict(vec_real)[0]
        # Label 0 represents Real News
        self.assertEqual(pred_real, 0)

        sample_fake = clean_text("WATCH: Shocking viral video reveals proof that secret cabal poisoned the town water supply!")
        vec_fake = vectorizer.transform([sample_fake])
        pred_fake = model.predict(vec_fake)[0]
        # Label 1 represents Fake News
        self.assertEqual(pred_fake, 1)

    def test_unseen_vocabulary_handling(self):
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)

        gibberish = "xyzabc123 qwertyuiop asdfghjkl zxcvbnm"
        clean_gibberish = clean_text(gibberish)
        vec = vectorizer.transform([clean_gibberish])
        # Should not throw exception and predict gracefully
        pred = model.predict(vec)
        self.assertIn(pred[0], (0, 1))

    def test_explain_prediction(self):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        text = clean_text("Exclusive video watch proof that corrupt officials hid the documents")
        explanation = explain_prediction(text, model, vectorizer, top_k=5)

        self.assertIn("fake_indicators", explanation)
        self.assertIn("real_indicators", explanation)
        self.assertIsInstance(explanation["fake_indicators"], list)
        self.assertIsInstance(explanation["real_indicators"], list)


if __name__ == "__main__":
    unittest.main()
