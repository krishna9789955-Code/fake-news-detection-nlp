"""
Unit and integration tests for Flask web application and REST API endpoints.
"""
import unittest
import json
from app import app


class TestAppEndpoints(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fake News Detection', response.data)

    def test_detect_page(self):
        response = self.client.get('/detect')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analyze News Headline', response.data)

    def test_dashboard_page(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Confusion Matrix', response.data)

    def test_about_page(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'TF-IDF Feature Engineering', response.data)

    def test_api_health(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))
        self.assertGreater(data.get('test_accuracy'), 90.0)

    def test_api_metrics(self):
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('metadata', data)
        self.assertIn('comparison', data)
        self.assertIsInstance(data['comparison'], list)
        self.assertGreater(len(data['comparison']), 1)

    def test_predict_headline_only(self):
        payload = {
            "headline": "Senate reaches consensus on bipartisan infrastructure spending bill"
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn(data.get('prediction'), ('Likely Real News', 'Likely Fake News'))
        self.assertIn('confidence', data)
        self.assertEqual(data.get('input_mode'), 'headline')

    def test_predict_article_only(self):
        payload = {
            "article": "The state governor announced new healthcare subsidies on Wednesday during a press briefing attended by medical directors."
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('input_mode'), 'article')
        self.assertIn('probabilities', data)

    def test_predict_combined(self):
        payload = {
            "headline": "Historic scientific breakthrough in renewable energy fusion",
            "article": "Researchers at the international laboratory confirmed continuous plasma containment exceeding ninety seconds."
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('input_mode'), 'combined')
        self.assertIn('explainability', data)
        self.assertIn('linguistics', data)

    def test_predict_empty_validation(self):
        response = self.client.post('/api/predict', json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get('success'))
        self.assertEqual(data.get('status'), 'unable_to_assess')

    def test_predict_short_text_validation(self):
        response = self.client.post('/api/predict', json={"headline": "hi"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get('success'))

    def test_api_history_flow(self):
        # 1. Clear history first
        clear_res = self.client.post('/api/history/clear')
        self.assertEqual(clear_res.status_code, 200)

        # 2. Make a prediction to generate a record
        self.client.post('/api/predict', json={
            "headline": "United Nations climate summit approves carbon reduction pact"
        })

        # 3. Retrieve history
        get_res = self.client.get('/api/history')
        self.assertEqual(get_res.status_code, 200)
        data = get_res.get_json()
        self.assertTrue(data.get('success'))
        self.assertGreaterEqual(len(data.get('history')), 1)


if __name__ == "__main__":
    unittest.main()
