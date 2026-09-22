"""
Automated Test Suite for Credit Scoring & Risk Assessment System
"""

import os
import unittest
import numpy as np
import pandas as pd
from src.data_preprocessing import load_raw_data, prepare_datasets, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.model_training import calculate_financial_cost, optimize_threshold_for_cost
from src.risk_engine import CreditRiskEngine, RISK_TIERS
from app import app


class TestCreditScoringSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CreditRiskEngine()
        cls.app_client = app.test_client()

    def test_01_dataset_loading_and_schema(self):
        """Verify raw German Credit dataset loads with expected shape and labels."""
        df = load_raw_data('german.data')
        self.assertEqual(len(df), 1000)
        self.assertIn('risk_label', df.columns)
        # Check target distribution: 700 Good (0), 300 Bad (1)
        counts = df['risk_label'].value_counts()
        self.assertEqual(counts[0], 700)
        self.assertEqual(counts[1], 300)

    def test_02_preprocessing_pipeline(self):
        """Verify feature transformation produces consistent numerical matrices."""
        data = prepare_datasets('german.data')
        self.assertEqual(len(data['X_train']), 800)
        self.assertEqual(len(data['X_test']), 200)
        self.assertGreater(len(data['feature_names']), 50)
        self.assertFalse(np.isnan(data['X_train']).any())

    def test_03_cost_matrix_calculation(self):
        """Verify the 5:1 penalty calculation logic."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        # Actual 0, Pred 1 -> 1 FP (Cost: 1)
        # Actual 1, Pred 0 -> 1 FN (Cost: 5)
        # Total cost: 1 + 5 = 6. Avg cost = 6/4 = 1.5
        total_cost, avg_cost, cm = calculate_financial_cost(y_true, y_pred)
        self.assertEqual(total_cost, 6.0)
        self.assertEqual(avg_cost, 1.5)
        self.assertEqual(cm['fn'], 1)
        self.assertEqual(cm['fp'], 1)

    def test_04_cost_threshold_optimization(self):
        """Verify threshold sweep finds an optimal cutoff minimizing loss."""
        y_true = np.array([0]*70 + [1]*30)
        # Synthetic probabilities
        np.random.seed(42)
        y_prob = np.concatenate([np.random.uniform(0.1, 0.6, 70), np.random.uniform(0.4, 0.9, 30)])
        res = optimize_threshold_for_cost(y_true, y_prob)
        self.assertIn('optimal_threshold', res)
        self.assertIn('min_cost_per_applicant', res)
        self.assertLessEqual(res['min_cost_per_applicant'], res['default_cost_per_applicant'])

    def test_05_risk_engine_scoring_and_bounds(self):
        """Verify FICO score conversion stays within 300 to 850."""
        # Low risk test
        score_low_risk = self.engine.calculate_credit_score(0.05)
        self.assertGreaterEqual(score_low_risk, 750)
        self.assertLessEqual(score_low_risk, 850)

        # High risk test
        score_high_risk = self.engine.calculate_credit_score(0.95)
        self.assertGreaterEqual(score_high_risk, 300)
        self.assertLessEqual(score_high_risk, 450)

    def test_06_single_applicant_evaluation_and_justification(self):
        """Verify full evaluation returns decision, tiers, and explanation factors."""
        applicant = {
            'checking_status': 'A13',
            'duration': 12,
            'credit_history': 'A32',
            'purpose': 'A40',
            'credit_amount': 2500,
            'savings_status': 'A64',
            'employment': 'A75',
            'installment_rate': 2,
            'personal_status': 'A93',
            'other_parties': 'A101',
            'residence_since': 4,
            'property_magnitude': 'A121',
            'age': 45,
            'other_payment_plans': 'A143',
            'housing': 'A152',
            'existing_credits': 2,
            'job': 'A173',
            'num_dependents': 1,
            'own_telephone': 'A192',
            'foreign_worker': 'A201'
        }
        res = self.engine.evaluate_applicant(applicant)
        self.assertIn('credit_score', res)
        self.assertIn('risk_tier', res)
        self.assertIn('decision', res)
        self.assertIn('positive_factors', res)
        self.assertIn('risk_flags', res)
        self.assertIn('recommendations', res)
        self.assertGreater(len(res['positive_factors']), 0)

    def test_07_what_if_simulation(self):
        """Verify what-if simulation calculates differential score correctly."""
        base = {
            'checking_status': 'A11',
            'duration': 48,
            'credit_history': 'A34',
            'purpose': 'A41',
            'credit_amount': 8000,
            'savings_status': 'A61',
            'employment': 'A71',
            'installment_rate': 4,
            'personal_status': 'A93',
            'other_parties': 'A101',
            'residence_since': 1,
            'property_magnitude': 'A124',
            'age': 25,
            'other_payment_plans': 'A141',
            'housing': 'A151',
            'existing_credits': 1,
            'job': 'A172',
            'num_dependents': 1,
            'own_telephone': 'A191',
            'foreign_worker': 'A201'
        }
        # Shorten duration and reduce loan amount
        mods = {'duration': 12, 'credit_amount': 2000, 'other_parties': 'A103'}
        sim = self.engine.simulate_what_if(base, mods)
        self.assertIn('simulated_score', sim)
        self.assertIn('score_change', sim)
        # Improving terms should raise or maintain score
        self.assertGreaterEqual(sim['simulated_score'], sim['original_score'])

    def test_08_web_endpoints(self):
        """Verify Flask web server routes respond properly."""
        # 1. Home
        res_home = self.app_client.get('/')
        self.assertEqual(res_home.status_code, 200)

        # 2. Models
        res_models = self.app_client.get('/models')
        self.assertEqual(res_models.status_code, 200)

        # 3. Batch
        res_batch = self.app_client.get('/batch')
        self.assertEqual(res_batch.status_code, 200)

        # 4. Metrics API
        res_metrics = self.app_client.get('/api/metrics')
        self.assertEqual(res_metrics.status_code, 200)
        data = res_metrics.get_json()
        self.assertIn('champion_model', data)

        # 5. Score API
        test_payload = {
            'checking_status': 'A12',
            'duration': 24,
            'credit_history': 'A32',
            'purpose': 'A40',
            'credit_amount': 3000,
            'savings_status': 'A62',
            'employment': 'A73',
            'installment_rate': 3,
            'personal_status': 'A93',
            'other_parties': 'A101',
            'residence_since': 2,
            'property_magnitude': 'A122',
            'age': 35,
            'other_payment_plans': 'A143',
            'housing': 'A152',
            'existing_credits': 1,
            'job': 'A173',
            'num_dependents': 1,
            'own_telephone': 'A191',
            'foreign_worker': 'A201'
        }
        res_score = self.app_client.post('/api/score', json=test_payload)
        self.assertEqual(res_score.status_code, 200)
        score_data = res_score.get_json()
        self.assertTrue(score_data['success'])
        self.assertIn('credit_score', score_data['data'])

        # 6. Sample CSV download
        res_csv = self.app_client.get('/api/sample-csv')
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn('text/csv', res_csv.headers['Content-Type'])


if __name__ == '__main__':
    unittest.main()
