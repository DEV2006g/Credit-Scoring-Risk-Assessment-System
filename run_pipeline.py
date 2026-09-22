"""
Credit Scoring & Risk Assessment System
End-to-End Pipeline Execution Script
"""

import os
import sys
import time

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import prepare_datasets
from src.model_training import train_and_evaluate_all_models
from src.risk_engine import CreditRiskEngine

def main():
    print("=" * 70)
    print("CREDIT SCORING & RISK ASSESSMENT SYSTEM - TRAINING PIPELINE")
    print("=" * 70)
    start_time = time.time()
    
    # Step 1: Preprocess Data
    print("\n[Step 1/3] Preprocessing raw UCI German Credit Dataset...")
    data_dict = prepare_datasets(data_path='german.data')
    print(f"-> Train samples: {len(data_dict['X_train'])}, Test samples: {len(data_dict['X_test'])}")
    print(f"-> Feature dimensionality after encoding: {len(data_dict['feature_names'])}")
    
    # Step 2: Train & Evaluate Models with Cost Matrix Optimization
    print("\n[Step 2/3] Training candidate models & optimizing decision thresholds...")
    summary, champion_model = train_and_evaluate_all_models(data_dict, models_dir='models')
    
    # Step 3: Verify Risk Engine Scoring
    print("\n[Step 3/3] Initializing Credit Risk Engine and verifying sample scoring...")
    engine = CreditRiskEngine()
    
    sample_prime = {
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
    
    eval_res = engine.evaluate_applicant(sample_prime)
    print(f"-> Sample Prime Applicant Score: {eval_res['credit_score']} ({eval_res['risk_tier']})")
    print(f"-> Default Probability: {eval_res['default_probability']}% | Decision: {eval_res['decision']}")
    print(f"-> Top Positive Factor: {eval_res['positive_factors'][0]['feature']} - {eval_res['positive_factors'][0]['detail']}")
    
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed}s")
    print(f"Champion Model: {summary['champion_model']}")
    print("=" * 70)

if __name__ == '__main__':
    main()
