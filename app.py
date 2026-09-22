"""
Credit Scoring & Risk Assessment System
Flask Web Application & API Server
"""

import os
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from src.risk_engine import CreditRiskEngine, RISK_TIERS
from src.data_preprocessing import FEATURE_DESCRIPTIONS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

app = Flask(__name__)
app.config['SECRET_KEY'] = 'credit-scoring-secure-key-2026'

# Initialize Risk Engine
risk_engine = CreditRiskEngine()


@app.route('/')
def index():
    """Main credit scoring dashboard."""
    metrics = risk_engine.metrics
    champion_name = metrics.get('champion_model', 'Logistic Regression')
    champ_metrics = metrics.get('models', {}).get(champion_name, {})
    
    return render_template(
        'index.html',
        descriptions=FEATURE_DESCRIPTIONS,
        risk_tiers=RISK_TIERS,
        champion_name=champion_name,
        champion_roc=champ_metrics.get('test_roc_auc', 0.81),
        champion_recall=champ_metrics.get('test_recall', 0.80),
        optimal_threshold=risk_engine.optimal_threshold
    )


@app.route('/models')
def models_view():
    """Model evaluation and financial cost optimization dashboard."""
    metrics = risk_engine.metrics
    return render_template(
        'models.html',
        metrics=metrics,
        champion_name=metrics.get('champion_model', 'Logistic Regression')
    )


@app.route('/batch')
def batch_view():
    """Batch applicant evaluation view."""
    return render_template('batch.html')


@app.route('/api/score', methods=['POST'])
def api_score():
    """
    Evaluates applicant data and returns credit score, risk tier,
    decision, justification, and recommendations.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Parse numerical values properly
        formatted_data = {}
        for k, v in data.items():
            if k in NUMERICAL_FEATURES:
                formatted_data[k] = float(v)
            else:
                formatted_data[k] = str(v)
                
        result = risk_engine.evaluate_applicant(formatted_data)
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """
    What-If scenario simulation: calculates score adjustments based on
    modifications to loan duration, amount, and installment rate.
    """
    try:
        body = request.get_json(force=True)
        base = body.get('base', {})
        mods = body.get('modifications', {})
        
        formatted_base = {}
        for k, v in base.items():
            if k in NUMERICAL_FEATURES:
                formatted_base[k] = float(v)
            else:
                formatted_base[k] = str(v)
                
        formatted_mods = {}
        for k, v in mods.items():
            if k in NUMERICAL_FEATURES:
                formatted_mods[k] = float(v)
            else:
                formatted_mods[k] = str(v)
                
        sim_result = risk_engine.simulate_what_if(formatted_base, formatted_mods)
        return jsonify({'success': True, 'data': sim_result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Returns trained models benchmark metrics and curve data."""
    return jsonify(risk_engine.metrics)


@app.route('/api/sample-csv', methods=['GET'])
def download_sample_csv():
    """Generates and serves a downloadable sample CSV for batch scoring."""
    sample_records = [
        {
            'applicant_id': 'APP-1001',
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
        },
        {
            'applicant_id': 'APP-1002',
            'checking_status': 'A12',
            'duration': 24,
            'credit_history': 'A32',
            'purpose': 'A42',
            'credit_amount': 3800,
            'savings_status': 'A62',
            'employment': 'A73',
            'installment_rate': 3,
            'personal_status': 'A92',
            'other_parties': 'A101',
            'residence_since': 2,
            'property_magnitude': 'A122',
            'age': 31,
            'other_payment_plans': 'A143',
            'housing': 'A151',
            'existing_credits': 1,
            'job': 'A173',
            'num_dependents': 1,
            'own_telephone': 'A191',
            'foreign_worker': 'A201'
        },
        {
            'applicant_id': 'APP-1003',
            'checking_status': 'A11',
            'duration': 48,
            'credit_history': 'A34',
            'purpose': 'A41',
            'credit_amount': 7500,
            'savings_status': 'A61',
            'employment': 'A71',
            'installment_rate': 4,
            'personal_status': 'A93',
            'other_parties': 'A101',
            'residence_since': 1,
            'property_magnitude': 'A124',
            'age': 23,
            'other_payment_plans': 'A141',
            'housing': 'A151',
            'existing_credits': 2,
            'job': 'A172',
            'num_dependents': 1,
            'own_telephone': 'A191',
            'foreign_worker': 'A201'
        },
        {
            'applicant_id': 'APP-1004',
            'checking_status': 'A14',
            'duration': 18,
            'credit_history': 'A31',
            'purpose': 'A43',
            'credit_amount': 1800,
            'savings_status': 'A63',
            'employment': 'A74',
            'installment_rate': 2,
            'personal_status': 'A94',
            'other_parties': 'A103',
            'residence_since': 3,
            'property_magnitude': 'A121',
            'age': 52,
            'other_payment_plans': 'A143',
            'housing': 'A152',
            'existing_credits': 1,
            'job': 'A174',
            'num_dependents': 2,
            'own_telephone': 'A192',
            'foreign_worker': 'A201'
        }
    ]
    df = pd.DataFrame(sample_records)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_credit_applicants.csv'
    )


@app.route('/api/batch-score', methods=['POST'])
def api_batch_score():
    """Evaluates an uploaded CSV file of applicants."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
        
    try:
        df = pd.read_csv(file)
        results = []
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            app_id = row_dict.get('applicant_id', f"APP-{idx+1001}")
            
            # Format inputs
            applicant = {}
            for k in NUMERICAL_FEATURES:
                applicant[k] = float(row_dict.get(k, 0))
            for k in CATEGORICAL_FEATURES:
                applicant[k] = str(row_dict.get(k, 'A14' if k == 'checking_status' else 'A40'))
                
            eval_res = risk_engine.evaluate_applicant(applicant)
            
            results.append({
                'applicant_id': app_id,
                'credit_score': eval_res['credit_score'],
                'risk_tier': eval_res['risk_tier'],
                'default_probability': eval_res['default_probability'],
                'decision': eval_res['decision'],
                'color': eval_res['color'],
                'credit_amount': applicant['credit_amount'],
                'duration': applicant['duration'],
                'primary_driver': eval_res['positive_factors'][0]['feature'] if eval_res['positive_factors'] else 'N/A',
                'primary_risk_flag': eval_res['risk_flags'][0]['feature'] if eval_res['risk_flags'] else 'None'
            })
            
        return jsonify({'success': True, 'count': len(results), 'records': results})
        
    except Exception as e:
        return jsonify({'error': f"Failed to process CSV: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Credit Scoring & Risk Assessment Server on http://127.0.0.1:{port}...")
    app.run(host='127.0.0.1', port=port, debug=False)
