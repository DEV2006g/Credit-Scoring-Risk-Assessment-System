"""
Credit Scoring & Risk Assessment System
Risk Engine: FICO Scorecard (300-850), Risk Tiering, Decision Justification & Recommendations
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Risk Tier Thresholds and Definitions
RISK_TIERS = [
    {
        'tier': 'Excellent',
        'min_score': 750,
        'max_score': 850,
        'decision': 'Approved',
        'color': '#10B981',  # Emerald green
        'badge_class': 'badge-success',
        'suggested_apr': '5.49%',
        'approval_limit_multiplier': 1.5,
        'summary': 'Prime borrower with exceptionally low probability of default. Instant automated approval recommended.'
    },
    {
        'tier': 'Good',
        'min_score': 670,
        'max_score': 749,
        'decision': 'Approved',
        'color': '#3B82F6',  # Blue
        'badge_class': 'badge-info',
        'suggested_apr': '8.25%',
        'approval_limit_multiplier': 1.1,
        'summary': 'Solid credit applicant with dependable financial profile. Standard automated loan clearance.'
    },
    {
        'tier': 'Fair',
        'min_score': 580,
        'max_score': 669,
        'decision': 'Conditional Approval',
        'color': '#F59E0B',  # Amber
        'badge_class': 'badge-warning',
        'suggested_apr': '13.75%',
        'approval_limit_multiplier': 0.8,
        'summary': 'Moderate risk profile. Approval recommended subject to collateral verification or secondary underwriter review.'
    },
    {
        'tier': 'Poor',
        'min_score': 500,
        'max_score': 579,
        'decision': 'Manual Review Required',
        'color': '#F97316',  # Orange
        'badge_class': 'badge-orange',
        'suggested_apr': '19.50%',
        'approval_limit_multiplier': 0.5,
        'summary': 'Elevated default risk. Requires guarantor, reduced loan amount, or substantial collateralization.'
    },
    {
        'tier': 'Very Poor',
        'min_score': 300,
        'max_score': 499,
        'decision': 'Declined',
        'color': '#EF4444',  # Red
        'badge_class': 'badge-danger',
        'suggested_apr': 'N/A',
        'approval_limit_multiplier': 0.0,
        'summary': 'High default probability exceeds institutional risk tolerance. Application declined.'
    }
]


class CreditRiskEngine:
    def __init__(self, model_path='models/credit_risk_model.joblib',
                 preprocessor_path='models/preprocessor.joblib',
                 metrics_path='models/model_metrics.json'):
        self.model = None
        self.preprocessor = None
        self.metrics = {}
        self.optimal_threshold = 0.5
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
                champ = self.metrics.get('champion_model')
                if champ and champ in self.metrics.get('models', {}):
                    self.optimal_threshold = self.metrics['models'][champ].get('optimal_threshold', 0.5)

    def calculate_credit_score(self, default_prob):
        """
        Converts probability of default (0.0 to 1.0) into a standard FICO credit score (300 to 850).
        Score = 850 - 550 * PD
        """
        score = int(round(850 - (550 * default_prob)))
        return max(300, min(850, score))

    def get_risk_tier_info(self, credit_score):
        """
        Returns tier metadata based on credit score.
        """
        for tier in RISK_TIERS:
            if tier['min_score'] <= credit_score <= tier['max_score']:
                return tier
        return RISK_TIERS[-1]

    def evaluate_applicant(self, applicant_data):
        """
        Assesses a single applicant dictionary:
        1. Preprocesses features
        2. Estimates Default Probability
        3. Computes Credit Score & Risk Tier
        4. Derives Decision Justification (Positive drivers & Negative risk flags)
        5. Formulates actionable recommendations
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded. Train models first.")
            
        df = pd.DataFrame([applicant_data])
        X_processed = self.preprocessor.transform(df)
        
        # Risk estimation
        prob_default = float(self.model.predict_proba(X_processed)[0, 1])
        credit_score = self.calculate_credit_score(prob_default)
        tier_info = self.get_risk_tier_info(credit_score)
        
        # Decision based on risk tier and optimal threshold
        cost_optimized_decision = 'Declined' if prob_default >= self.optimal_threshold else 'Approved'
        final_decision = tier_info['decision']
        
        # Loan terms derivation
        requested_amount = float(applicant_data.get('credit_amount', 3000))
        max_approved_limit = round(requested_amount * tier_info['approval_limit_multiplier'], -2)
        
        # Justification & Explanations
        justification = self._generate_decision_justification(applicant_data, prob_default)
        recommendations = self._generate_recommendations(applicant_data, prob_default, tier_info['tier'])
        
        return {
            'credit_score': credit_score,
            'default_probability': round(prob_default * 100, 2),
            'risk_tier': tier_info['tier'],
            'decision': final_decision,
            'cost_optimized_decision': cost_optimized_decision,
            'optimal_threshold': self.optimal_threshold,
            'color': tier_info['color'],
            'badge_class': tier_info['badge_class'],
            'suggested_apr': tier_info['suggested_apr'],
            'max_approved_limit': max_approved_limit,
            'summary': tier_info['summary'],
            'positive_factors': justification['positive_factors'],
            'risk_flags': justification['risk_flags'],
            'recommendations': recommendations
        }

    def _generate_decision_justification(self, data, pd_val):
        """
        Identifies key attributes that significantly elevate or reduce credit risk
        based on empirical domain heuristics and model weights.
        """
        positives = []
        flags = []
        
        # Checking status
        checking = data.get('checking_status', '')
        if checking in ['A13']:
            positives.append({'feature': 'Liquid Checking Account', 'impact': 'Strongly Positive', 'detail': 'Substantial balance (>= 200 DM) with regular salary credits.'})
        elif checking in ['A14']:
            positives.append({'feature': 'No Overdraft Record', 'impact': 'Positive', 'detail': 'Applicant has no deficit checking account balance.'})
        elif checking == 'A11':
            flags.append({'feature': 'Overdrawn Account', 'impact': 'Critical Risk Flag', 'detail': 'Existing checking account is running negative (< 0 DM).'})
        elif checking == 'A12':
            flags.append({'feature': 'Low Checking Buffer', 'impact': 'Moderate Risk Flag', 'detail': 'Checking balance is modest (under 200 DM).'})

        # Credit History
        history = data.get('credit_history', '')
        if history in ['A30', 'A31']:
            positives.append({'feature': 'Flawless Credit History', 'impact': 'Positive', 'detail': 'Prior bank credits paid back duly with zero delinquency.'})
        elif history == 'A34':
            flags.append({'feature': 'Critical Credit History', 'impact': 'Significant Risk Flag', 'detail': 'Account history shows other existing credits or critical past repayment issues.'})
        elif history == 'A33':
            flags.append({'feature': 'Past Payment Delays', 'impact': 'Moderate Risk Flag', 'detail': 'Applicant has experienced delayed loan payments in the past.'})

        # Duration
        duration = float(data.get('duration', 24))
        if duration <= 12:
            positives.append({'feature': 'Short Loan Horizon', 'impact': 'Positive', 'detail': f'Tenure of {int(duration)} months minimizes long-term default exposure.'})
        elif duration >= 36:
            flags.append({'feature': 'Extended Loan Duration', 'impact': 'High Risk Flag', 'detail': f'Tenure of {int(duration)} months creates prolonged risk exposure.'})

        # Savings
        savings = data.get('savings_status', '')
        if savings in ['A63', 'A64']:
            positives.append({'feature': 'Robust Liquidity Reserves', 'impact': 'Strongly Positive', 'detail': 'Savings accounts/bonds exceed 500-1000 DM acting as liquidity buffer.'})
        elif savings == 'A61':
            flags.append({'feature': 'Minimal Liquid Reserves', 'impact': 'Moderate Risk Flag', 'detail': 'Savings balance is under 100 DM, offering little emergency buffer.'})

        # Employment
        employment = data.get('employment', '')
        if employment in ['A74', 'A75']:
            positives.append({'feature': 'Career Tenure', 'impact': 'Positive', 'detail': 'Employed at current employer for over 4+ years indicating income stability.'})
        elif employment == 'A71':
            flags.append({'feature': 'Unemployed Status', 'impact': 'Critical Risk Flag', 'detail': 'Current unemployed status poses significant repayment uncertainty.'})
        elif employment == 'A72':
            flags.append({'feature': 'Short Job Tenure', 'impact': 'Moderate Risk Flag', 'detail': 'Employed for less than 1 year at current workplace.'})

        # Property / Collateral
        prop = data.get('property_magnitude', '')
        if prop == 'A121':
            positives.append({'feature': 'Real Estate Ownership', 'impact': 'Strongly Positive', 'detail': 'Applicant owns real estate property providing tangible collateral backing.'})
        elif prop == 'A124':
            flags.append({'feature': 'No Asset Ownership', 'impact': 'Moderate Risk Flag', 'detail': 'No documented real estate, building society, or vehicular collateral.'})

        # Co-debtor / Guarantor
        parties = data.get('other_parties', '')
        if parties in ['A102', 'A103']:
            positives.append({'feature': 'Guarantor / Co-applicant', 'impact': 'Positive', 'detail': 'Additional party shares loan liability, reducing single-point default risk.'})

        # Installment rate
        rate = float(data.get('installment_rate', 2))
        if rate >= 4:
            flags.append({'feature': 'High Debt Burden Rate', 'impact': 'Significant Risk Flag', 'detail': 'Installment absorbs >= 4% of disposable income per bracket.'})

        # Default fallback if empty
        if not positives:
            positives.append({'feature': 'Standard Application Profile', 'impact': 'Neutral', 'detail': 'Applicant meets baseline identification and statutory criteria.'})
        if not flags:
            flags.append({'feature': 'No High-Severity Risk Red Flags Detected', 'impact': 'Low Risk', 'detail': 'Profile does not exhibit major historical default markers.'})

        return {'positive_factors': positives[:4], 'risk_flags': flags[:4]}

    def _generate_recommendations(self, data, pd_val, tier):
        """
        Formulates tailored, actionable guidance for improving score or mitigating risk.
        """
        recs = []
        duration = float(data.get('duration', 24))
        credit_amount = float(data.get('credit_amount', 3000))
        checking = data.get('checking_status', '')
        savings = data.get('savings_status', '')
        parties = data.get('other_parties', '')
        
        if tier in ['Excellent', 'Good']:
            recs.append("Eligible for prime interest rate reductions upon enrolling in automatic salary deduction.")
            recs.append("Consider credit line expansion option after 6 months of prompt payment history.")
            return recs

        # Recommendations for Fair, Poor, or Very Poor
        if duration > 24:
            target_dur = max(12, int(duration * 0.6))
            recs.append(f"Shorten loan tenure from {int(duration)} months down to {target_dur} months to lower lifetime default exposure.")

        if credit_amount > 4000:
            target_amt = round(credit_amount * 0.75, -2)
            recs.append(f"Decrease requested principal from {int(credit_amount)} DM to approximately {int(target_amt)} DM or provide an upfront down payment.")

        if parties == 'A101':
            recs.append("Add a creditworthy co-applicant or guarantor (e.g. family member) to bolster repayment capacity.")

        if checking in ['A11', 'A12']:
            recs.append("Establish a minimum checking deposit buffer of at least 200 DM prior to final disbursement.")

        if savings in ['A61', 'A65']:
            recs.append("Build a verified emergency savings reserve to demonstrate counter-cyclical financial resiliency.")

        if not recs:
            recs.append("Maintain consistent on-time payments across all existing credit obligations to enhance historical score rating.")

        return recs[:4]

    def simulate_what_if(self, base_applicant, modifications):
        """
        Calculates score changes under hypothetical borrower adjustments:
        e.g., reducing loan duration, lowering requested amount, adding a guarantor.
        """
        modified = dict(base_applicant)
        modified.update(modifications)
        
        base_eval = self.evaluate_applicant(base_applicant)
        mod_eval = self.evaluate_applicant(modified)
        
        score_diff = mod_eval['credit_score'] - base_eval['credit_score']
        prob_diff = mod_eval['default_probability'] - base_eval['default_probability']
        
        return {
            'original_score': base_eval['credit_score'],
            'simulated_score': mod_eval['credit_score'],
            'score_change': score_diff,
            'original_tier': base_eval['risk_tier'],
            'simulated_tier': mod_eval['risk_tier'],
            'original_decision': base_eval['decision'],
            'simulated_decision': mod_eval['decision'],
            'original_prob': base_eval['default_probability'],
            'simulated_prob': mod_eval['default_probability'],
            'prob_change': round(prob_diff, 2),
            'positive_factors': mod_eval['positive_factors'],
            'risk_flags': mod_eval['risk_flags']
        }
