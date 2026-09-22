"""
Credit Scoring & Risk Assessment System
Data Preprocessing & Schema Definitions for UCI German Credit Dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# German Credit Dataset Columns based on german.doc
COLUMN_NAMES = [
    'checking_status',       # Att 1: Status of existing checking account (qualitative)
    'duration',              # Att 2: Duration in months (numerical)
    'credit_history',        # Att 3: Credit history (qualitative)
    'purpose',               # Att 4: Purpose (qualitative)
    'credit_amount',         # Att 5: Credit amount (numerical)
    'savings_status',        # Att 6: Savings account/bonds (qualitative)
    'employment',            # Att 7: Present employment since (qualitative)
    'installment_rate',      # Att 8: Installment rate in % of disposable income (numerical)
    'personal_status',       # Att 9: Personal status and sex (qualitative)
    'other_parties',         # Att 10: Other debtors / guarantors (qualitative)
    'residence_since',       # Att 11: Present residence since in years (numerical)
    'property_magnitude',    # Att 12: Property (qualitative)
    'age',                   # Att 13: Age in years (numerical)
    'other_payment_plans',   # Att 14: Other installment plans (qualitative)
    'housing',               # Att 15: Housing (qualitative)
    'existing_credits',      # Att 16: Number of existing credits at this bank (numerical)
    'job',                   # Att 17: Job (qualitative)
    'num_dependents',        # Att 18: Number of people liable to provide maintenance (numerical)
    'own_telephone',         # Att 19: Telephone registered (qualitative)
    'foreign_worker',        # Att 20: Foreign worker (qualitative)
    'class_raw'              # Att 21: 1 = Good, 2 = Bad
]

NUMERICAL_FEATURES = [
    'duration',
    'credit_amount',
    'installment_rate',
    'residence_since',
    'age',
    'existing_credits',
    'num_dependents'
]

CATEGORICAL_FEATURES = [
    'checking_status',
    'credit_history',
    'purpose',
    'savings_status',
    'employment',
    'personal_status',
    'other_parties',
    'property_magnitude',
    'other_payment_plans',
    'housing',
    'job',
    'own_telephone',
    'foreign_worker'
]

# Human-readable dictionary mapping raw codes to intuitive descriptions
FEATURE_DESCRIPTIONS = {
    'checking_status': {
        'A11': '< 0 DM (Overdrawn / Negative)',
        'A12': '0 <= ... < 200 DM (Low Balance)',
        'A13': '>= 200 DM / Salary Account (Moderate to High)',
        'A14': 'No Checking Account'
    },
    'credit_history': {
        'A30': 'No credits taken / all paid back duly',
        'A31': 'All credits at this bank paid back duly',
        'A32': 'Existing credits paid back duly till now',
        'A33': 'Delay in paying off in the past',
        'A34': 'Critical account / other credits existing'
    },
    'purpose': {
        'A40': 'Car (new)',
        'A41': 'Car (used)',
        'A42': 'Furniture / equipment',
        'A43': 'Radio / television',
        'A44': 'Domestic appliances',
        'A45': 'Repairs',
        'A46': 'Education',
        'A47': 'Vacation',
        'A48': 'Retraining',
        'A49': 'Business',
        'A410': 'Others'
    },
    'savings_status': {
        'A61': '< 100 DM (Minimal)',
        'A62': '100 <= ... < 500 DM (Modest)',
        'A63': '500 <= ... < 1000 DM (Intermediate)',
        'A64': '>= 1000 DM (Substantial)',
        'A65': 'Unknown / No Savings Account'
    },
    'employment': {
        'A71': 'Unemployed',
        'A72': '< 1 year',
        'A73': '1 <= ... < 4 years',
        'A74': '4 <= ... < 7 years',
        'A75': '>= 7 years (Tenured)'
    },
    'personal_status': {
        'A91': 'Male : Divorced / Separated',
        'A92': 'Female : Divorced / Separated / Married',
        'A93': 'Male : Single',
        'A94': 'Male : Married / Widowed',
        'A95': 'Female : Single'
    },
    'other_parties': {
        'A101': 'None',
        'A102': 'Co-applicant',
        'A103': 'Guarantor'
    },
    'property_magnitude': {
        'A121': 'Real Estate (Home / Land)',
        'A122': 'Building Society / Life Insurance',
        'A123': 'Car or Other Assets',
        'A124': 'Unknown / No Property'
    },
    'other_payment_plans': {
        'A141': 'Bank',
        'A142': 'Stores',
        'A143': 'None'
    },
    'housing': {
        'A151': 'Rent',
        'A152': 'Own',
        'A153': 'For Free'
    },
    'job': {
        'A171': 'Unemployed / Unskilled - Non-resident',
        'A172': 'Unskilled - Resident',
        'A173': 'Skilled Employee / Official',
        'A174': 'Management / Self-employed / Highly Qualified'
    },
    'own_telephone': {
        'A191': 'None',
        'A192': 'Yes (Registered under customer name)'
    },
    'foreign_worker': {
        'A201': 'Yes',
        'A202': 'No'
    }
}


def load_raw_data(data_path='german.data'):
    """
    Loads raw german.data file, applies column names, and maps target variable:
    class_raw: 1 (Good) -> 0 (Low Default Risk), 2 (Bad) -> 1 (High Default Risk)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    
    # Target mapping:
    # In standard credit risk modeling, target = 1 denotes Default / Bad Credit,
    # target = 0 denotes Non-Default / Good Credit.
    df['risk_label'] = df['class_raw'].map({1: 0, 2: 1}).astype(int)
    
    return df


def build_preprocessor():
    """
    Constructs a ColumnTransformer with RobustScaler for numerical features
    and OneHotEncoder for categorical features.
    """
    num_transformer = Pipeline([
        ('scaler', RobustScaler())
    ])
    
    cat_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERICAL_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    return preprocessor


def prepare_datasets(data_path='german.data', test_size=0.2, random_state=42):
    """
    Loads data, splits into stratified train/test sets, fits preprocessor on train,
    and returns processed sets with feature names.
    """
    df = load_raw_data(data_path)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['risk_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = list(preprocessor.get_feature_names_out())
    
    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'full_df': df
    }


if __name__ == '__main__':
    print("Testing data preprocessing pipeline...")
    data = prepare_datasets()
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"Total features extracted: {len(data['feature_names'])}")
    print(f"Class distribution in train: {np.bincount(data['y_train'])}")
    print(f"Class distribution in test: {np.bincount(data['y_test'])}")
    print("Data preprocessing module is working properly.")
