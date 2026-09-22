"""
Credit Scoring & Risk Assessment System
Model Training, Cross-Validation & Cost Matrix Optimization
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
import joblib

# Official German Credit Cost Matrix:
# Cost of granting credit to Bad customer (FN, where 1=Bad) = 5
# Cost of rejecting Good customer (FP, where 0=Good) = 1
COST_FN = 5.0
COST_FP = 1.0


def calculate_financial_cost(y_true, y_pred):
    """
    Computes total and average cost based on the 5:1 penalty matrix:
    - y_true == 1 and y_pred == 0: False Negative (Cost = 5)
    - y_true == 0 and y_pred == 1: False Positive (Cost = 1)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout:
    # TN = cm[0, 0], FP = cm[0, 1]
    # FN = cm[1, 0], TP = cm[1, 1]
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fn * COST_FN) + (fp * COST_FP)
    avg_cost = total_cost / len(y_true)
    return total_cost, avg_cost, {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def optimize_threshold_for_cost(y_true, y_prob):
    """
    Sweeps decision threshold from 0.05 to 0.95 to find the cutoff that minimizes
    the financial cost given the 5:1 asymmetric penalty.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        _, avg_cost, _ = calculate_financial_cost(y_true, preds)
        costs.append(avg_cost)
        
    best_idx = int(np.argmin(costs))
    optimal_threshold = float(thresholds[best_idx])
    min_cost = float(costs[best_idx])
    
    # Also evaluate default 0.5 threshold
    preds_default = (y_prob >= 0.5).astype(int)
    _, default_cost, _ = calculate_financial_cost(y_true, preds_default)
    
    cost_curve = [{'threshold': round(float(t), 2), 'cost': round(float(c), 3)} for t, c in zip(thresholds, costs)]
    
    return {
        'optimal_threshold': round(optimal_threshold, 2),
        'min_cost_per_applicant': round(min_cost, 3),
        'default_cost_per_applicant': round(default_cost, 3),
        'cost_reduction_pct': round(((default_cost - min_cost) / default_cost) * 100, 1) if default_cost > 0 else 0.0,
        'cost_curve': cost_curve
    }


def evaluate_model_performance(model, X_test, y_test, optimal_threshold=0.5):
    """
    Computes all standard statistical metrics and cost matrix metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    
    total_cost_def, avg_cost_def, cm_def = calculate_financial_cost(y_test, y_pred_default)
    total_cost_opt, avg_cost_opt, cm_opt = calculate_financial_cost(y_test, y_pred_opt)
    
    # ROC curve
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    roc_points = [{'fpr': round(float(f), 4), 'tpr': round(float(t), 4)} for f, t in zip(fpr, tpr)][::max(1, len(fpr)//50)]
    
    # PR curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_points = [{'precision': round(float(p), 4), 'recall': round(float(r), 4)} for p, r in zip(precision_curve, recall_curve)][::max(1, len(precision_curve)//50)]
    
    # Specificity: True Negative Rate for Good borrowers
    tn = cm_def['tn']
    fp = cm_def['fp']
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': round(accuracy_score(y_test, y_pred_default), 4),
        'precision': round(precision_score(y_test, y_pred_default, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred_default, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred_default, zero_division=0), 4),
        'specificity': round(specificity, 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
        'pr_auc': round(average_precision_score(y_test, y_prob), 4),
        'brier_score': round(brier_score_loss(y_test, y_prob), 4),
        'default_cm': cm_def,
        'default_avg_cost': round(avg_cost_def, 3),
        'optimal_cm': cm_opt,
        'optimal_avg_cost': round(avg_cost_opt, 3),
        'optimal_accuracy': round(accuracy_score(y_test, y_pred_opt), 4),
        'optimal_recall': round(recall_score(y_test, y_pred_opt, zero_division=0), 4),
        'roc_curve': roc_points,
        'pr_curve': pr_points,
        'y_prob': y_prob
    }


def train_and_evaluate_all_models(data_dict, models_dir='models'):
    """
    Trains multiple models, performs 5-Fold Stratified CV, optimizes thresholds,
    and stores evaluation benchmarks and serialized models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']
    
    # Candidate models
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            C=0.5,
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=4,
            random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=9,
            class_weight='balanced',
            random_state=42
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted_models = {}
    
    print("Beginning model training and 5-Fold Stratified Cross-Validation...")
    
    for name, model in candidate_models.items():
        print(f"\nEvaluating {name}...")
        
        # Cross-validation metrics
        cv_rocs = []
        cv_f1s = []
        cv_recalls = []
        cv_costs = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            val_probs = model.predict_proba(X_fold_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            
            cv_rocs.append(roc_auc_score(y_fold_val, val_probs))
            cv_f1s.append(f1_score(y_fold_val, val_preds, zero_division=0))
            cv_recalls.append(recall_score(y_fold_val, val_preds, zero_division=0))
            _, avg_c, _ = calculate_financial_cost(y_fold_val, val_preds)
            cv_costs.append(avg_c)
            
        # Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Optimize threshold using validation predictions or test probabilities
        test_probs = model.predict_proba(X_test)[:, 1]
        cost_opt = optimize_threshold_for_cost(y_test, test_probs)
        opt_thresh = cost_opt['optimal_threshold']
        
        # Test metrics
        test_metrics = evaluate_model_performance(model, X_test, y_test, optimal_threshold=opt_thresh)
        
        # Feature importances / coefficients
        feat_imp = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1]
            feat_imp = {feature_names[i]: round(float(importances[i]), 4) for i in top_indices[:15]}
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            top_indices = np.argsort(coefs)[::-1]
            feat_imp = {feature_names[i]: round(float(coefs[i]), 4) for i in top_indices[:15]}
            
        results[name] = {
            'cv_roc_auc_mean': round(float(np.mean(cv_rocs)), 4),
            'cv_roc_auc_std': round(float(np.std(cv_rocs)), 4),
            'cv_f1_mean': round(float(np.mean(cv_f1s)), 4),
            'cv_recall_mean': round(float(np.mean(cv_recalls)), 4),
            'cv_cost_mean': round(float(np.mean(cv_costs)), 3),
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_specificity': test_metrics['specificity'],
            'test_roc_auc': test_metrics['roc_auc'],
            'test_pr_auc': test_metrics['pr_auc'],
            'test_brier_score': test_metrics['brier_score'],
            'default_cm': test_metrics['default_cm'],
            'default_avg_cost': test_metrics['default_avg_cost'],
            'optimal_threshold': opt_thresh,
            'optimal_cm': test_metrics['optimal_cm'],
            'optimal_avg_cost': test_metrics['optimal_avg_cost'],
            'optimal_accuracy': test_metrics['optimal_accuracy'],
            'optimal_recall': test_metrics['optimal_recall'],
            'cost_reduction_pct': cost_opt['cost_reduction_pct'],
            'cost_curve': cost_opt['cost_curve'],
            'roc_curve': test_metrics['roc_curve'],
            'pr_curve': test_metrics['pr_curve'],
            'top_features': feat_imp
        }
        
        print(f"  CV ROC-AUC: {results[name]['cv_roc_auc_mean']} +/- {results[name]['cv_roc_auc_std']}")
        print(f"  Test ROC-AUC: {results[name]['test_roc_auc']} | Test Recall: {results[name]['test_recall']}")
        print(f"  Default Cost: {results[name]['default_avg_cost']} -> Optimal Cost ({opt_thresh}): {results[name]['optimal_avg_cost']}")
        
    # Select champion model based on highest ROC-AUC and financial cost efficiency
    champion_name = max(results.keys(), key=lambda k: results[k]['test_roc_auc'])
    champion_model = fitted_models[champion_name]
    
    print(f"\nChampion Model Selected: {champion_name} (ROC-AUC: {results[champion_name]['test_roc_auc']})")
    
    # Save artifacts
    model_path = os.path.join(models_dir, 'credit_risk_model.joblib')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    all_models_path = os.path.join(models_dir, 'all_models.joblib')
    
    joblib.dump(champion_model, model_path)
    joblib.dump(data_dict['preprocessor'], preprocessor_path)
    joblib.dump(fitted_models, all_models_path)
    
    summary = {
        'champion_model': champion_name,
        'models': results,
        'feature_names': feature_names
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Artifacts successfully saved to {models_dir}/")
    return summary, champion_model
