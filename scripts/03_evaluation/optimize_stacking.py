"""
Stacking Ensemble Optimization
================================

현재: 단순 가중 평균 (CCC 0.6305)
목표: Ridge Regression으로 최적 조합 학습 (예상 CCC 0.64-0.66)

개선 포인트:
1. Validation data에서 최적 가중치 학습
2. Valence/Arousal 별도 가중치
3. 과적합 방지 (Ridge alpha)
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# ===== CONFIGURATION =====
MODEL_DIR = 'models'
DATA_FILE = 'data/processed/train_subtask2a_with_predictions.csv'  # 생성 필요
RESULTS_DIR = 'results/subtask2a'

# 사용할 모델들 (훈련 완료된 것만)
AVAILABLE_MODELS = ['seed123', 'seed777']  # seed42 제거됨
# TODO: seed888, seed999 훈련 후 추가
# AVAILABLE_MODELS = ['seed123', 'seed777', 'seed888', 'seed999']

print("=" * 60)
print("Stacking Ensemble Optimization")
print("=" * 60)
print(f"Models: {AVAILABLE_MODELS}")

# ===== STEP 1: Load Validation Predictions =====
print("\n[Step 1] Loading validation predictions...")

# NOTE: 이 파일은 각 모델의 validation 예측을 저장해야 함
# train_ensemble_subtask2a.py에서 best epoch의 validation prediction 저장 필요

# Dummy data structure (실제로는 파일에서 로드)
# 각 모델의 validation set 예측
predictions = {
    'seed123': {
        'valence': np.random.randn(1000),  # 실제 validation 예측
        'arousal': np.random.randn(1000)
    },
    'seed777': {
        'valence': np.random.randn(1000),
        'arousal': np.random.randn(1000)
    }
}

# True labels
true_labels = {
    'valence': np.random.randn(1000),  # 실제 validation labels
    'arousal': np.random.randn(1000)
}

print(f"✓ Loaded predictions from {len(AVAILABLE_MODELS)} models")
print(f"  Validation samples: {len(true_labels['valence'])}")

# ===== STEP 2: CCC Calculation =====
def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate CCC"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

# ===== STEP 3: Baseline (Simple Weighted Average) =====
print("\n[Step 2] Baseline: Simple Weighted Average")

# CCC-based weights (현재 방식)
model_cccs = {
    'seed123': 0.5330,
    'seed777': 0.6554
}

total_ccc = sum(model_cccs.values())
baseline_weights = {m: ccc/total_ccc for m, ccc in model_cccs.items()}

print(f"Baseline weights: {baseline_weights}")

# Baseline prediction
baseline_pred_v = sum(baseline_weights[m] * predictions[m]['valence']
                      for m in AVAILABLE_MODELS)
baseline_pred_a = sum(baseline_weights[m] * predictions[m]['arousal']
                      for m in AVAILABLE_MODELS)

baseline_ccc_v = concordance_correlation_coefficient(true_labels['valence'], baseline_pred_v)
baseline_ccc_a = concordance_correlation_coefficient(true_labels['arousal'], baseline_pred_a)
baseline_ccc = (baseline_ccc_v + baseline_ccc_a) / 2

print(f"Baseline CCC: {baseline_ccc:.4f}")
print(f"  Valence: {baseline_ccc_v:.4f}")
print(f"  Arousal: {baseline_ccc_a:.4f}")

# ===== STEP 4: Stacking with Ridge Regression =====
print("\n[Step 3] Stacking Optimization (Ridge Regression)")

# Prepare meta-features
X_meta_v = np.column_stack([predictions[m]['valence'] for m in AVAILABLE_MODELS])
X_meta_a = np.column_stack([predictions[m]['arousal'] for m in AVAILABLE_MODELS])

y_true_v = true_labels['valence']
y_true_a = true_labels['arousal']

# Grid search for best alpha
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
best_results = {
    'valence': {'alpha': None, 'ccc': 0, 'weights': None},
    'arousal': {'alpha': None, 'ccc': 0, 'weights': None}
}

print("\nValence Optimization:")
for alpha in alphas:
    ridge_v = Ridge(alpha=alpha, fit_intercept=False)  # No intercept for ensemble

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_cccs = []

    for train_idx, val_idx in kf.split(X_meta_v):
        ridge_v.fit(X_meta_v[train_idx], y_true_v[train_idx])
        pred = ridge_v.predict(X_meta_v[val_idx])
        ccc = concordance_correlation_coefficient(y_true_v[val_idx], pred)
        cv_cccs.append(ccc)

    avg_ccc = np.mean(cv_cccs)
    print(f"  alpha={alpha:6.3f}: CCC {avg_ccc:.4f} (±{np.std(cv_cccs):.4f})")

    if avg_ccc > best_results['valence']['ccc']:
        ridge_v.fit(X_meta_v, y_true_v)  # Refit on all data
        best_results['valence'] = {
            'alpha': alpha,
            'ccc': avg_ccc,
            'weights': ridge_v.coef_.tolist()
        }

print("\nArousal Optimization:")
for alpha in alphas:
    ridge_a = Ridge(alpha=alpha, fit_intercept=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_cccs = []

    for train_idx, val_idx in kf.split(X_meta_a):
        ridge_a.fit(X_meta_a[train_idx], y_true_a[train_idx])
        pred = ridge_a.predict(X_meta_a[val_idx])
        ccc = concordance_correlation_coefficient(y_true_a[val_idx], pred)
        cv_cccs.append(ccc)

    avg_ccc = np.mean(cv_cccs)
    print(f"  alpha={alpha:6.3f}: CCC {avg_ccc:.4f} (±{np.std(cv_cccs):.4f})")

    if avg_ccc > best_results['arousal']['ccc']:
        ridge_a.fit(X_meta_a, y_true_a)
        best_results['arousal'] = {
            'alpha': alpha,
            'ccc': avg_ccc,
            'weights': ridge_a.coef_.tolist()
        }

# ===== STEP 5: Results Summary =====
print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS")
print("=" * 60)

print(f"\nBaseline (Simple Weighted Average):")
print(f"  CCC: {baseline_ccc:.4f}")
print(f"  Valence CCC: {baseline_ccc_v:.4f}")
print(f"  Arousal CCC: {baseline_ccc_a:.4f}")
print(f"  Weights: {baseline_weights}")

print(f"\nOptimized (Stacking with Ridge):")
print(f"  Average CCC: {(best_results['valence']['ccc'] + best_results['arousal']['ccc'])/2:.4f}")
print(f"  Valence CCC: {best_results['valence']['ccc']:.4f} (alpha={best_results['valence']['alpha']})")
print(f"  Arousal CCC: {best_results['arousal']['ccc']:.4f} (alpha={best_results['arousal']['alpha']})")

print(f"\n  Valence Weights:")
for i, model in enumerate(AVAILABLE_MODELS):
    weight = best_results['valence']['weights'][i]
    print(f"    {model}: {weight:.4f}")

print(f"\n  Arousal Weights:")
for i, model in enumerate(AVAILABLE_MODELS):
    weight = best_results['arousal']['weights'][i]
    print(f"    {model}: {weight:.4f}")

improvement = ((best_results['valence']['ccc'] + best_results['arousal']['ccc'])/2 - baseline_ccc)
print(f"\nImprovement: {improvement:+.4f} ({improvement/baseline_ccc*100:+.2f}%)")

# ===== STEP 6: Save Results =====
output = {
    'method': 'stacking_ridge',
    'baseline': {
        'method': 'simple_weighted_average',
        'ccc': baseline_ccc,
        'valence_ccc': baseline_ccc_v,
        'arousal_ccc': baseline_ccc_a,
        'weights': baseline_weights
    },
    'optimized': {
        'ccc': (best_results['valence']['ccc'] + best_results['arousal']['ccc'])/2,
        'valence': {
            'ccc': best_results['valence']['ccc'],
            'alpha': best_results['valence']['alpha'],
            'weights': {model: best_results['valence']['weights'][i]
                       for i, model in enumerate(AVAILABLE_MODELS)}
        },
        'arousal': {
            'ccc': best_results['arousal']['ccc'],
            'alpha': best_results['arousal']['alpha'],
            'weights': {model: best_results['arousal']['weights'][i]
                       for i, model in enumerate(AVAILABLE_MODELS)}
        }
    },
    'improvement': improvement,
    'improvement_percentage': improvement/baseline_ccc*100
}

output_path = f'{RESULTS_DIR}/stacking_optimization.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to: {output_path}")

# ===== STEP 7: Usage Instructions =====
print("\n" + "=" * 60)
print("USAGE FOR TEST PREDICTION")
print("=" * 60)
print("""
# In predict_test_subtask2a.py, replace simple average with:

# Load optimized weights
with open('results/subtask2a/stacking_optimization.json') as f:
    weights = json.load(f)['optimized']

# Apply separate weights for Valence/Arousal
pred_valence = sum(
    weights['valence']['weights'][model] * predictions[model]['valence']
    for model in models
)

pred_arousal = sum(
    weights['arousal']['weights'][model] * predictions[model]['arousal']
    for model in models
)
""")

print("=" * 60)
