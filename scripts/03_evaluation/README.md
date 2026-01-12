# 03_evaluation - Model Evaluation & Optimization Scripts

## 🎯 Overview

This folder contains scripts for evaluating models, optimizing ensemble weights, and validating predictions.

---

## 📁 Files

### `calculate_ensemble_weights.py`

**Purpose**: Calculate optimal ensemble weights using grid search

**Method**:
- Exhaustive grid search over weight combinations
- Validation set CCC as optimization metric
- Tests all model combinations (2-model to 5-model ensembles)

**Algorithm**:
```python
# Grid search parameters
weight_step = 0.01  # 1% increments
constraints: sum(weights) = 1.0

# Optimization target
maximize: CCC_overall = (CCC_valence + CCC_arousal) / 2
```

**Key Findings**:
- **Best: 2-model ensemble** (seed777 + arousal_specialist)
  - CCC: 0.6833
  - Weights: 50.16% / 49.84%
- **3-model ensemble** with seed888 → CCC 0.6729 (worse)
- **Conclusion**: More models ≠ better performance

**Usage**:
```bash
python calculate_ensemble_weights.py
```

**Output**: `results/subtask2a/optimal_ensemble.json`

---

### `optimize_stacking.py`

**Purpose**: Test stacking ensemble methods with meta-learners

**Approaches Tested**:
1. **Linear Regression** (Ridge, Lasso)
2. **Non-linear** (Random Forest, XGBoost)
3. **Weighted Average** (grid search)

**Process**:
```
Train Set → Base Models → Meta-features
Validation Set → Meta-learner Training
Test Set → Final Predictions
```

**Results**:
- Linear stacking: CCC 0.6687
- Non-linear stacking: CCC 0.6729
- **Weighted average**: CCC 0.6833 ✅ (Best)

**Finding**: Simple weighted average outperforms complex meta-learning

**Usage**:
```bash
python optimize_stacking.py
```

---

### `validate_predictions.py`

**Purpose**: Validate prediction file format and content

**Checks**:
1. **Format Validation**
   - CSV structure (user, timestamp, valence, arousal)
   - Required columns present
   - No missing values

2. **Content Validation**
   - Prediction ranges: valence [-1, 1], arousal [-1, 1]
   - User IDs match test data
   - Timestamp sequences complete
   - Total prediction count (46 users × timestamps)

3. **Statistical Checks**
   - Distribution analysis
   - Outlier detection
   - Correlation with baseline

**Usage**:
```bash
python validate_predictions.py pred_subtask2a.csv
```

**Output**:
```
✅ Format: Valid
✅ Content: Valid
✅ Ranges: Valid (Valence: [-0.98, 0.95], Arousal: [-0.87, 0.92])
✅ Users: 46/46 complete
✅ Predictions: 1,266 total
```

---

### `verify_test_data.py`

**Purpose**: Verify test data integrity before prediction generation

**Verifications**:
1. **File Existence**
   - test_subtask2a.csv present
   - Required columns available

2. **Data Integrity**
   - User count (expected: 46)
   - Timestamp sequences
   - Feature completeness

3. **Compatibility**
   - Feature dimensions match training data
   - User ID format consistency

**Usage**:
```bash
python verify_test_data.py
```

**Output**:
```
✅ Test file found: data/test/test_subtask2a.csv
✅ Users: 46
✅ Features: Complete
✅ Timestamps: Valid sequences
✅ Ready for prediction generation
```

---

## 🚀 Quick Start

### Complete Evaluation Workflow

```bash
# Step 1: Verify test data
python verify_test_data.py

# Step 2: Calculate optimal ensemble weights
python calculate_ensemble_weights.py

# Step 3: Test stacking methods (optional)
python optimize_stacking.py

# Step 4: Generate predictions (in ../02_prediction/)
cd ../02_prediction
python predict_optimized.py

# Step 5: Validate predictions
cd ../03_evaluation
python validate_predictions.py ../../pred_subtask2a.csv
```

---

## 📊 Ensemble Performance Analysis

### Model Comparison

| Model Combination | CCC | Weights |
|-------------------|-----|---------|
| **seed777 + arousal_specialist** | **0.6833** | 50.16% / 49.84% ✅ |
| seed777 + seed888 | 0.6687 | 55% / 45% |
| seed777 + seed888 + arousal | 0.6729 | 40% / 30% / 30% |
| All 5 models | 0.6654 | Various |

### Key Insights

1. **2-model is optimal**: Adding more models dilutes performance
2. **Arousal specialist critical**: +6% improvement in Arousal CCC
3. **Balanced weights**: Near 50:50 split indicates complementary strengths
4. **Simplicity wins**: Simple weighted average beats complex meta-learning

---

## 🔍 Performance Metrics

### Expected Results

**Conservative Estimate**:
- Overall CCC: 0.6733
- Arousal CCC: 0.5700
- Valence CCC: 0.7766

**Expected**:
- Overall CCC: 0.6833
- Arousal CCC: 0.5832
- Valence CCC: 0.7834

**Optimistic**:
- Overall CCC: 0.6933
- Arousal CCC: 0.5950
- Valence CCC: 0.7916

**Target**: CCC ≥ 0.62 ✅ (All scenarios exceed target by 8-11%)

---

## 📖 References

See project documentation:
- [PROJECT_STATUS.md](../../docs/PROJECT_STATUS.md) - Performance tracking
- [TRAINING_STRATEGY.md](../../docs/TRAINING_STRATEGY.md) - Ensemble strategy
- [FINAL_REPORT.md](../../docs/FINAL_REPORT.md) - Complete analysis
- [results/subtask2a/optimal_ensemble.json](../../results/subtask2a/optimal_ensemble.json) - Optimal weights

---

**Last Updated**: 2026-01-12
