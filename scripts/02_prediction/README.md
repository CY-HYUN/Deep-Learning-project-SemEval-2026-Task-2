# 02_prediction - Prediction Generation Scripts

## 🎯 Overview

This folder contains scripts for generating predictions using trained models.

---

## 📁 Files

### `predict_optimized.py`

**Purpose**: Generate predictions for test data using optimal 2-model ensemble

**Ensemble Configuration**:
```python
{
  "models": ["seed777", "arousal_specialist"],
  "weights": {
    "seed777": 0.5016,              # 50.16%
    "arousal_specialist": 0.4984    # 49.84%
  },
  "expected_ccc": "0.6733-0.6933 (avg 0.6833)"
}
```

**Key Features**:
- Dynamic input dimension handling (863 vs 866 features)
- Weighted ensemble predictions
- Automatic CSV generation for Codabench submission

**Usage**:
```bash
python predict_optimized.py
```

**Output**: `pred_subtask2a.csv` (46 test users)

---

### `predict_notebook.ipynb`

**Purpose**: Jupyter notebook version for interactive prediction

**When to Use**:
- Local development with Jupyter
- Step-by-step debugging
- Visualization of predictions

---

### `run_prediction_colab.ipynb`

**Purpose**: Self-contained Google Colab notebook for prediction generation (⭐ Production version)

**Features**:
- Complete 9-step pipeline
- Handles feature dimension mismatches (864→863→866)
- Automatic file downloads
- Google Drive integration

**Workflow**:
1. Setup environment
2. Upload models and data
3. Install dependencies
4. Load models (seed777 + arousal_specialist)
5. Process test data
6. Generate predictions
7. Save CSV output
8. Download results
9. Create submission.zip

**Technical Highlights**:
- Dynamic feature slicing for different model architectures
- Handles user_stats dimension differences (12 vs 15 features)
- Google Colab Pro A100 GPU support
- Execution time: ~35 minutes

**Usage**:
1. Open in Google Colab
2. Set runtime to GPU (T4 or A100)
3. Execute all cells sequentially
4. Download `pred_subtask2a.csv` and `submission.zip`

---

## 🚀 Quick Start

### Google Colab (Recommended)

```bash
# 1. Open run_prediction_colab.ipynb in Google Colab
# 2. Upload required files:
#    - models/subtask2a_seed777_best.pt
#    - models/subtask2a_arousal_specialist_seed1111_best.pt
#    - data/test/test_subtask2a.csv
#    - results/subtask2a/optimal_ensemble.json
# 3. Run all cells
# 4. Download pred_subtask2a.csv
```

### Local Execution

```bash
# Requirements
pip install torch transformers pandas numpy scikit-learn

# Generate predictions
python predict_optimized.py

# Output: pred_subtask2a.csv
```

---

## 📊 Expected Output

### File Format: `pred_subtask2a.csv`

```csv
user,timestamp,valence,arousal
USER_001,0,-0.123,0.456
USER_001,1,-0.089,0.389
...
```

**Specifications**:
- 46 test users
- 1,266 bytes file size
- Valence & Arousal predictions per timestep
- Ready for Codabench submission

---

## 🔧 Technical Details

### Dynamic Dimension Handling

The prediction pipeline automatically handles different input dimensions:

```python
# seed777: 768 + 64 + 5 + 12 + 14 = 863 features
# arousal_specialist: 768 + 64 + 5 + 15 + 14 = 866 features

# Dynamic slicing in model forward pass
user_stats_needed = self.input_dim - 768 - 64 - 5 - 14
user_stats_sliced = user_stats[:, :user_stats_needed]
```

### Ensemble Weighting

```python
# Weighted average of predictions
final_pred = (
    seed777_pred * 0.5016 +
    arousal_pred * 0.4984
)
```

---

## 📖 References

See project documentation:
- [PROJECT_STATUS.md](../../docs/PROJECT_STATUS.md) - Current project status
- [FINAL_REPORT.md](../../docs/FINAL_REPORT.md) - Section 11.6: Google Colab Prediction Pipeline
- [NEXT_ACTIONS.md](../../docs/NEXT_ACTIONS.md) - Submission guide

---

**Last Updated**: 2026-01-12
