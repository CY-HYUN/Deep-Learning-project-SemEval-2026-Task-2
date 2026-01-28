# Demo Visualizations for SemEval 2026 Task 2a

This folder contains visualizations for the SemEval 2026 Task 2a project, including:
- **Demo notebook visualizations** (3 files, Jan 28) - Used in live presentation
- **Additional project visualizations** (5 files, Jan 21) - Supplementary analysis

---

## 📊 Part 1: Demo Notebook Visualizations (Live Presentation)

### 1. User 137 Emotional Timeline
**File**: `01_user137_emotional_timeline.png` (134 KB, 2085×1035 px)

**Source**: Demo notebook Cell 13 ([demo_live_presentation.ipynb](../scripts/demo/demo_live_presentation.ipynb))

**Description**:
- 2 subplots showing User 137's emotional journey over 3 years (42 entries, Jan 2021 - Dec 2023)
- **Top plot**: Valence timeline (blue line) with mean reference line (dashed)
- **Bottom plot**: Arousal timeline (red line) with mean reference line (dashed)
- Demonstrates temporal patterns: Valence upward trend, Arousal volatility

**Used in Presentation**:
- Part 2, Demo Step 1 (8:10-9:00, 50 seconds)
- Shows historical data before making predictions

---

### 2. Prediction Results (Timeline + Circumplex)
**File**: `02_prediction_results_combined.png` (238 KB, 2325×885 px)

**Source**: Demo notebook Cell 19

**Description**:
- Combined visualization with 2 side-by-side plots
- **Left plot**: Emotional trajectory with forecast
  - Historical Valence (blue line) and Arousal (red line)
  - Predicted values marked with large stars (★)
  - Vertical gray line separating historical vs forecast
- **Right plot**: Russell's Circumplex Model
  - Historical entries as colored dots (viridis colormap = time progression)
  - Predicted point as gold star (★)
  - 4 quadrants labeled: Anxious/Tense, Excited/Alert, Sad/Depressed, Calm/Content
  - Colorbar showing time progression

**Used in Presentation**:
- Part 2, Demo Step 4 (10:25-11:15, 50 seconds) ⭐ **CRITICAL SECTION**
- Demonstrates model's forecasting capability and emotional state interpretation

---

### 3. Model Contribution Analysis
**File**: `03_model_contribution_analysis.png` (68 KB, 2085×735 px)

**Source**: Demo notebook Cell 21

**Description**:
- 2 side-by-side bar charts comparing model predictions
- **Left chart**: Valence predictions (blue tones)
  - seed777, arousal_specialist, Ensemble
  - Red dashed line = last observed value (0.732)
- **Right chart**: Arousal predictions (red/coral tones)
  - seed777, arousal_specialist, Ensemble
  - Red dashed line = last observed value (0.466)
- Value labels on top of each bar (3 decimal places)

**Used in Presentation**:
- Not explicitly covered in 10-12 minute script (optional if time permits)
- Shows complementary strengths of 2-model ensemble
- Demonstrates balanced contribution (50:50 weighting)

---

## 📊 Part 2: Additional Project Visualizations (Supplementary)

### 4. Russell's Circumplex Model (Standalone)
**File**: `01_russells_circumplex.png` (418 KB, 2889×2975 px)

**Source**: Training/analysis scripts (created Jan 21)

**Description**:
- High-resolution standalone Russell's Circumplex visualization
- Larger and more detailed than the combined version in `02_prediction_results_combined.png`
- Shows emotional state distribution across 4 quadrants
- Useful for presentations requiring large, detailed Circumplex diagram

**Difference from demo version**:
- Higher resolution (2889×2975 vs 2325×885)
- Single-plot focus (vs combined with timeline)
- Possibly different data visualization (training set vs User 137 demo)

---

### 5. Scatter Density Analysis
**File**: `02_scatter_density.png` (334 KB, 2966×2375 px)

**Source**: Training/analysis scripts (created Jan 21)

**Description**:
- Scatter plot with density visualization
- Likely shows distribution of Valence/Arousal predictions or training data
- High-resolution visualization for detailed analysis

**Use Case**:
- Data distribution analysis
- Model performance visualization
- Training dataset exploration

---

### 6. Model Comparison (Extended)
**File**: `03_model_comparison.png` (233 KB, 3570×2072 px)

**Source**: Training/analysis scripts (created Jan 21)

**Description**:
- Extended model comparison visualization
- Larger and more detailed than `03_model_contribution_analysis.png`
- May include additional models or metrics beyond the 2-model ensemble

**Difference from demo version**:
- Higher resolution (3570×2072 vs 2085×735)
- Possibly includes more models (3-model, 5-model ensembles tested)
- May show additional metrics beyond Valence/Arousal predictions

---

### 7. Feature Importance Analysis
**File**: `04_feature_importance.png` (269 KB, 4017×1774 px)

**Source**: Training/analysis scripts (created Jan 21)

**Description**:
- Feature importance visualization showing contribution of 47 features
- Likely uses SHAP values, permutation importance, or similar method
- Wide format (4017×1774) suggests horizontal bar chart

**Feature Categories**:
- 20 Temporal features (lag, rolling stats, arousal-specific)
- 15 Text features (length, sentiment, lexical)
- 12 User features (mean, std, historical statistics)

**Use Case**:
- Understanding which features drive predictions
- Model interpretability
- Feature engineering validation

---

### 8. Training Progress
**File**: `05_training_progress.png` (230 KB, 3570×1774 px)

**Source**: Training/analysis scripts (created Jan 21)

**Description**:
- Training progress visualization (loss curves, metrics over epochs)
- Likely shows CCC score improvement during training
- May include separate curves for Valence and Arousal

**Typical Content**:
- Training loss vs validation loss
- CCC score progression
- Epoch-by-epoch performance
- Possibly shows comparison between different seeds (42, 123, 777)

**Use Case**:
- Training diagnostics
- Convergence analysis
- Hyperparameter tuning documentation

---

## 📁 File Organization Summary

### Demo Notebook Files (3 files, 439 KB total)
1. `01_user137_emotional_timeline.png` (134 KB) - Cell 13
2. `02_prediction_results_combined.png` (238 KB) - Cell 19 ⭐
3. `03_model_contribution_analysis.png` (68 KB) - Cell 21

### Additional Analysis Files (5 files, 1.5 MB total)
4. `01_russells_circumplex.png` (418 KB) - Standalone Circumplex
5. `02_scatter_density.png` (334 KB) - Data distribution
6. `03_model_comparison.png` (233 KB) - Extended comparison
7. `04_feature_importance.png` (269 KB) - Feature analysis
8. `05_training_progress.png` (230 KB) - Training curves

**Total**: 8 files, ~2 MB

---

## 🔧 Regeneration

To regenerate demo notebook visualizations (files 1-3):

```bash
cd D:\Study\Github\Deep-Learning-project-SemEval-2026-Task-2\scripts\demo
python extract_visualizations.py
```

**Script**: [extract_visualizations.py](../scripts/demo/extract_visualizations.py)

**Output**: All 3 PNG files saved to `demo_visualizations/` folder

---

## 📅 Last Updated

**Date**: January 28, 2026 (08:47)

**Source Notebook**: [demo_live_presentation.ipynb](../scripts/demo/demo_live_presentation.ipynb)

**Presentation Script**: [Live_Demo_Script_EN_Full.md](../docs/03_submission/Live_Demo_Script_EN_Full.md)

---

## 🎯 User 137 Demo Data

All visualizations are based on User 137's emotional diary:

- **Entries**: 42 emotional diary entries
- **Time span**: January 15, 2021 - December 17, 2023 (3 years)
- **Last observed values**:
  - Valence: 0.732 (positive mood)
  - Arousal: 0.466 (moderate energy)
- **Ensemble predictions**:
  - Valence: 0.498 (-0.234, regression toward mean)
  - Arousal: 0.499 (+0.033, slight increase)

**Note**: User 137's diary timestamps (2021-2023) represent the training data being analyzed, while the project research period (November 2024 - January 2026) is when the analysis was conducted.

---

## 🖼️ Visual Specifications

- **Resolution**: 150 DPI (high quality for presentations)
- **Style**: Seaborn whitegrid
- **Format**: PNG with tight bounding box
- **Colors**:
  - Valence: Blue tones (`blue`, `steelblue`)
  - Arousal: Red tones (`red`, `coral`)
  - Predictions: Gold star (`gold`)
  - Historical: Viridis colormap (time progression)

---

**Generated by**: [extract_visualizations.py](../scripts/demo/extract_visualizations.py)
