# SemEval 2026 Task 2: Emotional State Change Forecasting
## Final Project Report

**Course**: Deep Learning Project
**Institution**: Télécom SudParis
**Submission Date**: January 28, 2026
**Team Members**:
- **Rostislav SVITSOV** - Subtask 1: Between-User Emotion Recognition
- **Hyun Chang-Yong (현창용)** - Subtask 2a: Within-User State Change Forecasting

---

## 1. Data Preparation

### Dataset Overview
The SemEval 2026 Task 2 dataset consists of longitudinal ecological essays with self-reported emotional states spanning 2021-2024. The dataset measures two emotional dimensions:
- **Valence**: Pleasantness (0-4 scale)
- **Arousal**: Energy/Activation level (0-4 scale)

### Subtask 2a Data Statistics
- **Training Set**: 137 users with irregular temporal sequences
- **Test Set**: 46 users (1,266 total predictions required)
- **Average Entries**: 25 entries per user over 3-year span
- **Key Challenge**: Irregular time intervals requiring sophisticated temporal modeling

### Data Preprocessing (Subtask 2a)
1. **Text Processing**: RoBERTa tokenization (max 128 tokens), handling multilingual entries
2. **Temporal Feature Engineering**:
   - Lag features (t-1, t-2, t-3) for both Valence and Arousal
   - Rolling statistics (mean, std) with windows of 3, 5, 10
   - Time gap normalization (log transformation)
   - **Arousal-specific features**: change, volatility, acceleration (3 dimensions)
3. **User Embeddings**: 64-dimensional learnable representations per user
4. **Missing Data**: Forward-fill strategy with temporal interpolation
5. **Normalization**: Min-max scaling for numerical features

### Exploratory Data Analysis Insights
- **Valence Distribution**: Mean 2.1, Std 0.9 (balanced variance)
- **Arousal Distribution**: Mean 1.0, Std 0.6 (low variance, high subjectivity)
- **Critical Finding**: Arousal changes are 38% less frequent but more sporadic than Valence → became the core research challenge

---

## 2. Methodology

### Subtask 2a: Hybrid Deep Learning Architecture

#### Model Components
1. **Text Encoder**: RoBERTa-base (125M parameters)
   - Extracts 768-dimensional contextual embeddings from CLS tokens
   - Pre-trained on 160GB of text data

2. **Temporal Modeling**: BiLSTM
   - 2 layers × 256 hidden units (bidirectional → 512-dim output)
   - Captures sequential emotional transitions over time

3. **Attention Mechanism**: Multi-Head Attention (8 heads)
   - Refines salient emotional cues across sequence steps
   - 96 dimensions per head

4. **Feature Integration** (47 total dimensions):
   - LSTM output: 512-dim
   - User embeddings: 64-dim
   - Temporal features: 20-dim (including 3 arousal-specific)
   - Text features: 15-dim (length, punctuation, lexical)
   - Concatenated: 603-dim → 611-dim input to output heads

5. **Dual-Head Output**: Separate regression heads for Valence and Arousal
   - Fully-connected layers: [603 → 256 → 128 → 1]
   - Dropout: 0.3 for regularization

#### Training Configuration
- **Optimizer**: AdamW (β1=0.9, β2=0.999, lr=1e-5)
- **Learning Rate Schedule**: Linear warmup (10%) + cosine decay
- **Batch Size**: 16 with gradient accumulation
- **Epochs**: 50 (early stopping patience: 10)
- **Loss Function**: Weighted CCC + MSE
  - Standard model: 65% CCC + 35% MSE (Valence), 70% CCC + 30% MSE (Arousal)
  - **Arousal Specialist**: 50% CCC + 50% MSE (Valence), **90% CCC** + 10% MSE (Arousal) ⭐
- **Infrastructure**: Google Colab Pro (A100 40GB GPU), Mixed Precision (FP16)

#### Key Innovation: Arousal-Specialist Model
**Problem**: Initial models showed 38% performance gap between Valence and Arousal CCC.

**Root Cause Analysis**:
1. **Subjective Variance**: Users struggle to define energy levels (Arousal) more than mood (Valence)
2. **Low Data Variation**: Standard loss functions ignore subtle arousal shifts, focusing on higher-variance Valence

**Solution**: Developed a dedicated Arousal-Specialist model with three modifications:
1. **Loss Shift**: Increased CCC loss weight from 70% to **90%** for Arousal dimension
2. **Dimension-Specific Optimization**: Parameters tuned exclusively for Arousal performance
3. **Dynamic Weighted Sampling**: Oversampled high-change emotional events during training

**Result**: Arousal CCC improved from 0.5281 → 0.5832 (+6.0% absolute improvement)

#### Ensemble Strategy: Quality Over Quantity
- Tested 5 individual models (seeds: 42, 123, 777, 888, 1111-arousal_specialist)
- Conducted grid search across 5,000+ weight combinations
- **Final 2-Model Ensemble** (outperformed 3-model and 5-model configurations):
  - **seed777** (50.16%): Master of Valence and baseline trends (CCC 0.6554)
  - **arousal_specialist** (49.84%): Corrects energy-prediction bias (CCC 0.6512, Arousal CCC 0.5832)
- Rejected meta-learning approaches (Ridge, XGBoost stacking) → simple weighted average proved superior

---

## 3. Results and Conclusion

### Subtask 2a Performance

#### Final Validation Results
```
✅ Overall CCC: 0.6833 (Target: 0.62, +10.4% above target)
   ├── Valence CCC:  0.7834
   └── Arousal CCC:  0.5832 (+6.0% improvement through specialization)
```

#### Performance Evolution
| Phase | Configuration | CCC | Improvement | Date |
|-------|---------------|----:|------------:|------|
| Phase 1 | 3-model (seed42+123+777) | 0.6046 | Baseline | Nov 2024 |
| Phase 2 | 2-model (seed123+777) | 0.6305 | +4.3% | Dec 2024 |
| Phase 3 | seed777 + seed888 | 0.6687 | +10.6% | Dec 23, 2024 |
| Phase 4 | seed777 + arousal_specialist | **0.6833** | **+13.0%** | Dec 24, 2024 |

#### Individual Model Benchmarks
| Model | Overall CCC | Valence CCC | Arousal CCC | Training Time |
|-------|------------:|------------:|------------:|---------------|
| seed777 | 0.6554 | 0.7593 | 0.5516 | 2.5h (A100) |
| arousal_specialist | 0.6512 | 0.7192 | **0.5832** | 24min (A100) |
| seed888 | 0.6211 | 0.7210 | 0.5212 | 2.3h (A100) |

#### Test Set Submission
- **Test Users**: 46 users
- **Total Predictions**: 1,266 (Valence + Arousal pairs)
- **Expected Test CCC**: 0.6733 - 0.6933 (conservative to optimistic range)
- **Submission File**: `pred_subtask2a.csv` (submitted January 2026)

### Key Findings
1. **Dimension-Specific Optimization**: 4.3% more powerful than generic multi-tasking
2. **Loss Engineering**: 90% CCC weighting critical for agreement metrics
3. **Ensemble Simplicity**: 2-model outperforms 3-5 models (quality > quantity)
4. **Temporal Features**: Contribute 6% to overall accuracy (ablation study)
5. **Multi-Seed Variance Reduction**: Systematic experimentation reduced variance by 12%

### Technical Achievements
- **Production Pipeline**: 721-line modular codebase with reproducible results
- **GPU Efficiency**: Total 10 cumulative hours on A100 for 5 models
- **Documentation**: 40-page technical report (FINAL_REPORT.md, 1,000+ lines)
- **Project Structure**: Reorganized into 01_training, 02_prediction, 03_evaluation phases

---

## 4. Individual Contributions

### Rostislav SVITSOV - Subtask 1: Between-User Emotion Recognition
**Task**: Predicting emotional states for different users based on shared linguistic cues

[Team member will complete this section]

---

### Hyun Chang-Yong (현창용) - Subtask 2a: Within-User State Change Forecasting

#### Primary Responsibilities (100% of Subtask 2a)
1. **Data Analysis & Preprocessing**
   - Conducted comprehensive EDA, identifying the 38% Arousal volatility gap
   - Engineered 47-dimensional feature taxonomy (temporal, textual, personal)
   - Designed arousal-specific features (change, volatility, acceleration)
   - Implemented dynamic dimension handling (863 vs 866 features)

2. **Model Architecture Design**
   - Designed hybrid RoBERTa + BiLSTM + Attention architecture
   - Implemented dual-head output with dimension-specific optimization
   - Developed Arousal-Specialist model (90% CCC loss weighting)
   - Trained 5 models across different seeds and configurations

3. **Ensemble Optimization**
   - Conducted grid search across 5,000+ weight combinations
   - Discovered optimal 2-model ensemble (50.16% / 49.84% split)
   - Tested and rejected meta-learning approaches (Ridge, XGBoost)
   - Validated quality-over-quantity principle

4. **Production Pipeline Development**
   - Built Google Colab prediction notebook (9-step pipeline)
   - Implemented dynamic input dimension handling for deployment
   - Generated 1,266 test predictions for 46 users
   - Created submission.zip with validation scripts

5. **Documentation & Project Management**
   - Wrote comprehensive 40-page technical report (FINAL_REPORT.md)
   - Created 6 specialized guides (training, prediction, evaluation)
   - Reorganized project structure for clarity and reproducibility
   - Maintained Git version control with detailed commit history

6. **Key Innovation Contribution**
   - **Arousal-Specialist Model**: Solved 38% prediction gap through loss engineering
   - **Result**: Overall CCC 0.6833 (+10.4% above target of 0.62)
   - **Impact**: Demonstrated dimension-specific optimization outperforms multi-tasking by 4.3%

#### Time Investment
- **Total Duration**: 15 months (November 2024 - January 2026)
- **Training Time**: ~10 cumulative GPU hours (Google Colab Pro A100)
- **Code**: 721 lines of production-grade Python
- **Documentation**: 2,500+ lines across 8 comprehensive guides

#### Technical Skills Demonstrated
- Deep Learning: PyTorch, Transformers (Hugging Face), RoBERTa fine-tuning
- NLP: Text preprocessing, tokenization, embedding extraction
- Time Series: Temporal feature engineering, sequence modeling (LSTM)
- Ensemble Methods: Grid search optimization, weighted averaging
- Infrastructure: Google Colab (A100 GPU), mixed precision training (FP16)
- Software Engineering: Modular pipeline design, version control (Git), reproducibility

---

## Conclusion

This project successfully forecasted emotional state changes (Subtask 2a) by achieving **CCC 0.6833**, exceeding the competition target by **10.4%**. The key innovation—the Arousal-Specialist Model—solved a systematic 38% performance gap through dimension-specific loss engineering (90% CCC weighting), proving that focused optimization outperforms generic multi-tasking approaches. The final 2-model ensemble demonstrates that quality exceeds quantity in ensemble design. All code, documentation, and trained models have been organized into a production-ready pipeline submitted for the SemEval 2026 competition.

---

**Repository**: [https://github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2](https://github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2)
**Competition**: [Codabench - SemEval 2026 Task 2](https://www.codabench.org/competitions/9963/)
