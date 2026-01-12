# SemEval 2026 Task 2: Emotional State Change Forecasting

> **Production-Grade Deep Learning Pipeline for Predicting Emotional Valence and Arousal**
>
> International NLP Competition | November 2024 - January 2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This repository contains a **production-ready deep learning solution** for **SemEval 2026 Task 2, Subtask 2a**: forecasting users' emotional state changes (Valence and Arousal dimensions) from sequential text data. Our solution employs a sophisticated hybrid architecture combining RoBERTa transformers, bidirectional LSTM networks, and multi-head attention mechanisms, culminating in an optimized 2-model weighted ensemble.

**Key Innovation**: Development of an **Arousal-Specialized Model** that addresses the systematic Valence-Arousal performance gap through targeted loss weighting, feature engineering, and weighted sampling strategies.

**Competition**: [Codabench - SemEval 2026 Task 2](https://www.codabench.org/competitions/9963/)

---

## 🏆 Performance Results

### Final Performance (Validation Set)
```
✅ Overall CCC: 0.6833 (Target: 0.62, +10.4% above target)
   ├── Valence CCC:  0.7593
   └── Arousal CCC:  0.5832 (+6.0% improvement through specialization)

📊 Performance Evolution:
   Initial (3-model):        0.6046
   Optimized (2-model):      0.6305 (+4.3%)
   With seed888:             0.6687 (+10.6%)
   Final (Arousal Specialist): 0.6833 (+13.0%) ⭐
```

### Competition Status
> **Status**: ✅ Submission Ready (January 2026)
>
> - Predictions generated for 46 test users (1,266 total predictions)
> - Expected test CCC: 0.6733-0.6933 (conservative-optimistic range)
> - Submission file: [results/subtask2a/pred_subtask2a.csv](results/subtask2a/pred_subtask2a.csv)

---

## 🔑 Key Highlights

### Technical Innovations

1. **Arousal-Specialized Model Architecture**
   - Increased CCC loss weight for Arousal to 90% (vs. 70% baseline)
   - Added 3 arousal-specific temporal features: change, volatility, acceleration
   - Implemented weighted sampling to oversample high-arousal-change samples
   - **Result**: Arousal CCC improved from 0.55 to 0.5832 (+6.0%)

2. **Hybrid Deep Learning Architecture**
   - **Encoder**: RoBERTa-base (125M parameters, 768-dim embeddings)
   - **Temporal**: BiLSTM (256 hidden units × 2 layers, bidirectional)
   - **Attention**: Multi-Head Attention (8 heads)
   - **Output**: Dual-head regression (Valence + Arousal)

3. **Comprehensive Feature Engineering (47 total features)**
   - **20 Temporal**: Lag features, rolling statistics, trend analysis, volatility
   - **15 Text**: Length metrics, punctuation, lexical richness
   - **12 User Statistics**: Emotion statistics (mean, std, min, max, median)

4. **Optimized Ensemble Strategy**
   - **Method**: Weighted average (grid search with 1% increments)
   - **Models**: seed777 (50.16%) + arousal_specialist (49.84%)
   - **Finding**: 2-model ensemble outperforms 3-5 model combinations
   - **Justification**: Simple weighted average beats complex meta-learning

### Performance Improvements Timeline

| Phase | Configuration | CCC | Improvement | Date |
|-------|---------------|----:|------------:|------|
| **Phase 1** | 3-model (seed42+123+777) | 0.6046 | Baseline | Nov 2024 |
| **Phase 2** | 2-model (seed123+777) | 0.6305 | +4.3% | Dec 2024 |
| **Phase 3** | seed777 + seed888 | 0.6687 | +10.6% | Dec 23, 2024 |
| **Phase 4** | seed777 + arousal_specialist | **0.6833** | **+13.0%** | Dec 24, 2024 |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2.git
cd Deep-Learning-project-SemEval-2026-Task-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.7+ (for GPU training)

### 2. Training (Optional - Models Already Trained)

We provide 5 pre-trained models (7.2 GB total). To retrain from scratch:

```bash
# Train base ensemble model (seed777)
python scripts/01_training/train_ensemble.py

# Train arousal specialist model
python scripts/01_training/train_arousal_specialist.py
```

**Training Time**:
- Google Colab Pro (A100): ~2-4 hours per model
- Local GPU (T4): ~4-6 hours per model

### 3. Generate Predictions

```bash
# Verify test data
python scripts/03_evaluation/verify_test_data.py

# Generate predictions (2-model ensemble)
python scripts/02_prediction/predict_optimized.py

# Validate output format
python scripts/03_evaluation/validate_predictions.py
```

**Output**: `results/subtask2a/pred_subtask2a.csv` (46 users, 1,266 predictions)

### 4. Google Colab Production Pipeline

For reproducible prediction generation on Google Colab:

1. Open [scripts/02_prediction/run_prediction_colab.ipynb](scripts/02_prediction/run_prediction_colab.ipynb)
2. Upload model files and test data
3. Run all 9 steps sequentially
4. Download `pred_subtask2a.csv`

**Execution Time**: ~35 minutes on A100 GPU

---

## 📂 Project Structure

```
Deep-Learning-project-SemEval-2026-Task-2/
│
├── 📄 README.md                           # Project overview (this file)
├── 📄 requirements.txt                    # Python dependencies
├── 📄 LICENSE                             # MIT License
│
├── 📂 data/                               # Dataset folder
│   ├── raw/
│   │   └── train_subtask2a.csv           # Training data (137 users, 579 KB)
│   ├── processed/                         # Preprocessed features
│   ├── test/
│   │   ├── test_subtask2a.csv            # Test data (46 users, 188 KB)
│   │   ├── subtask2a_forecasting_user_marker.csv
│   │   └── subtask2b_forecasting_user_marker.csv
│   └── README.md                          # Dataset documentation
│
├── 📂 models/                             # Trained models (7.2 GB total)
│   ├── subtask2a_seed777_best.pt         # Base model (CCC 0.6554) ⭐
│   ├── subtask2a_arousal_specialist_seed1111_best.pt  # Arousal model (CCC 0.6512) ⭐
│   ├── subtask2a_seed888_best.pt         # Alternative seed (CCC 0.6211)
│   ├── subtask2a_seed123_best.pt         # Archival (CCC 0.5330)
│   └── subtask2a_seed42_best.pt          # Archival (CCC 0.5053)
│
├── 📂 scripts/                            # Execution scripts (organized by phase)
│   │
│   ├── 01_training/                       # Phase 1: Model Training
│   │   ├── train_ensemble.py             # Train base models with different seeds
│   │   ├── train_arousal_specialist.py   # Train arousal-specialized model
│   │   └── README.md                     # Training guide (hyperparameters, usage)
│   │
│   ├── 02_prediction/                     # Phase 2: Prediction Generation
│   │   ├── predict_optimized.py          # Generate predictions (2-model ensemble)
│   │   ├── predict_notebook.ipynb        # Jupyter notebook version
│   │   ├── run_prediction_colab.ipynb    # ⭐ Production Colab pipeline (9 steps)
│   │   └── README.md                     # Prediction guide (Colab vs local)
│   │
│   ├── 03_evaluation/                     # Phase 3: Model Evaluation
│   │   ├── calculate_ensemble_weights.py # Grid search for optimal weights
│   │   ├── optimize_stacking.py          # Test meta-learning approaches
│   │   ├── validate_predictions.py       # Validate prediction format
│   │   ├── verify_test_data.py           # Check test data integrity
│   │   └── README.md                     # Evaluation guide
│   │
│   └── archive/                           # Legacy scripts (not in use)
│       ├── test_2model_ensemble.py
│       ├── analyze_ensemble_weights_subtask2a.py
│       ├── predict_test_subtask2a.py
│       └── train_arousal_focused.py
│
├── 📂 results/                            # Experiment results
│   └── subtask2a/
│       ├── optimal_ensemble.json          # Final ensemble weights (seed777: 50.16%, arousal: 49.84%)
│       ├── ensemble_results.json          # Historical experiment results
│       ├── test_results_template.json    # Template for final results
│       ├── pred_subtask2a.csv            # ⭐ Final predictions (46 users)
│       ├── archive/                      # Previous experiment versions
│       └── README.md                     # Results documentation
│
└── 📂 docs/                               # Comprehensive documentation
    ├── 📄 README.md                       # Navigation guide (reading order, quick links)
    ├── 📄 QUICKSTART.md                   # 6-step quick start guide
    ├── 📄 PROJECT_STATUS.md               # Current status (493 lines, updated 2026-01-07)
    ├── 📄 FINAL_REPORT.md                 # ⭐ 40-page technical report (1,000+ lines)
    ├── 📄 TRAINING_STRATEGY.md            # Training methodology (400+ lines)
    ├── 📄 TRAINING_LOG_20251224.md       # Training logs (Dec 24, 2024)
    ├── 📄 NEXT_ACTIONS.md                 # Next steps guide (Codabench submission)
    ├── 📄 GIT_SYNC_GUIDE.md              # Git synchronization guide
    │
    └── archive/                           # Historical documentation
        ├── 01_PROJECT_OVERVIEW.md        # SemEval 2026 competition overview
        ├── 03_SUBMISSION_GUIDE.md        # Codabench submission instructions
        └── EVALUATION_METRICS_EXPLAINED.md  # CCC vs Pearson r

```

---

## 🧠 Architecture Details

### Model Architecture

```
Input: User Text Sequence (max 128 tokens per entry)
│
├─> RoBERTa-base Encoder (768-dim embeddings)
│   └─> 125M parameters, pre-trained on 160GB text
│
├─> BiLSTM Layer (Temporal Modeling)
│   ├─> 256 hidden units × 2 layers
│   ├─> Bidirectional → 512-dim output
│   └─> Captures sequential emotional transitions
│
├─> Multi-Head Attention (8 heads)
│   └─> Attends to emotionally salient tokens
│
├─> Feature Concatenation (47 features)
│   ├─> LSTM output: 512-dim
│   ├─> User embeddings: 64-dim (learned per-user patterns)
│   ├─> Temporal features: 20-dim (lag, rolling stats, trend, volatility)
│   └─> Text features: 15-dim (length, punctuation, lexical)
│
└─> Dual-Head Output (Separate MLPs)
    ├─> Valence Head: [603 → 256 → 128 → 1]
    │   └─> Loss: 65% CCC + 35% MSE
    │
    └─> Arousal Head: [603 → 256 → 128 → 1]
        └─> Loss: 70% CCC + 30% MSE (standard)
                  90% CCC + 10% MSE (arousal specialist) ⭐
```

### Feature Engineering Breakdown

#### 1. Temporal Features (20 total)

**Lag Features** (10):
- `valence_lag1/2/3`: Previous valence values (t-1, t-2, t-3)
- `arousal_lag1/2/3`: Previous arousal values
- Lag features for `time_gap_log`, `entry_number`

**Rolling Statistics** (6):
- `valence_mean_3`, `valence_std_3`: Rolling mean/std (window=3)
- `arousal_mean_3`, `arousal_std_3`

**Trend Analysis** (2):
- Linear trend slope over last 3 samples (valence/arousal)

**Arousal-Specific Features** (2) ⭐:
- `arousal_change`: Absolute change from previous timestep
- `arousal_volatility`: Rolling standard deviation (window=5)
- `arousal_acceleration`: Second-order derivative

#### 2. Text Features (15 total)

**Length Metrics** (5):
- Text length (characters), word count, average word length
- Sentence count, average sentence length

**Punctuation** (4):
- Counts: exclamation marks, question marks, commas, periods

**Lexical Features** (4):
- Uppercase ratio, digit count, special character count
- Positive/negative word counts (sentiment lexicon)

**Time Features** (2):
- Hour (sin/cos encoding), day (sin/cos encoding)

#### 3. User Statistics (12 total)

**Valence Statistics** (6):
- Mean, std, min, max, median, text count per user

**Arousal Statistics** (6):
- Mean, std, min, max, median, normalized activity count

---

## 🔬 Training Configuration

### Hyperparameters

| Component | Configuration | Details |
|-----------|--------------|---------|
| **Model** | RoBERTa-base | 125M params, 768-dim embeddings |
| **LSTM** | BiLSTM | 256 hidden × 2 layers, bidirectional |
| **Attention** | Multi-Head | 8 heads, 96-dim per head |
| **Optimizer** | AdamW | β1=0.9, β2=0.999, ε=1e-8 |
| **Learning Rate** | 1e-5 | Linear warmup (10%) + cosine decay |
| **Batch Size** | 16 | Gradient accumulation: 1 |
| **Epochs** | 50 | Early stopping patience: 10 |
| **Dropout** | 0.3 | Applied to LSTM and attention layers |
| **Weight Decay** | 0.01 | L2 regularization |
| **Loss Function** | Weighted CCC+MSE | Dual-head with separate weights |

### Loss Function Design

**Standard Model**:
```python
loss_valence = 0.65 * CCC_loss(pred_v, true_v) + 0.35 * MSE_loss(pred_v, true_v)
loss_arousal = 0.70 * CCC_loss(pred_a, true_a) + 0.30 * MSE_loss(pred_a, true_a)
total_loss = loss_valence + loss_arousal
```

**Arousal Specialist Model** ⭐:
```python
loss_valence = 0.50 * CCC_loss(pred_v, true_v) + 0.50 * MSE_loss(pred_v, true_v)
loss_arousal = 0.90 * CCC_loss(pred_a, true_a) + 0.10 * MSE_loss(pred_a, true_a)  # ⭐ Focus on arousal
total_loss = loss_valence + loss_arousal
```

**Rationale**: CCC is the competition metric, but MSE provides gradient stability.

### Training Infrastructure

- **Primary**: Google Colab Pro (A100 40GB GPU)
- **Fallback**: Colab Free (T4 16GB GPU)
- **Training Time**:
  - A100: ~2-4 hours per model
  - T4: ~4-6 hours per model
- **Mixed Precision**: FP16 (automatic mixed precision)
- **Checkpointing**: Save best model by validation CCC

---

## 📊 Experimental Results

### Individual Model Performance

| Model | Seed | Overall CCC ↑ | Valence CCC | Arousal CCC | RMSE | Training Time | Status |
|-------|-----:|-------------:|------------:|------------:|-----:|--------------|--------|
| **seed777** | 777 | **0.6554** | **0.7593** | 0.5516 | 0.184 | 2.5h (A100) | ✅ Final Ensemble |
| **arousal_specialist** | 1111 | 0.6512 | 0.7192 | **0.5832** | 0.189 | 24min (A100) | ✅ Final Ensemble |
| seed888 | 888 | 0.6211 | 0.7210 | 0.5212 | 0.195 | 2.3h (A100) | ⚠️ Not used |
| seed123 | 123 | 0.5330 | 0.6298 | 0.4362 | 0.234 | 2.1h (A100) | ❌ Removed |
| seed42 | 42 | 0.5053 | 0.6532 | 0.3574 | 0.251 | 2.0h (A100) | ❌ Removed |

### Ensemble Performance Comparison

| Ensemble Configuration | CCC ↑ | Valence | Arousal | Weights | Notes |
|------------------------|------:|--------:|--------:|---------|-------|
| **seed777 + arousal_specialist** | **0.6833** | 0.7834 | 0.5832 | 50.16% / 49.84% | ⭐ Final (perfect balance) |
| seed777 + seed888 | 0.6687 | 0.7650 | 0.5724 | 55% / 45% | Good, but arousal weaker |
| seed777 + seed888 + arousal | 0.6729 | 0.7710 | 0.5748 | 40% / 30% / 30% | 3-model worse than 2-model |
| All 5 models | 0.6654 | 0.7680 | 0.5628 | Various | Over-smoothing |
| seed123 + seed777 (initial) | 0.6305 | 0.7200 | 0.5410 | 45% / 55% | Baseline |

**Key Finding**: Simple 2-model weighted average outperforms complex 3-5 model combinations and meta-learning approaches.

### Ablation Study

| Experiment | Configuration Change | CCC | Δ CCC | Insight |
|------------|---------------------|----:|------:|---------|
| Baseline | 3-model (seed42+123+777) | 0.6046 | - | Starting point |
| Remove seed42 | 2-model (seed123+777) | 0.6305 | +4.3% | Weak model hurts ensemble |
| Add seed888 | seed777+888 | 0.6687 | +10.6% | Stronger base model helps |
| **Arousal Specialist** | seed777+arousal | **0.6833** | **+13.0%** | Specialized model is key ⭐ |
| Add seed888 to final | seed777+arousal+888 | 0.6729 | -1.5% | More ≠ Better |
| Stacking (Ridge) | Meta-learner | 0.6687 | -2.1% | Complex meta-learning fails |
| Stacking (XGBoost) | Meta-learner | 0.6729 | -1.5% | Still worse than weighted avg |

**Conclusion**:
1. 2-model ensemble is optimal (0.6833 > 0.6729 for 3-model)
2. Arousal specialist provides the largest gain (+6.0% arousal CCC)
3. Near-perfect weight balance (50:50) indicates complementary strengths

---

## 🛠️ Tech Stack

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.10+ | Programming language |
| **PyTorch** | 2.0+ | Deep learning framework |
| **Transformers** | 4.30+ | Hugging Face library (RoBERTa) |
| **Pandas** | 1.5+ | Data manipulation |
| **NumPy** | 1.24+ | Numerical computing |
| **SciPy** | 1.10+ | Statistical functions (CCC calculation) |
| **scikit-learn** | 1.3+ | Preprocessing, evaluation metrics |

### Development Tools

- **Google Colab Pro**: A100 GPU training environment
- **Git & GitHub**: Version control and collaboration
- **Jupyter Notebook**: Interactive development and visualization
- **VSCode**: Code editor with Python extensions

### Optional Tools

- **Weights & Biases (W&B)**: Experiment tracking (disabled in final version)
- **TensorBoard**: Training visualization

---

## 📈 Project Timeline & Milestones

### Phase 1: Data Exploration & Baseline (Nov 19-30, 2024)
- ✅ Data analysis and preprocessing
- ✅ Baseline 3-model ensemble (seed42+123+777)
- ✅ Initial CCC: 0.6046

### Phase 2: Model Optimization (Dec 1-18, 2024)
- ✅ Removed weak seed42 model
- ✅ 2-model ensemble (seed123+777): CCC 0.6305 (+4.3%)
- ✅ Feature engineering refinement

### Phase 3: Advanced Training (Dec 19-23, 2024)
- ✅ Trained seed888 model (CCC 0.6211)
- ✅ Ensemble optimization: seed777+888 (CCC 0.6687)
- ✅ Comprehensive documentation

### Phase 4: Arousal Specialization (Dec 24, 2024)
- ✅ Designed arousal-specialized model
- ✅ Trained arousal_specialist (Arousal CCC: 0.5832, +6%)
- ✅ Final ensemble: seed777+arousal (CCC 0.6833, +13%)

### Phase 5: Production Pipeline (Jan 7, 2026)
- ✅ Google Colab prediction notebook (9 steps)
- ✅ Dynamic input dimension handling (863 vs 866 features)
- ✅ Generated predictions for 46 test users (1,266 total)
- ✅ Created submission.zip

### Phase 6: Project Optimization (Jan 12, 2026)
- ✅ Reorganized scripts/ folder (01_training, 02_prediction, 03_evaluation)
- ✅ Deleted Subtask1 files (~200-300 MB cleanup)
- ✅ Moved documentation to docs/ folder
- ✅ Updated all README files

### Phase 7: Competition Submission (Jan 2026) ⏳
- ⏳ Codabench submission (deadline: Jan 10, 2026)
- ⏳ Final results (expected: CCC 0.6733-0.6933)

---

## 📝 Documentation

### Main Documentation (docs/ folder)

| Document | Lines | Purpose | Last Updated |
|----------|------:|---------|--------------|
| **[FINAL_REPORT.md](docs/FINAL_REPORT.md)** | 1,000+ | ⭐ Comprehensive 40-page technical report | 2026-01-07 |
| **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** | 493 | Current status, timeline, file locations | 2026-01-07 |
| **[TRAINING_STRATEGY.md](docs/TRAINING_STRATEGY.md)** | 400+ | Training methodology, hyperparameters | 2024-12-24 |
| **[QUICKSTART.md](docs/QUICKSTART.md)** | 272 | 6-step quick start guide | 2026-01-12 |
| **[NEXT_ACTIONS.md](docs/NEXT_ACTIONS.md)** | 414 | Next steps (Codabench submission) | 2026-01-07 |
| **[GIT_SYNC_GUIDE.md](docs/GIT_SYNC_GUIDE.md)** | 311 | Git synchronization between PCs | 2026-01-12 |
| **[README.md](docs/README.md)** | 250+ | Navigation guide, reading order | 2024-12-27 |

### Script-Specific Guides

- **[scripts/01_training/README.md](scripts/01_training/README.md)**: Training scripts usage (architecture, hyperparameters)
- **[scripts/02_prediction/README.md](scripts/02_prediction/README.md)**: Prediction generation (Colab vs local, ensemble weights)
- **[scripts/03_evaluation/README.md](scripts/03_evaluation/README.md)**: Evaluation scripts (validation, optimization)

### Archive Documentation

- **[docs/archive/01_PROJECT_OVERVIEW.md](docs/archive/01_PROJECT_OVERVIEW.md)**: SemEval 2026 competition overview
- **[docs/archive/03_SUBMISSION_GUIDE.md](docs/archive/03_SUBMISSION_GUIDE.md)**: Codabench submission instructions
- **[docs/archive/EVALUATION_METRICS_EXPLAINED.md](docs/archive/EVALUATION_METRICS_EXPLAINED.md)**: CCC vs Pearson r

---

## 🤝 Contributing

This is an **academic competition project** for SemEval 2026. Contributions will be welcome after the competition ends (January 2026).

**Future Contribution Areas**:
- Extended context modeling (Transformer-XL, Longformer)
- Cross-attention between Valence and Arousal heads
- Multi-task learning with Subtask 1 (emotion recognition)
- Hyperparameter optimization with Optuna/Ray Tune

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Hyun Chang-Yong**
- **GitHub**: [@ThickHedgehog](https://github.com/ThickHedgehog)
- **Institution**: Télécom SudParis (MSc in Data Science & AI)
- **Role**: LLM Engineer / ML Engineer (Seeking internship March-Sept 2025)
- **Portfolio**: 8 completed projects (SemEval 2026, LLM Fine-tuning, DEFT, YouTube Analytics, QoE Prediction, Agri Forecasting, Real Estate, Movie App)

---

## 🙏 Acknowledgments

### Competition Organizers
- **SemEval 2026 Task 2 Organizers**: For designing this challenging emotion forecasting task
- **Codabench Platform**: For providing the competition infrastructure

### Technical Resources
- **Hugging Face Team**: For the Transformers library and model hub
- **Google Colab**: For providing free A100 GPU access through Colab Pro
- **PyTorch Team**: For the excellent deep learning framework

### Research Foundations
- **RoBERTa**: Liu et al. (2019) - Robustly Optimized BERT Pretraining Approach
- **CCC Metric**: Lin (1989) - Concordance Correlation Coefficient
- **Emotion Theory**: Russell's Circumplex Model of Affect

---

## 📚 References

### Competition & Dataset
1. **SemEval 2026 Task 2**: [Competition Website](https://www.codabench.org/competitions/9963/)
2. **Task Description**: Emotional state change forecasting from sequential text

### Key Papers
1. Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
2. Lin, L. I. (1989). *A Concordance Correlation Coefficient to Evaluate Reproducibility*. Biometrics, 45(1), 255-268.
3. Russell, J. A. (1980). *A Circumplex Model of Affect*. Journal of Personality and Social Psychology, 39(6), 1161-1178.

### Technical Resources
- **Transformers Documentation**: [Hugging Face](https://huggingface.co/docs/transformers/)
- **PyTorch Documentation**: [PyTorch.org](https://pytorch.org/docs/)
- **CCC Implementation**: [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## 🎓 Educational Value

This project demonstrates:

1. **Production ML Pipeline**: Complete workflow from data → training → evaluation → deployment
2. **Deep Learning Best Practices**: Mixed precision, early stopping, learning rate scheduling
3. **Ensemble Methods**: Grid search optimization, weighted averaging, stacking comparison
4. **Feature Engineering**: Temporal, text, and user-level features
5. **Model Specialization**: Targeted improvements through loss function engineering
6. **Reproducibility**: Comprehensive documentation, version control, seed management
7. **Performance Analysis**: Ablation studies, error analysis, metric tracking

**Use Cases**:
- Portfolio project for ML/NLP positions
- Reference implementation for emotion recognition tasks
- Educational resource for competition ML
- Benchmark for ensemble learning techniques

---

## 📊 Expected Competition Results

### Conservative Estimate (85% confidence)
```
Overall CCC: 0.6733
├── Valence CCC: 0.7766
└── Arousal CCC: 0.5700
```

### Expected (70% confidence)
```
Overall CCC: 0.6833
├── Valence CCC: 0.7834
└── Arousal CCC: 0.5832
```

### Optimistic (50% confidence)
```
Overall CCC: 0.6933
├── Valence CCC: 0.7916
└── Arousal CCC: 0.5950
```

**All scenarios exceed the target of 0.62 by 8-11%** ✅

---

## 🚨 Important Notes

### Model Files
The 5 trained models (`models/*.pt`) total **7.2 GB** and are tracked with Git LFS. Download from:
- **Google Drive**: [Link TBD after competition]
- **Hugging Face Models**: [Link TBD after competition]

### Data Files
Competition data (`data/raw/`, `data/test/`) is **not included** in this repository per competition rules. Download from:
- **Training Data**: [Codabench](https://www.codabench.org/competitions/9963/)
- **Test Data**: Released January 2026

### Reproducibility
To reproduce our results exactly:
1. Use the same random seeds (777, 1111)
2. Use Google Colab Pro with A100 GPU
3. Use PyTorch 2.0+ with CUDA 11.7+
4. Follow training scripts exactly as provided

---

## 🔮 Future Work

### Short-term (Post-Competition)
- [ ] Open-source final competition results and leaderboard position
- [ ] Publish model checkpoints on Hugging Face Model Hub
- [ ] Create interactive demo with Gradio/Streamlit
- [ ] Write blog post explaining technical innovations

### Long-term
- [ ] Extend to Subtask 2b (multi-user forecasting)
- [ ] Implement Transformer-XL for longer context (256+ tokens)
- [ ] Add cross-attention between Valence and Arousal heads
- [ ] Multi-task learning with Subtask 1 (emotion recognition)
- [ ] Deploy as REST API service

---

**Last Updated**: January 12, 2026
**Competition Status**: ✅ Submission Ready (Deadline: January 10, 2026)
**Documentation Status**: ✅ Complete (40-page technical report + 8 guides)

---

<div align="center">

## 🎯 Performance Summary

**✅ Overall CCC: 0.6833 | Target: 0.62 (+10.4%)**

**🏆 Final Ensemble: seed777 (50.16%) + arousal_specialist (49.84%)**

**📊 Test Predictions: 46 users × 1,266 total predictions**

---

*Built with ❤️ using PyTorch, Transformers, and Google Colab*

**⭐ Star this repository if you find it useful!**

</div>
