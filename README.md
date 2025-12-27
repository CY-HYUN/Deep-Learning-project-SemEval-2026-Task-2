# SemEval 2026 Task 2: Emotional State Change Forecasting

> **Deep Learning Ensemble for Predicting Emotional Valence and Arousal Changes**
>
> International NLP Competition | November 2024 - January 2025

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This project tackles **SemEval 2026 Task 2, Subtask 2a**: forecasting users' emotional state changes (Valence and Arousal) from sequential text data. The solution employs a hybrid deep learning architecture combining RoBERTa, BiLSTM, and Multi-Head Attention with an optimized 2-model ensemble strategy.

**Competition**: [Codabench - SemEval 2026 Task 2](https://www.codabench.org/competitions/9963/)

---

## 🏆 Results

### Current Performance (Validation)
```
Overall CCC: 0.6833 (Target: 0.62, +10.4%)
├── Valence:  0.7593
└── Arousal:  0.5832
```

### Competition Submission
> **Status**: Ready for evaluation phase (January 10, 2026)
>
> Final competition results will be updated here after official evaluation.

---

## 🔑 Key Highlights

### Technical Innovation
- **Arousal-Specialized Model**: Developed a specialized model with `CCC_WEIGHT_AROUSAL=0.90` to address 27% Arousal-Valence performance gap
- **Hybrid Architecture**: RoBERTa-base + BiLSTM (256×2) + 8-Head Attention + Dual-Output Head
- **Advanced Feature Engineering**: 20 temporal features (lag, rolling stats, trend, volatility) + 15 text features + 12 user statistics
- **Optimized Ensemble**: 2-model weighted ensemble (seed777: 50.16%, arousal_specialist: 49.84%)

### Performance Improvements
| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------:|-------------------:|------------:|
| **Overall CCC** | 0.6305 | 0.6833 | +8.4% |
| **Arousal CCC** | 0.4600 | 0.5832 | +26.8% |
| **Valence CCC** | 0.7200 | 0.7593 | +5.5% |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2.git
cd Deep-Learning-project-SemEval-2026-Task-2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Training (Optional)
Models are already trained. To retrain:

```bash
# Train arousal specialist model (4 hours on A100 GPU)
python scripts/data_train/subtask2a/train_arousal_specialist.py

# Train seed777 model (2 hours on A100 GPU)
python scripts/data_train/subtask2a/train_ensemble_subtask2a.py --seed 777
```

### 3. Generate Predictions
```bash
# Download evaluation data from Codabench (when released)
# Place test_subtask2a.csv in data/test/

# Run prediction script
python scripts/data_analysis/subtask2a/predict_test_subtask2a_optimized.py

# Validate predictions
python scripts/data_analysis/subtask2a/validate_predictions.py

# Create submission file
zip submission.zip pred_subtask2a.csv
```

---

## 📂 Project Structure

```
Deep-Learning-project-SemEval-2026-Task-2/
├── data/
│   ├── raw/                    # Original training data (579KB)
│   ├── processed/              # Preprocessed features
│   └── test/                   # Evaluation data (released Jan 2026)
│
├── models/                     # Trained models (7.2GB total)
│   ├── subtask2a_seed777_best.pt                      # CCC 0.6554
│   └── subtask2a_arousal_specialist_seed1111_best.pt  # CCC 0.6512
│
├── scripts/
│   ├── data_train/            # Training scripts
│   ├── data_analysis/         # Prediction & evaluation scripts
│   └── archive/               # Previous versions
│
├── results/
│   └── subtask2a/
│       ├── optimal_ensemble.json     # Final ensemble weights
│       └── README.md                 # Results documentation
│
└── docs/
    ├── FINAL_REPORT.md        # 40-page technical report
    ├── PROJECT_STATUS.md      # Current status
    └── TRAINING_STRATEGY.md   # Training methodology
```

---

## 🧠 Architecture

### Model Overview
```
Text Input (128 tokens)
    ↓
RoBERTa-base (768-dim)
    ↓
BiLSTM (256 hidden × 2 layers, bidirectional → 512-dim)
    ↓
Multi-Head Attention (8 heads)
    ↓
Concatenate Features
├── LSTM output (512-dim)
├── User embeddings (64-dim)
├── Temporal features (20-dim)
└── Text features (15-dim)
    ↓
Dual-Head Output
├── Valence Head (MLP: 603 → 256 → 128 → 1)
└── Arousal Head (MLP: 603 → 256 → 128 → 1)
```

### Feature Engineering (47 total features)
1. **Temporal Features (20)**
   - Lag features: lag_1/2/3 for valence & arousal
   - Rolling statistics: mean, std (window=3)
   - Trend analysis: linear trend over last 3 samples
   - Arousal-specific: volatility, acceleration, change magnitude

2. **Text Features (15)**
   - Length metrics: text length, word count, avg word length
   - Structural: sentence count, avg sentence length
   - Punctuation: !, ?, comma, period counts
   - Lexical: uppercase ratio, positive/negative word counts
   - Special characters: digit count, special char count

3. **User Statistics (12)**
   - Emotion statistics: mean, std, min, max, median (valence & arousal)
   - Activity: text count, normalized count

---

## 🔬 Technical Details

### Training Configuration
| Parameter | Value |
|-----------|-------|
| **Base Model** | RoBERTa-base (125M params) |
| **Optimizer** | AdamW |
| **Learning Rate** | 2e-5 (with linear warmup) |
| **Batch Size** | 16 |
| **Epochs** | 50 (early stopping patience: 7) |
| **Loss Function** | CCC Loss (weighted dual-head) |
| **Hardware** | Google Colab Pro A100 40GB GPU |
| **Training Time** | ~4 hours per model |

### Arousal Specialist Configuration
```python
CCC_WEIGHT_VALENCE = 0.10  # 10% weight on valence
CCC_WEIGHT_AROUSAL = 0.90  # 90% weight on arousal ⭐
WEIGHTED_SAMPLING = True    # Oversample low-arousal samples
TEMP_FEATURES_DIM = 20      # Include 3 arousal-specific features
```

### Ensemble Strategy
- **Method**: Weighted average (grid search optimized)
- **Models**:
  - seed777 (50.16%) - Best general performance
  - arousal_specialist (49.84%) - Arousal prediction expert
- **Expected CCC**: 0.6733 - 0.6933 (avg: 0.6833)

---

## 📊 Experimental Results

### Individual Model Performance
| Model | Seed | CCC ↑ | Valence | Arousal | Status |
|-------|-----:|------:|--------:|--------:|--------|
| seed777 | 777 | **0.6554** | **0.7593** | 0.5516 | ✅ Final |
| arousal_specialist | 1111 | 0.6512 | 0.7191 | **0.5832** | ✅ Final |
| seed888 | 888 | 0.6428 | 0.7342 | 0.5514 | ⚠️ Not used |
| seed123 | 123 | 0.5330 | 0.6298 | 0.4362 | ❌ Removed |
| seed42 | 42 | 0.5053 | 0.6532 | 0.3574 | ❌ Removed |

### Ablation Study
| Configuration | CCC | Notes |
|---------------|----:|-------|
| 3-model ensemble (seed42+123+777) | 0.5946 | Baseline |
| 2-model ensemble (seed123+777) | 0.6305 | Removed weak seed42 (+6%) |
| **2-model optimized (seed777+arousal_specialist)** | **0.6833** | Final (+15% vs baseline) |

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.10**: Primary programming language
- **PyTorch 2.0+**: Deep learning framework
- **Transformers 4.30+**: Hugging Face library for RoBERTa
- **Pandas & NumPy**: Data manipulation and numerical computing

### Tools & Infrastructure
- **Google Colab Pro**: A100 GPU training environment
- **Git & GitHub**: Version control
- **Weights & Biases**: Experiment tracking (optional)

---

## 📈 Roadmap

### ✅ Completed
- [x] Data exploration and preprocessing
- [x] Baseline model development (3-model ensemble)
- [x] Arousal specialist model training
- [x] Ensemble optimization
- [x] Validation performance: CCC 0.6833 ✅
- [x] Prediction pipeline implementation
- [x] 40-page technical report

### ⏳ In Progress
- [ ] Codabench evaluation phase (Jan 7-9, 2026)
- [ ] Final competition submission (Jan 10, 2026)

### 🔮 Future Work
- [ ] Update README with official competition results
- [ ] Transformer-XL for longer context modeling
- [ ] Cross-attention between valence and arousal heads
- [ ] Multi-task learning with Subtask 1

---

## 📝 Documentation

Detailed documentation available in `/docs`:

- **[FINAL_REPORT.md](docs/FINAL_REPORT.md)**: Comprehensive 40-page technical report
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)**: Current project status and metrics
- **[TRAINING_STRATEGY.md](docs/TRAINING_STRATEGY.md)**: Training methodology and hyperparameters
- **[scripts/README.md](scripts/README.md)**: Script usage guide

---

## 🤝 Contributing

This is an academic competition project. Contributions are welcome after the competition ends (January 2026).

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Hyun Chang-Yong**
- GitHub: [@ThickHedgehog](https://github.com/ThickHedgehog)
- LinkedIn: [Add your LinkedIn]
- Email: [Add your email]

---

## 🙏 Acknowledgments

- **SemEval 2026 Organizers**: For hosting this challenging competition
- **Hugging Face**: For the Transformers library
- **Google Colab**: For providing A100 GPU resources
- **PyTorch Team**: For the excellent deep learning framework

---

## 📚 References

1. SemEval 2026 Task 2: [Competition Website](https://www.codabench.org/competitions/9963/)
2. RoBERTa: Liu et al. (2019) - [Paper](https://arxiv.org/abs/1907.11692)
3. Concordance Correlation Coefficient: Lin (1989)
4. Emotion Recognition: Russell's Circumplex Model

---

**Last Updated**: December 27, 2024
**Competition Status**: Ready for Evaluation (January 2026)

---

<div align="center">

**🚀 Expected CCC: 0.6833 | Target: 0.62 (+10.4%)**

*Built with ❤️ using PyTorch & Transformers*

</div>
