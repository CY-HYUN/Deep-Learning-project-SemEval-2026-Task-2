# SemEval 2026 Task 2a - Project Structure

**Project**: Emotional State Change Forecasting
**Author**: Hyun Chang-Yong
**Institution**: Télécom SudParis, France
**Date**: January 2026

---

## 📁 Repository Organization

This repository contains a complete implementation of SemEval 2026 Task 2a (Subtask 2a: State Change Forecasting), achieving **CCC 0.6833** (+10.4% above target).

---

## 🗂️ Folder Structure

```
Deep-Learning-project-SemEval-2026-Task-2/
│
├── README.md                          # Main project README
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git exclusion rules
├── .gitattributes                     # Git LFS configuration
│
├── data/                              # Data folder (excluded from Git)
│   ├── raw/                           # Original competition data (Git LFS/excluded)
│   ├── processed/                     # Preprocessed data (small files tracked)
│   ├── test/                          # Test data (excluded)
│   └── trial/                         # Trial data (tracked for testing)
│
├── models/                            # Trained models (4.3 GB, excluded from Git)
│   ├── seed777/                       # Valence master model
│   ├── arousal_specialist/            # Arousal expert model
│   └── ...                            # Other experimental models
│
├── scripts/                           # All executable scripts
│   ├── README.md                      # Scripts documentation
│   ├── 01_training/                   # Training scripts
│   │   ├── train_roberta_baseline.py  # Baseline RoBERTa training
│   │   ├── train_ensemble.py          # Ensemble model training
│   │   └── ...
│   ├── 02_prediction/                 # Prediction scripts
│   │   ├── predict_ensemble.py        # Generate predictions
│   │   └── ...
│   ├── 03_evaluation/                 # Evaluation scripts
│   │   ├── evaluate_ccc.py            # CCC metric calculation
│   │   └── ...
│   ├── demo/                          # Live demo files
│   │   ├── demo_live_presentation.ipynb  # 🎯 MAIN DEMO NOTEBOOK
│   │   └── extract_visualizations.py  # Visualization generator
│   └── archive/                       # Old/deprecated scripts (excluded)
│
├── demo_visualizations/               # 🎨 Demo visualizations (8 files)
│   ├── README.md                      # Visualization documentation
│   ├── 01_user137_emotional_timeline.png
│   ├── 02_prediction_results_combined.png
│   ├── 03_model_contribution_analysis.png
│   ├── 01_russells_circumplex.png
│   ├── 02_scatter_density.png
│   ├── 03_model_comparison.png
│   ├── 04_feature_importance.png
│   └── 05_training_progress.png
│
├── results/                           # Prediction results & analysis
│   └── subtask2a/
│       ├── predictions_final.csv      # Final submission predictions
│       ├── evaluation_metrics.json    # Performance metrics
│       └── archive/                   # Old results (excluded)
│
└── docs/                              # Documentation
    ├── 01_core/                       # Core documentation (public)
    │   ├── QUICKSTART.md              # Quick start guide
    │   ├── PROJECT_STATUS.md          # Current project status
    │   └── TRAINING_STRATEGY.md       # Training methodology
    │
    ├── 02_development/                # Development notes (public)
    │   ├── model_experiments.md       # Experiment logs
    │   └── hyperparameter_tuning.md   # Hyperparameter optimization
    │
    ├── 03_submission/                 # Submission materials
    │   ├── Live_Demo_Script_EN_Full.md  # 🎯 LIVE DEMO SCRIPT (10-12 min)
    │   └── final_submission/
    │       ├── README.md              # Submission overview
    │       └── Final_PPT_and_REPORT/
    │           └── Final_Submission_Docs/
    │               ├── SemEval 2026 Task2_ Emotional State Change Forecasting Joint Presentation.pptx  # 📊 FINAL PPT
    │               └── SemEval_2026_Task2_Report.docx  # 📄 FINAL REPORT
    │
    ├── 04_communication/              # Internal emails (excluded from Git)
    └── 05_archive/                    # Old/deprecated docs (excluded)
```

---

## 🎯 Key Files for External Review

### For Professors / Reviewers / Interviewers

#### 1. **Main Documentation**
- 📖 **[README.md](README.md)** - Project overview, quick start, results summary
- 📖 **[docs/01_core/QUICKSTART.md](docs/01_core/QUICKSTART.md)** - Installation & usage guide
- 📖 **[docs/01_core/PROJECT_STATUS.md](docs/01_core/PROJECT_STATUS.md)** - Current status & achievements

#### 2. **Final Deliverables**
- 📊 **[Final Presentation (PPTX)](docs/03_submission/final_submission/Final_PPT_and_REPORT/Final_Submission_Docs/SemEval%202026%20Task2_%20Emotional%20State%20Change%20Forecasting%20Joint%20Presentation.pptx)** - 31 slides, joint presentation
- 📄 **[Final Report (DOCX)](docs/03_submission/final_submission/Final_PPT_and_REPORT/Final_Submission_Docs/SemEval_2026_Task2_Report.docx)** - Comprehensive technical report

#### 3. **Live Demo**
- 🎯 **[Demo Notebook](scripts/demo/demo_live_presentation.ipynb)** - Interactive demo with User 137 example
- 🎤 **[Demo Script](docs/03_submission/Live_Demo_Script_EN_Full.md)** - 10-12 minute presentation script (bilingual: English + Korean)
- 🎨 **[Visualizations](demo_visualizations/)** - 8 high-quality PNG files with documentation

#### 4. **Code**
- 🔧 **[Training Scripts](scripts/01_training/)** - Model training pipeline
- 🔧 **[Prediction Scripts](scripts/02_prediction/)** - Inference pipeline
- 🔧 **[Evaluation Scripts](scripts/03_evaluation/)** - CCC metric calculation

---

## 📊 Data & Models

### Data Files (Not in Git)
- **Raw data**: `data/raw/` - Original competition data (available from SemEval organizers)
- **Test data**: `data/test/` - Test set (released Jan 5, 2026)
- **Processed data**: `data/processed/` - Preprocessed features (can be regenerated)

### Model Files (Not in Git - 4.3 GB)
- **Location**: `models/`
- **Reproducibility**: All models can be retrained using scripts in `scripts/01_training/`
- **Download**: Contact author for pre-trained models (optional)

**Why excluded from Git?**
- Models are too large for GitHub (4.3 GB)
- Training scripts provided for full reproducibility
- Results can be validated using prediction scripts

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/yourusername/Deep-Learning-project-SemEval-2026-Task-2.git
cd Deep-Learning-project-SemEval-2026-Task-2

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup
```bash
# Download competition data from SemEval organizers
# Place in data/raw/

# Or use trial data for testing
# data/trial/ is included in repository
```

### 3. Training (Optional)
```bash
# Train baseline model
python scripts/01_training/train_roberta_baseline.py

# Train arousal specialist
python scripts/01_training/train_arousal_specialist.py

# Train ensemble
python scripts/01_training/train_ensemble.py
```

### 4. Demo
```bash
# Open demo notebook
jupyter notebook scripts/demo/demo_live_presentation.ipynb

# Or generate visualizations
python scripts/demo/extract_visualizations.py
```

---

## 📈 Results Summary

- **Final CCC**: 0.6833 (+10.4% above target 0.62)
- **Valence CCC**: 0.7593
- **Arousal CCC**: 0.5832 (+6% from arousal specialist)
- **Prediction Time**: <2 seconds on T4 GPU
- **Infrastructure**: Google Colab (free tier)

---

## 🔬 Technical Highlights

### Model Architecture
- **RoBERTa-base**: 125M parameters, 768-dim embeddings
- **BiLSTM**: 256 units × 2 layers, bidirectional
- **Multi-Head Attention**: 8 heads
- **47 Features**: Temporal (20) + Text (15) + User (12)

### Key Innovation
- **Arousal-Specialist Model**: 90% CCC loss weighting (+6% Arousal improvement)
- **2-Model Ensemble**: seed777 (50.16%) + arousal_specialist (49.84%)
- **Quality > Quantity**: 2-model beats 3-model and 5-model ensembles

### Feature Engineering
- **Arousal-specific features**: Change, volatility, acceleration
- **Temporal features**: Lag-1/2/3, rolling statistics, trend
- **Text features**: Length, sentiment keywords, lexical diversity
- **User features**: Mean, std, historical baselines

---

## 📝 Documentation Index

### Public Documentation (Tracked by Git)
1. **Core Docs** (`docs/01_core/`)
   - QUICKSTART.md - Installation & usage
   - PROJECT_STATUS.md - Current status
   - TRAINING_STRATEGY.md - Model training methodology

2. **Development Notes** (`docs/02_development/`)
   - Experiment logs
   - Hyperparameter tuning results
   - Ablation study findings

3. **Submission Materials** (`docs/03_submission/`)
   - Live_Demo_Script_EN_Full.md - Presentation script
   - Final PPTX & DOCX in `final_submission/`

### Internal Documentation (Excluded from Git)
- `docs/04_communication/` - Professor emails, internal planning
- `docs/05_archive/` - Old/deprecated files
- PPT generation prompts, planning documents

---

## 🛠️ Development Workflow

### Typical Research Cycle
1. **Experiment** → Train new model variant
2. **Evaluate** → Calculate CCC scores
3. **Compare** → Benchmark against baselines
4. **Document** → Update experiment logs
5. **Iterate** → Refine based on results

### Code Organization
- **Training**: `scripts/01_training/`
- **Prediction**: `scripts/02_prediction/`
- **Evaluation**: `scripts/03_evaluation/`
- **Demo**: `scripts/demo/`

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for full list.

**Core Libraries:**
- PyTorch 2.0+
- Transformers (Hugging Face)
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Scipy (CCC calculation)

---

## 🙏 Acknowledgments

- **SemEval 2026 Organizers** for the competition and dataset
- **Télécom SudParis** for academic support
- **Google Colab** for free GPU resources

---

## 📧 Contact

**Author**: Hyun Chang-Yong
**Email**: [your-email@example.com]
**Institution**: Télécom SudParis, France
**GitHub**: [your-github-username]

---

## 📄 License

[Specify license if applicable]

---

**Last Updated**: January 28, 2026
