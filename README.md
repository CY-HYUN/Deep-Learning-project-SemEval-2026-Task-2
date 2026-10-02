# SemEval 2026 Task 2a — Emotional State Change Forecasting

A hybrid RoBERTa + BiLSTM + attention pipeline that forecasts each user's next emotional
state change (valence and arousal) from their diary-entry history — my entry for
[SemEval 2026 Task 2, Subtask 2a](https://www.codabench.org/competitions/9963/) (Nov 2025 – Jan 2026).
Best single-model validation CCC **0.6554** (target 0.62); an arousal specialist lifted arousal CCC 0.5516 → 0.5832.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/)

---

## Results

Measured validation CCC (Concordance Correlation Coefficient), per trained model.
Sources: [`results/subtask2a/ensemble_results.json`](results/subtask2a/ensemble_results.json)
and [`docs/02_development/TRAINING_LOG_20251224.md`](docs/02_development/TRAINING_LOG_20251224.md).

| Model (seed) | Overall CCC | Valence CCC | Arousal CCC | Note |
|---|---:|---:|---:|---|
| **seed777** | **0.6554** | 0.7593 | 0.5516 | Best single model |
| **arousal_specialist (1111)** | 0.6512 | 0.7192 | **0.5832** | Dimension-specialized (see below) |
| seed888 | 0.6211 | — | — | Alternative baseline |
| seed123 | 0.5330 | 0.6298 | 0.4362 | |
| seed42 | 0.5053 | 0.6532 | 0.3574 | Dropped from ensemble |

**Headline findings (measured):**

- **Best single-model validation CCC 0.6554** (seed777), exceeding the project target of 0.62.
- **Arousal was the bottleneck**: arousal CCC ranged 0.357–0.552 across seeds while valence
  reached 0.759. A dimension-specialized model (90% CCC loss weight on arousal, 3 extra
  arousal features, weighted sampling) lifted **arousal CCC 0.5516 → 0.5832 (+0.0316)**,
  trading only −0.0042 overall CCC — and trained in ~24 min on an A100 vs ~2 h for a full run.
- **Seed variance is large**: the same architecture scored CCC 0.5053–0.6554 across random
  seeds, which motivated multi-seed training and ensembling.

**Final submission (labeled honestly):** the Codabench submission uses a 2-model weighted
ensemble — seed777 (50.16%) + arousal_specialist (49.84%), weights proportional to each
model's validation CCC ([`results/subtask2a/optimal_ensemble.json`](results/subtask2a/optimal_ensemble.json)).
Its CCC is **a heuristic estimate, not a measured score** — the weighted mean
of the two models' measured CCCs plus an assumed +0.02–0.04 ensemble boost
([`scripts/03_evaluation/calculate_ensemble_weights.py`](scripts/03_evaluation/calculate_ensemble_weights.py)) —
never validated on held-out data. Predictions for all 46 test users were submitted to
Codabench in January 2026; the official test score is not recorded in this repository.

---

## Quick Start

Clone the repository (Git LFS is needed for the processed feature table):

```bash
git clone https://github.com/CY-HYUN/Deep-Learning-project-SemEval-2026-Task-2.git
cd Deep-Learning-project-SemEval-2026-Task-2
pip install -r requirements.txt
```

**What runs on a fresh clone** (no weights or competition data needed — all verified):

```bash
# 1. Validate the committed submission file (46 user-level predictions)
cd results/subtask2a
python ../../scripts/03_evaluation/validate_predictions.py
cd ../..

# 2. Reproduce the ensemble-selection arithmetic (rewrites optimal_ensemble.json with the same values)
python scripts/03_evaluation/calculate_ensemble_weights.py

# 3. Regenerate the demo visualizations (synthetic sample data; overwrites demo_visualizations/*.png)
pip install matplotlib seaborn
python scripts/demo/extract_visualizations.py
```

**Demo notebook** — a pre-executed walkthrough of the pipeline on generated sample data
(no model weights required); view it directly on GitHub or open locally:

```bash
jupyter notebook "scripts/demo/demo_live_presentation(Subtask2a).ipynb"
```

**What needs assets that are not in the repo** (honest notes):

- **Model weights** (5 checkpoints, ~1.5 GB each, 7.2 GB total) are gitignored and not published.
  `scripts/02_prediction/predict_optimized.py` and the Colab pipeline
  `scripts/02_prediction/run_prediction_colab.ipynb` require them.
- **Competition data** (`data/raw/`, `data/test/`) is not redistributed per SemEval rules —
  download it from [Codabench](https://www.codabench.org/competitions/9963/). Once
  `data/test/` is in place, `python scripts/03_evaluation/verify_test_data.py` checks it.
  What *is* committed: the processed feature table
  (`data/processed/subtask2a_features.csv`, 2,764 rows × 65 columns, via Git LFS) and a
  small trial sample (`data/trial/trial_data.csv`).
- **Training** was done on Google Colab (A100/T4), not locally:
  `scripts/01_training/train_ensemble.py` is a Colab notebook export in JSON form (open it
  in Colab/Jupyter — it is not runnable with `python`), and
  `scripts/01_training/train_arousal_specialist.py` imports `google.colab`. Seeds are set
  as constants inside the scripts, not CLI flags.

---

## Architecture

Verified against [`scripts/01_training/train_arousal_specialist.py`](scripts/01_training/train_arousal_specialist.py):

```
Per-user sequence of 7 diary entries
│
├─ RoBERTa-base (125M) ── [CLS] embedding per entry (768-dim)
├─ User embedding (64-dim, learned per user)
├─ Engineered features (temporal lags, rolling stats, text stats, user baselines)
│
├─ Linear projection → BiLSTM (hidden 256 × 2 layers, bidirectional → 512-dim)
├─ Multi-head self-attention (4 heads) over the sequence
├─ Fusion MLP: 512 → 256 → 128 (GELU, dropout 0.2)
│
└─ Dual heads (valence / arousal): 128 → 64 → 1 each
   Loss = weighted CCC + MSE per head
          standard model:    65/35 (valence), 70/30 (arousal)
          arousal specialist: 50/50 (valence), 90/10 (arousal)
```

**Training config** (from code): sequence length 7, batch size 10, up to 20 epochs with
early stopping (patience 7), LR 1.5e-5 (RoBERTa) / 8e-5 (other layers), AdamW, seed fixed
per run. Trained on Google Colab (A100 40GB; ~2 h per base model, ~24 min for the specialist).

**Feature engineering**: the committed feature table
([`data/processed/subtask2a_features.csv`](data/processed/subtask2a_features.csv)) contains
55 engineered columns on top of the 10 raw fields — temporal lags (t-1..t-3), rolling
mean/std, velocity, time-gap encodings, text statistics (length, punctuation, sentiment
word counts, tense), cyclical time encodings, and per-user valence/arousal baselines.
Details in [docs/DETAILS.md](docs/DETAILS.md).

---

## Data

| Split | Entries | Users | In repo? |
|---|---:|---:|---|
| Train | 2,764 | 137 | No (competition rules) — processed features committed via LFS |
| Test | 784 | 46 | No (competition rules) |
| Submission | 46 rows | 46 | Yes — [`results/subtask2a/pred_subtask2a.csv`](results/subtask2a/pred_subtask2a.csv) |

Each user contributes a timeline of diary entries with self-reported valence/arousal; the
task is to forecast each test user's next *state change* (one valence delta + one arousal
delta per user — hence 46 prediction rows).

---

## Key Design Decisions

1. **Dimension-specific optimization over multi-task tuning.** Instead of tuning one model
   harder, I trained a second model whose loss, features, and sampling all target the weak
   dimension (arousal). Measured effect: +0.0316 arousal CCC for −0.0042 overall CCC.
2. **CCC-proportional 2-model ensemble.** The submitted ensemble weights the two
   complementary models by their validation CCC (50.16 / 49.84). Model *combination
   selection* used a heuristic estimate (weighted mean + assumed boost), which favored the
   2-model pair over 3–5-model pools — a limitation, since the ensemble was never re-scored
   on a held-out split (see below).
3. **Multi-seed training as variance control.** Five seeds exposed a 0.15 CCC spread and
   let me drop the weakest models from the ensemble pool.

**Known limitations (stated up front):**

- The ensemble CCC is an estimate, not a measurement; individual model
  CCCs are measured. Re-scoring the blended predictions on the validation split would
  close this gap.
- Official Codabench test results are not recorded in this repository.
- Training entry points are Colab artifacts (notebook JSON / `google.colab` imports), so
  local retraining requires porting.
- A stacking/meta-learning comparison was scaffolded
  ([`scripts/03_evaluation/optimize_stacking.py`](scripts/03_evaluation/optimize_stacking.py))
  but depends on saved validation predictions that were never generated — it was not run.

---

## Repository Map

```
├── README.md                      # This file
├── requirements.txt               # Core deps (viz extras: matplotlib, seaborn)
├── data/                          # processed features (LFS) + trial sample; raw/test not redistributed
├── scripts/
│   ├── 01_training/               # Colab training artifacts (notebook JSON + Colab script)
│   ├── 02_prediction/             # Ensemble inference (needs weights) + Colab pipeline
│   ├── 03_evaluation/             # Weight calc, submission validation, test-data checks
│   └── demo/                      # Pre-executed demo notebook + visualization generator
├── results/subtask2a/             # Submission CSV, ensemble weights, measured model CCCs
├── demo_visualizations/           # 8 committed PNGs (regenerable)
└── docs/
    ├── DETAILS.md                 # Extended methodology (code-verified)
    ├── 01_core/                   # Project status, training strategy
    ├── 02_development/            # Training logs (measured numbers), improvement analysis
    └── 03_submission/             # Final presentation (PPTX) + report (DOCX)
```

Academic deliverables: a 33-slide joint presentation and a technical report (DOCX) are in
[`docs/03_submission/final_submission/Final_PPT_and_REPORT/Final_Submission_Docs/`](docs/03_submission/final_submission/Final_PPT_and_REPORT/Final_Submission_Docs/)
(slides 18–32 cover this subtask; slides 1–17 are a teammate's Subtask 1 work, and slide 33 is a shared closing slide).

---

## Tech Stack

PyTorch 2.0+, Hugging Face Transformers (RoBERTa-base), pandas / NumPy / SciPy /
scikit-learn; Google Colab (A100/T4) for training; Git LFS for the committed feature table.

## Author

**Changyong Hyun** — [@CY-HYUN](https://github.com/CY-HYUN) · MSc, Télécom SudParis

References: Liu et al. 2019 (RoBERTa, [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)) ·
Lin 1989 (CCC) · Russell 1980 (Circumplex Model of Affect).
