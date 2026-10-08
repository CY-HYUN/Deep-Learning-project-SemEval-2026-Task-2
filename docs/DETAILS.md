# Extended Methodology — SemEval 2026 Task 2a

Supplementary detail moved out of the top-level README. Every number here is either
measured (cited to the file that records it) or verified directly against the code in
`scripts/`. See the [README](../README.md) for the summary and honest labeling of which
figures are measured vs estimated.

---

## 1. Task

SemEval 2026 Task 2, Subtask 2a: given a user's timeline of diary entries (text +
self-reported valence/arousal), forecast that user's next emotional *state change* — one
valence delta and one arousal delta per user. Metric used in this project: Concordance Correlation
Coefficient (CCC), averaged over valence and arousal.

**What this repo's validation numbers are.** `validate()` in both training scripts computes Pearson r
per dimension and averages the two (`train_arousal_specialist.py:671-673`), although the logs call it
"CCC". The split is a random 15% of entries stratified by user, so the same users are in training and
validation (`train_test_split(..., stratify=df['user_id'])`), and the target is the valence / arousal
level of the last entry in the sequence, not the state change (`train_arousal_specialist.py:400-401`).
True CCC was not measured. Every "r" or "score" below is this number.

Data (verified by loading the CSVs):

- Train: 2,764 entries, 137 users, 10 columns (`user_id`, `text_id`, `text`, `timestamp`,
  `collection_phase`, `is_words`, `valence`, `arousal`, `state_change_valence`,
  `state_change_arousal`).
- Test: 784 entries, 46 users; the marker file flags `is_forecasting_user == True` for the
  46 users needing predictions.
- Submission format: one row per user — `user_id, pred_state_change_valence,
  pred_state_change_arousal` (46 rows; validated by
  `scripts/03_evaluation/validate_predictions.py`).

## 2. Model architecture

From `scripts/01_training/train_arousal_specialist.py` (`FinalEmotionModel`):

| Component | Spec (from code) |
|---|---|
| Text encoder | RoBERTa-base, `[CLS]` token embedding per entry (768-dim) |
| User embedding | `nn.Embedding(num_users, 64)` |
| Input projection | Linear → 512 (LSTM input) |
| Sequence model | BiLSTM, hidden 256 × 2 layers, bidirectional (512-dim output) |
| Attention | `nn.MultiheadAttention(embed_dim=512, num_heads=4)` |
| Fusion | Linear 512→256, GELU, Dropout 0.2, Linear 256→128 |
| Output heads | Two separate MLPs (valence / arousal): 128→64→1, GELU, Dropout 0.2 |

Sequence length: 7 entries per training sample (`SEQ_LENGTH = 7`).

## 3. Loss design

Weighted CCC + MSE per head. CCC is the competition metric; the MSE term stabilizes
gradients.

```python
# Standard models (seed42/123/777/888)
loss_valence = 0.65 * CCC_loss + 0.35 * MSE_loss
loss_arousal = 0.70 * CCC_loss + 0.30 * MSE_loss

# Arousal specialist (seed 1111)
loss_valence = 0.50 * CCC_loss + 0.50 * MSE_loss
loss_arousal = 0.90 * CCC_loss + 0.10 * MSE_loss   # arousal-focused
```

## 4. Arousal specialist — the core experiment

Problem (measured, `results/subtask2a/ensemble_results.json`): arousal r lagged valence
badly across all seeds — seed777 scored valence 0.7593 vs arousal 0.5516; the worst seed
(42) scored arousal 0.3574.

Three targeted changes (all visible in `train_arousal_specialist.py`):

1. **Loss re-weighting**: arousal head CCC weight 0.70 → 0.90.
2. **Three arousal-specific features**: `arousal_change` (abs diff per user),
   `arousal_volatility` (rolling std, window 5), `arousal_acceleration` (diff of change).
3. **Weighted sampling**: `WeightedRandomSampler` with weights proportional to
   `arousal_change + 0.5`, oversampling high-arousal-change entries.

Measured outcome (`docs/02_development/TRAINING_LOG_20251224.md`, best epoch 15/20):

| Metric | seed777 baseline | Arousal specialist | Delta |
|---|---:|---:|---:|
| Arousal r | 0.5516 | not quoted (leak, see caveat) | — |
| Valence r | 0.7593 | 0.7192 | −0.0401 |
| Overall (mean r) | 0.6554 | 0.6512 (includes the leaky arousal score) | −0.0042 |
| Training time (A100) | ~2 h | ~24 min | |

**Caveat:** `arousal_change` is |arousal(t) − arousal(t−1)| computed on the entry being predicted
(`train_arousal_specialist.py:221`, used as input at `:386`, target at `:401`), so it contains the
target. Any arousal gain may come from this feature rather than from the loss and sampling changes.
Not re-measured.

## 5. Ensemble selection (and its limitation)

`scripts/03_evaluation/calculate_ensemble_weights.py` enumerates model combinations
(2–5 models) and, for each combination:

- assigns weights proportional to each model's measured validation score, and
- *estimates* the ensemble score as the weighted mean of individual scores plus an assumed
  ensemble boost of +0.02 to +0.04.

Under that heuristic, seed777 + arousal_specialist ranks first (weights 50.16% / 49.84% —
reproduced with the same values by rerunning the script). **Limitation**: the blended
predictions were never re-scored on a held-out split, so the projected ensemble score was
never validated — the best measured result is therefore the seed777 single model at mean
Pearson r 0.6554; the heuristic also structurally favors small ensembles of strong models (adding a
weaker model lowers the weighted mean while the boost stays fixed). A stacking comparison (`scripts/03_evaluation/optimize_stacking.py`, Ridge on
validation predictions) was scaffolded but not used — it depends on saved per-model
validation predictions that were never generated, and currently runs on random placeholder arrays.

## 6. Feature engineering

The training scripts compute their own features in-script from the raw training CSV (31 in the base
model: 17 temporal, 4 per-user, 10 text). The committed feature table
`data/processed/subtask2a_features.csv` (2,764 × 65, Git LFS) is a separate export that adds 55
engineered columns to the 10 raw fields:

- **Temporal**: lags t-1..t-3 for valence/arousal, rolling mean/std, velocity,
  `entry_number`, `relative_position`, `hours_since_start`, `time_gap_hours`,
  `time_gap_log`.
- **Text statistics**: length/char/word/sentence counts, average word length,
  exclamation/question counts, uppercase ratio, capitalized words, first-person count,
  positive/negative/high-arousal/low-arousal word counts, sentiment score, tense flags.
- **Time encodings**: hour/day-of-week sin+cos, month, weekend flag.
- **User baselines**: per-user valence/arousal mean and std, entry count, timestamp span.
- **Categorical encodings**: `is_words_encoded`, `collection_phase_encoded`.

The arousal specialist adds `arousal_change`, `arousal_volatility`,
`arousal_acceleration` at training time (computed in the training script).

## 7. Training configuration

From `train_arousal_specialist.py` constants and the training log:

| Setting | Value |
|---|---|
| Batch size | 10 |
| Max epochs | 20 (specialist) / 30 (seed888 run, per log) |
| Early stopping | patience 7, checkpoint on best validation score (mean Pearson r) |
| LR (RoBERTa) | 1.5e-5 |
| LR (other layers) | 8e-5 |
| Optimizer | AdamW |
| Seeds | 42, 123, 777, 888, 1111 (set as in-script constants, not CLI flags) |
| Hardware | Google Colab A100 40GB (T4 fallback) |

## 8. Measured results archive

- `results/subtask2a/ensemble_results.json` — 2025-11-20 evaluation of seeds 42/123/777
  (score logged as "ccc" but computed as mean Pearson r, per-dimension r, RMSE, best epoch).
- `docs/02_development/TRAINING_LOG_20251224.md` — seed888 and arousal-specialist training
  runs with measured validation metrics (Korean-language lab log).
- `results/subtask2a/optimal_ensemble.json` — final ensemble weights + estimated score range.
- `results/subtask2a/pred_subtask2a.csv` — submitted predictions (46 users).

## 9. Known issues encountered

- **Feature-dimension mismatch (863 vs 866)** between training and inference
  preprocessing. The prediction pipeline (`scripts/02_prediction/`) detects the dimension so the
  checkpoints load, but this only matches the total size: the Colab notebook feeds 5 lag + 12 user +
  14 text features where training used 17 + 4 + 10, and one entry instead of a 7-entry sequence.
  The mismatch was not fixed.
- **`is_forecasting_user` misreading** (Jan 2026): initially misinterpreted as a
  data-leakage marker; organizers clarified it flags the 46 users needing predictions.
  Full post-mortem preserved in `docs/05_archive/misunderstanding_2026-01-14/`.
- **Colab-coupled training code**: training entry points import `google.colab` /
  are notebook JSON, so local retraining requires porting.
