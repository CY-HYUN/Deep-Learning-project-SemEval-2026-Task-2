# Evaluation Metrics Explained - SemEval 2026 Task 2

**Created**: 2025-12-10
**Last Updated**: 2025-12-10
**Status**: Official metrics released December 6, 2025

---

## 📊 Overview

This document explains the official evaluation metrics for SemEval 2026 Task 2, released on December 6, 2025.

**Key Points**:
- All subtasks use **Pearson r** and **MAE** (Mean Absolute Error)
- **Leaderboard ranking** is based on **Pearson r**
- Each subtask has slightly different evaluation approaches
- Official evaluation script available on GitHub

---

## 🎯 Subtask 1 vs Subtask 2A - Key Differences

### What Each Subtask Predicts

| Aspect | Subtask 1 | Subtask 2A |
|--------|-----------|------------|
| **Task Name** | Longitudinal Affect Assessment | State Change Forecasting |
| **Predicts** | Absolute emotion values | Emotion change values |
| **Output for text** | valence, arousal | Δ_valence, Δ_arousal |
| **Example** | "Text makes me feel 3.0 valence" | "Text changes my valence by +1.0" |
| **Evaluation Focus** | Between-user + Within-user | Per-user correlation |
| **Ranking Metric** | Composite r (Fisher's z) | Pearson r |

### Intuitive Explanation

**Subtask 1**:
> "How happy does this text make you?" (absolute feeling)

**Subtask 2A**:
> "How much happier does this text make you compared to before?" (change in feeling)

---

## 📐 Subtask 1: Longitudinal Affect Assessment

### Three-Part Evaluation

Subtask 1 evaluates models on **three different perspectives**:

#### 1️⃣ Between-User Correlation

**Question**: Does the model understand that different people have different baseline emotions?

**Calculation**:
```python
# For each user, calculate mean emotion across all their texts
user_A_mean_pred = mean([text1_pred, text2_pred, text3_pred, ...])
user_A_mean_gold = mean([text1_gold, text2_gold, text3_gold, ...])

# Do this for all users
all_users_mean_pred = [user_A_mean, user_B_mean, user_C_mean, ...]
all_users_mean_gold = [user_A_mean, user_B_mean, user_C_mean, ...]

# Calculate correlation across users
r_between = pearson_r(all_users_mean_pred, all_users_mean_gold)
```

**Example**:
```
User A: Generally happy person (mean valence = 3.5)
User B: Generally neutral person (mean valence = 2.0)
User C: Generally sad person (mean valence = 1.2)

Good model: Predicts A > B > C
Bad model: Predicts all users similar
```

---

#### 2️⃣ Within-User Correlation

**Question**: Does the model understand that the same person feels different emotions for different texts?

**Calculation**:
```python
# For User A only
user_A_texts_pred = [text1_pred, text2_pred, text3_pred, ...]
user_A_texts_gold = [text1_gold, text2_gold, text3_gold, ...]
r_A = pearson_r(user_A_texts_pred, user_A_texts_gold)

# Do this for all users, then average
r_within = mean([r_A, r_B, r_C, ...])
```

**Example**:
```
User A reads 3 texts:
- Text 1 (happy news): valence = 4.0
- Text 2 (neutral news): valence = 2.5
- Text 3 (sad news): valence = 1.0

Good model: Predicts Text1 > Text2 > Text3 for User A
Bad model: Predicts all texts same for User A
```

---

#### 3️⃣ Composite Correlation (Ranking Metric) ⭐

**Question**: Overall, how good is the model at both between-user and within-user prediction?

**Calculation** (Fisher's z-transformation):
```python
# Convert correlations to Fisher's z
z_between = arctanh(r_between)
z_within = arctanh(r_within)

# Average in z-space
z_composite = (z_between + z_within) / 2

# Convert back to correlation
r_composite = tanh(z_composite)
```

**Why Fisher's z?**: Correlation values are not normally distributed. Fisher's z-transformation makes them more suitable for averaging.

**Leaderboard**: Ranked by **r_composite**

---

### MAE (Mean Absolute Error)

Same three-part structure:
- **MAE_between**: Error in predicting user-level averages
- **MAE_within**: Average error within each user
- **MAE_composite**: Combined using Fisher's z transformation

**Lower is better** for MAE.

---

## 📉 Subtask 2A: State Change Forecasting (Our Focus)

### Single Evaluation Approach

Subtask 2A uses a **simpler evaluation** than Subtask 1:

**Question**: For each user, how well do we predict their emotional state changes?

**Calculation**:
```python
# For each user separately
for user in all_users:
    user_pred_changes = [Δ1_pred, Δ2_pred, Δ3_pred, ...]
    user_gold_changes = [Δ1_gold, Δ2_gold, Δ3_gold, ...]

    r_user = pearson_r(user_pred_changes, user_gold_changes)
    mae_user = mean_absolute_error(user_pred_changes, user_gold_changes)

# Report per-user metrics
```

**Leaderboard Ranking**: Based on **Pearson r** (averaged across users)

---

### What Pearson r Measures for State Change

**Pearson r** measures:
- ✅ **Direction**: Do predicted changes go in the right direction? (positive vs negative)
- ✅ **Relative magnitude**: Are bigger changes predicted as bigger?
- ❌ **Absolute values**: Doesn't heavily penalize being off by a constant

**Example**:

```python
# Scenario 1: Perfect correlation (r = 1.0)
gold_changes = [-1.0, -0.5,  0.0,  0.5,  1.0]
pred_changes = [-1.0, -0.5,  0.0,  0.5,  1.0]
# Perfect match!

# Scenario 2: Still perfect correlation (r = 1.0)
gold_changes = [-1.0, -0.5,  0.0,  0.5,  1.0]
pred_changes = [-0.8, -0.3,  0.2,  0.7,  1.2]
# Off by +0.2, but same pattern!

# Scenario 3: Bad correlation (r = 0.0)
gold_changes = [-1.0, -0.5,  0.0,  0.5,  1.0]
pred_changes = [ 0.1,  0.2,  0.1,  0.2,  0.1]
# All predictions similar, no pattern captured

# Scenario 4: Negative correlation (r = -1.0)
gold_changes = [-1.0, -0.5,  0.0,  0.5,  1.0]
pred_changes = [ 1.0,  0.5,  0.0, -0.5, -1.0]
# Completely backwards!
```

**MAE** measures:
- ✅ **Absolute accuracy**: How far off are the predictions?
- ✅ **No penalty for systematic bias**

**Combining both**:
- **High r, Low MAE**: Excellent model ✅
- **High r, High MAE**: Right patterns but wrong scale ⚠️
- **Low r, Low MAE**: Predicting near-zero changes (safe but useless) ⚠️
- **Low r, High MAE**: Bad model ❌

---

## 🎯 What This Means for Our Model (Subtask 2A)

### Current Training Setup

Our model is trained with:
- **65% CCC Loss + 35% MSE Loss** (Valence)
- **70% CCC Loss + 30% MSE Loss** (Arousal)

**CCC (Concordance Correlation Coefficient)** vs **Pearson r**:

| Metric | What it measures | Formula relationship |
|--------|------------------|---------------------|
| **Pearson r** | Linear correlation only | r = correlation |
| **CCC** | Correlation + Agreement | CCC = r × (accuracy factor) |

**Relationship**:
```
CCC ≤ Pearson r

CCC = Pearson r  (when predictions have same mean/variance as gold)
CCC < Pearson r  (when predictions are biased or have wrong scale)
```

### Expected Performance

**On validation set**:
- Our model: **CCC = 0.5846 to 0.6046**
- Expected Pearson r: **~0.60 to 0.65** (slightly higher than CCC)

**Why Pearson r might be higher**:
- Pearson r doesn't penalize constant offsets
- CCC penalizes scale/bias differences
- Our model might have slight bias that CCC catches

---

## 📊 Test Data Information (December 2025 Release)

### Subtask 1 Test Data

**Size**: 1,737 longitudinal texts from 91 authors

**Additional Column**:
```csv
user_id,text_id,text,timestamp,is_seen_user
```

- `is_seen_user` = `True`: User was in training set (but with future texts)
- `is_seen_user` = `False`: Completely new user

**Implication**: Model needs to handle both seen and unseen users

---

### Subtask 2A Test Data

**Size**: Based on training users, but NO additional text data

**Additional Column**:
```csv
user_id,is_forecasting_user,...
```

- `is_forecasting_user` = `True`: User will be evaluated for forecasting
- `is_forecasting_user` = `False`: Not part of evaluation

**Implication**:
- We already have all text data from training set
- Test phase evaluates predictions on **existing users' last state change**
- No new texts to process

**Format**:
```python
# Training data has:
user_A: [text1, text2, text3, text4, text5]  # Last change is NaN

# Test phase asks to predict:
state_change_valence = valence(text6) - valence(text5)  # But we never see text6!
```

**This means**: Subtask 2A is predicting **future unseen state change**, not just processing test texts.

---

## 🚨 Important Realization for Our Model

### Current Model Design

Our model predicts state change based on:
```python
Input: [text_t, user_embedding, lag_features]
Output: Δ_valence, Δ_arousal
```

**Problem**: This predicts change FROM current text, but test wants change TO next unseen text.

### What Test Actually Requires

```python
# Training: We see both texts
text_t = "I had a good day at work"
text_t+1 = "My cat is sick"
gold_change = emotion(text_t+1) - emotion(text_t)

# Test: We only see current text
text_t = "I finished my project"
text_t+1 = ??? (unseen)
predict_change = ???  # How do we predict change to unknown future text?
```

**Solution Approaches**:

1. **Use historical patterns** (what we're doing ✅):
   - User embeddings capture personal patterns
   - Lag features show recent trajectory
   - Model learns typical change magnitude per user

2. **Predict "expected next state"**:
   - Model learns users' emotional trajectories
   - Predicts likely next emotional state based on history

3. **Ensemble averaging**:
   - Multiple models capture different aspects
   - Reduces prediction variance

---

## 📈 Performance Targets

### Competitive Ranges (Based on similar SemEval tasks)

| CCC Range | Pearson r Range | Performance Level | Ranking |
|-----------|-----------------|-------------------|---------|
| 0.70+ | 0.75+ | Excellent | Top 5-10% |
| 0.60-0.70 | 0.65-0.75 | Very Good | Top 10-30% |
| 0.50-0.60 | 0.55-0.65 | Good | Top 30-50% |
| 0.40-0.50 | 0.45-0.55 | Acceptable | Top 50-70% |
| < 0.40 | < 0.45 | Needs Work | Bottom 30% |

### Our Model's Expected Performance

**Validation Performance**:
- Best single model: CCC 0.6554
- Ensemble: CCC 0.5846-0.6046

**Expected Test Performance**:
- Pearson r: **0.60-0.65** (accounting for CCC→r conversion)
- Target ranking: **Top 20-40%** (competitive performance)

---

## ✅ Action Items

### Immediate (Before Test Data Release)

- [x] Understand evaluation metrics
- [ ] Verify our model optimizes for the right metric
- [ ] Consider adding Pearson r to training loss
- [ ] Test prediction pipeline on validation set
- [ ] Calculate both CCC and Pearson r on validation

### Upon Test Data Release

- [ ] Check `is_forecasting_user` marker
- [ ] Generate predictions for forecasting users only
- [ ] Calculate both Pearson r and MAE
- [ ] Submit to Codabench
- [ ] Analyze results vs validation performance

### Model Improvements (Optional)

**If we want to optimize specifically for Pearson r**:
```python
# Option 1: Add Pearson r loss component
loss = 0.5 * CCC_loss + 0.3 * MSE_loss + 0.2 * (1 - pearson_r_loss)

# Option 2: Use correlation-based loss only
loss = 1 - pearson_r(predictions, targets)
```

**Trade-off**: Pearson r doesn't penalize scale errors, but CCC does (more robust).

---

## 📚 Summary

### Key Takeaways

1. **Subtask 1**: Complex evaluation with between-user, within-user, and composite metrics
2. **Subtask 2A**: Simpler per-user Pearson r and MAE evaluation
3. **Ranking**: Based on Pearson r for Subtask 2A
4. **Our Model**: Trained on CCC, expect similar or slightly better Pearson r
5. **Test Data**: No new texts, predict change to unknown future state
6. **Target**: 0.60-0.65 Pearson r for competitive performance

### Confidence Assessment

| Aspect | Confidence Level | Notes |
|--------|------------------|-------|
| Understanding metrics | ✅ High | Clear documentation |
| Model compatibility | ✅ High | CCC correlates with Pearson r |
| Performance target | ⚠️ Medium | Validation ≠ test performance |
| Test data format | ⚠️ Medium | Need to see actual release |

---

**Document Status**: ✅ Complete
**Next Update**: After test data release (December 2025)
**Questions**: Contact task organizers at nisoni@cs.stonybrook.edu
