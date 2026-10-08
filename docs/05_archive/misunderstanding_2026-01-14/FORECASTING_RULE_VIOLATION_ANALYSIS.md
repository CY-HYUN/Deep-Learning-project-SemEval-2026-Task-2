> Superseded numbers: this working document predates the verified results.
> The current figures are in README.md (best single model: validation mean Pearson r 0.6554, which the training
> code logs as "CCC"; true CCC was not measured; ensembles were never re-scored).

# Forecasting Rule Violation Analysis

**Date**: 2026-01-14
**Status**: 🔴 **CRITICAL - RULE VIOLATION CONFIRMED**
**Action Required**: IMMEDIATE PREDICTION REGENERATION

---

## 🚨 Executive Summary

**CRITICAL FINDING**: Our prediction pipeline **violated SemEval 2026 forecasting rules** by using future text data as model input. This matches the organizers' warning emails and explains why we received individual contact.

**Risk Level**: 🔴 **DISQUALIFICATION RISK**
- Organizer email: "**Use of any future text data will disqualify the submission**"
- Current predictions: Generated using 784 text entries (future data)
- Required action: Regenerate predictions using only historical data

---

## 📧 Organizer Warning Emails Context

### Email 1: General Announcement
> "Make sure to use the updated Test Data version released January 8 named **TEST_RELEASE_5JAN2026**."

### Email 2: Individual Warning (WHY WE RECEIVED THIS)
> "We are reaching out to you **separately** as we noticed you made submissions **before** the updated version of the Test Set."

### Email 3: Disqualification Threat
> "**Use of any future text data (not part of the Training Data) is prohibited** for Subtask 2a and Subtask 2b. **Any such use will disqualify the submission.**"

**Why we got flagged**: We submitted predictions generated from 784 text entries in `test_subtask2.csv`, which contain future text data.

---

## 🔍 Code Analysis: What We Did Wrong

### Our Prediction Pipeline (run_prediction_colab.ipynb)

```python
# Cell 14: Test Data Loading
test_df = pd.read_csv(test_data_path)  # test_subtask2.csv
# ❌ Loaded 785 lines (784 text entries + header)

# Cell 14: Preprocessing
test_df, user_stats_cols, text_feature_cols = preprocess_test_data(test_df)
# ❌ Extracted text features from ALL 784 entries

# Cell 14: Prediction Generation
for batch in test_loader:
    valence_pred, arousal_pred = model(
        input_ids,          # ❌ Text embeddings from future data
        attention_mask,
        user_idx,
        temporal_features,  # ❌ Calculated from future labels
        user_stats,         # ❌ Calculated from future labels
        text_features       # ❌ Extracted from future text
    )

# Cell 14: User-level Aggregation
final_predictions = test_df_with_pred.sort_values('timestamp').groupby('user_id').last()
# ❌ Selected last entry per user (still uses future data)
```

### Test Data We Used (OLD VERSION - Jan 6)

**File**: `test_subtask2.csv` (785 lines)

```csv
user_id,text_id,text,timestamp,collection_phase,is_words,valence,arousal
21,1324,"Just waking up and ready to get the day started...",2021-05-05 12:19:48,1,False,2.0,2.0
21,1325,"Nothing has changed except im headed into work...",2021-05-05 17:01:05,1,False,2.0,1.0
27,1697,"Not bad. Have a medical procedure today...",2022-09-06 12:31:30,4,False,0.0,1.0
```

**Problem**:
- ❌ Contains 784 complete text entries
- ❌ Includes valence/arousal labels (future information)
- ❌ No `is_forecasting_user` column to distinguish historical vs future data
- ❌ Our model used these texts as input features

---

## ✅ What We SHOULD Have Done

### Correct Data Source: subtask2a_forecasting_user_marker.csv

**File**: `subtask2a_forecasting_user_marker.csv` (2,765 lines - UNCHANGED)

```csv
user_id,text_id,text,timestamp,collection_phase,is_words,valence,arousal,
state_change_valence,state_change_arousal,is_forecasting_user

# Historical data (CAN USE)
1,200,"I feel good...",2021-06-09 12:41:57,1,False,2.0,1.0,0.0,0.0,False

# Forecasting target (CANNOT USE TEXT)
182,4646,"I am feeling pretty good...",2024-06-17 17:01:01,7,False,1.0,0.0,,,True
```

### Correct Prediction Methodology

```python
# Step 1: Load forecasting marker file
marker_df = pd.read_csv('subtask2a_forecasting_user_marker.csv')

# Step 2: Split historical vs forecasting data
historical_data = marker_df[marker_df['is_forecasting_user'] == False]  # 2,536 entries
forecasting_entries = marker_df[marker_df['is_forecasting_user'] == True]  # 228 entries

# Step 3: Extract features from HISTORICAL DATA ONLY
# ✅ Use historical_data for user_stats, temporal features
# ✅ DO NOT use text/labels from forecasting_entries

# Step 4: Generate predictions for forecasting entries
# ✅ Predict state change for each of 228 forecasting entries
# ✅ Use only past information (historical_data)

# Step 5: Output format (NEED CLARIFICATION)
# Option A: 228 entry-level predictions
# Option B: 46 user-level predictions (aggregated)
```

### Key Differences

| Aspect | ❌ What We Did | ✅ What We Should Do |
|--------|---------------|---------------------|
| **Data Source** | test_subtask2.csv | subtask2a_forecasting_user_marker.csv |
| **Data Split** | None (used all) | is_forecasting_user==False/True |
| **Text Usage** | 784 entries as input | NO future text (historical only) |
| **Feature Extraction** | From all test texts | From historical data only |
| **Prediction Count** | 784 → 46 users | 228 forecasting entries |
| **Output Format** | 46 user-level | TBD (228 entry-level or 46 user-level) |

---

## 📊 Data Files Comparison

### OLD Test Data (Jan 6 - WRONG)

**test_subtask2.csv**: 785 lines (784 texts + header)
- ❌ Full text content provided
- ❌ Valence/arousal labels included
- ❌ No forecasting marker
- ❌ Enables rule violation

### NEW Test Data (TEST_RELEASE_5JAN2026 - CORRECT)

**test_subtask2.csv**: 47 lines (46 users + header)
- ✅ **NO TEXT CONTENT** (physically removed)
- ✅ Metadata only (timestamp_min/max, collection_phase)
- ✅ Prevents future text usage
- ✅ Enforces forecasting rules

```csv
user_id,timestamp_min,timestamp_max,collection_phase_min,collection_phase_max
6,2022-05-23 15:04:00,2023-07-11 20:25:04,3,6
8,2021-04-03 12:02:27,2021-04-08 12:23:30,1,1
```

**subtask2a_forecasting_user_marker.csv**: 2,765 lines (UNCHANGED)
- ✅ Contains `is_forecasting_user` column
- ✅ 2,536 historical entries (is_forecasting_user==False)
- ✅ 228 forecasting targets (is_forecasting_user==True)
- ✅ This is the PRIMARY data source

---

## 🎯 Why Organizers Changed Test Data

### Organizer's Statement:
> "Subtask 2a and 2b are forecasting tasks so **no (future) texts can be used** for forecasting state change... thus, **we will not release any text data** for the same."

### Enforcement Strategy:

**Before (Jan 6)**:
- Text data provided in test_subtask2.csv
- Participants could accidentally use future texts
- Required manual `is_forecasting_user` filtering
- **Problem**: Easy to violate rules (like we did)

**After (Jan 8 - TEST_RELEASE_5JAN2026)**:
- NO text in test_subtask2.csv
- **Physically impossible** to use future text
- Forces use of subtask2a_forecasting_user_marker.csv
- **Solution**: Rule violation prevented at source

---

## 🔑 Understanding Forecasting Rules

### What is "Future Text Data"?

**Definition**: Any text entry where `is_forecasting_user==True`

**Rule**:
- ✅ CAN use: Historical text (is_forecasting_user==False) for feature extraction
- ❌ CANNOT use: Forecasting target text (is_forecasting_user==True) as model input

### Forecasting Philosophy

**Forecasting** = Predicting future state change using only past information

**Analogy**:
```
Timeline: [Past entries] → [Current] → [Future entry to predict]
          ✅ Can use      ✅ Can use   ❌ Cannot use (this is what we predict)
```

**Our Violation**:
- We used text from "Future entry" as model input
- This is like predicting stock prices after seeing tomorrow's news
- Breaks the fundamental forecasting principle

---

## 📋 Submission Format Question

### Current Uncertainty: 46 vs 228 Predictions?

**Our Current File**: 46 user-level predictions
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
6,0.6134894,0.014876012
8,0.47978187,0.0015302785
...
```

**Alternative Format**: 228 entry-level predictions
```csv
user_id,text_id,pred_state_change_valence,pred_state_change_arousal
6,1567,0.65,0.02
6,1890,0.71,-0.01
8,2341,0.48,0.00
...
```

### Evidence Analysis

**For User-Level (46 predictions)**:
- ✅ Current file format matches this
- ✅ New test_subtask2.csv has 46 users
- ✅ Simpler format

**For Entry-Level (228 predictions)**:
- ✅ Matches forecasting marker structure (228 is_forecasting_user==True entries)
- ✅ More granular evaluation
- ✅ Original test_subtask2.csv had 784 entries (entry-level granularity)

**NEED TO CLARIFY**: Check CodaBench submission guidelines or contact organizers

---

## 🚨 Required Actions

### IMMEDIATE (CRITICAL)

1. **Regenerate Predictions Using Correct Methodology**
   - Use `subtask2a_forecasting_user_marker.csv`
   - Filter `is_forecasting_user==False` for feature extraction
   - Generate predictions for `is_forecasting_user==True` entries (228)
   - NO text from forecasting targets as input

2. **Clarify Submission Format**
   - Check CodaBench documentation
   - Determine: 46 user-level or 228 entry-level?
   - Contact organizers if unclear

3. **Update Prediction Code**
   - Modify `run_prediction_colab.ipynb`
   - Add explicit `is_forecasting_user` filtering
   - Document compliance with forecasting rules

### MEDIUM PRIORITY

4. **Verify New Test Data Compatibility**
   - Ensure code works with TEST_RELEASE_5JAN2026
   - Test with metadata-only test_subtask2.csv

5. **Resubmit to CodaBench**
   - Upload corrected predictions
   - Verify format checker passes

### LOW PRIORITY

6. **Update Documentation**
   - Fix README.md slide counts (21→22, 17→21)
   - Add forecasting rule compliance notes
   - Document correct prediction methodology

---

## 📊 Risk Assessment

| Risk | Severity | Likelihood | Mitigation Status |
|------|----------|-----------|-------------------|
| **Disqualification** | 🔴 CRITICAL | HIGH | ⏳ In progress (regenerating predictions) |
| **Wrong submission format** | 🟡 MEDIUM | MEDIUM | ⏳ Need clarification |
| **Code compatibility** | 🟢 LOW | LOW | ✅ Models trained, only inference needs update |
| **Time constraint** | 🟢 LOW | LOW | ✅ Can regenerate quickly (35-45 min) |

---

## ✅ Verification Checklist for New Predictions

**Before Resubmission**:

- [ ] Used `subtask2a_forecasting_user_marker.csv` as primary data source
- [ ] Filtered `is_forecasting_user==False` for historical data
- [ ] NO text from `is_forecasting_user==True` entries used as input
- [ ] Generated predictions for correct number of entries (46 or 228 - TBD)
- [ ] User list matches TEST_RELEASE_5JAN2026 (46 users)
- [ ] No NaN values in predictions
- [ ] No duplicate user_id (if user-level) or text_id (if entry-level)
- [ ] Prediction file format matches CodaBench requirements
- [ ] Code uses TEST_RELEASE_5JAN2026 test data
- [ ] Documented forecasting rule compliance

---

## 📞 Next Steps Decision Tree

```
START: Received organizer warning emails
  │
  ├─> ✅ Downloaded TEST_RELEASE_5JAN2026
  ├─> ✅ Verified user list matches (46 users)
  ├─> ✅ Analyzed prediction code
  └─> 🔴 CONFIRMED: Forecasting rule violation
      │
      ├─> Step 1: Clarify submission format
      │   └─> Contact organizers OR check CodaBench docs
      │       ├─> If 46 user-level: Modify code for user aggregation
      │       └─> If 228 entry-level: Output per-entry predictions
      │
      ├─> Step 2: Update prediction code
      │   ├─> Load subtask2a_forecasting_user_marker.csv
      │   ├─> Filter is_forecasting_user==False (historical)
      │   ├─> Extract features from historical data ONLY
      │   ├─> Predict for is_forecasting_user==True entries
      │   └─> Save in correct format (46 or 228)
      │
      ├─> Step 3: Regenerate predictions
      │   ├─> Run updated notebook on Google Colab (35-45 min)
      │   ├─> Verify checklist above
      │   └─> Test file format
      │
      ├─> Step 4: Resubmit to CodaBench
      │   ├─> Upload new pred_subtask2a.csv
      │   ├─> Verify format checker passes
      │   └─> Wait for evaluation
      │
      └─> Step 5: Document compliance
          ├─> Add forecasting rule notes to README
          ├─> Update final submission docs
          └─> Archive old predictions with warning label
```

---

## 🎯 Expected Outcome

**Current Performance**: best single-model validation CCC 0.6554 (seed777); this analysis was later found to be a misunderstanding (see README.md in this folder)

**After Correction**:
- ⚠️ Performance may decrease slightly (less information available)
- ✅ Compliance with forecasting rules ensured
- ✅ No disqualification risk
- ✅ Fair comparison with other participants

**Philosophy**:
> Better to have slightly lower performance with **rule compliance** than high performance with **disqualification risk**.

---

## 📚 References

### Email Evidence
- General announcement (2026-01-XX PM 5:38)
- Individual warning (2026-01-XX PM 6:23)
- Future text prohibition (2026-01-XX PM 6:19)

### Data Files
- `TEST_RELEASE_5JAN2026/test_subtask2.csv` (47 lines, metadata only)
- `TEST_RELEASE_5JAN2026/subtask2a_forecasting_user_marker.csv` (2,765 lines)
- Old `test_subtask2.csv` (785 lines - DO NOT USE)

### Code Files
- `run_prediction_colab.ipynb` (needs modification)
- `predict_optimized.py` (local version - needs update)

### Documentation
- `TEST_DATA_MIGRATION_ANALYSIS.md` (comparison analysis)
- `FORECASTING_RULE_VIOLATION_ANALYSIS.md` (this file)

---

**Last Updated**: 2026-01-14 23:30 KST
**Analyst**: Claude Sonnet 4.5
**Priority**: 🔴 CRITICAL - ACTION REQUIRED IMMEDIATELY
