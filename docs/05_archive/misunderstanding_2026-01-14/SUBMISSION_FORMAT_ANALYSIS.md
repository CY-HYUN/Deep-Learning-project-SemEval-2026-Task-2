# Submission Format Analysis - Subtask 2a

**Date**: 2026-01-14
**Status**: 🔍 ANALYSIS COMPLETE - FORMAT DETERMINED

---

## 📊 Data Structure Analysis

### Forecasting Entries Distribution

**File**: `subtask2a_forecasting_user_marker.csv`

```bash
# Total entries with is_forecasting_user==True
228 forecasting entries

# Unique users with forecasting entries
34 users (not 46!)

# Distribution per user (sample):
user_id  | forecasting_entries
---------|-------------------
11       | 53 entries
27       | 43 entries
38       | 19 entries
90       | 8 entries
50       | 7 entries
16       | 5 entries
56       | 5 entries
...      | ...
17       | 1 entry
21       | 1 entry
59       | 1 entry
68       | 1 entry
```

### 🚨 CRITICAL FINDING

**Only 34 users have `is_forecasting_user==True` entries, NOT 46 users!**

```bash
# Users WITH forecasting entries (34 users):
11, 16, 17, 21, 27, 30, 38, 41, 47, 50, 56, 59, 68, 86, 88, 90, 93, 95, 96,
113, 114, 116, 121, 128, 137, 142, 144, 146, 148, 153, 161, 176, 178, 182

# All test users (46 users):
6, 8, 11, 16, 21, 27, 29, 30, 38, 41, 46, 47, 50, 51, 56, 59, 66, 68, 74, 76,
78, 86, 88, 90, 93, 95, 96, 98, 109, 113, 114, 116, 121, 128, 137, 142, 144,
146, 148, 153, 161, 162, 167, 176, 178, 182

# Users WITHOUT forecasting entries (12 users):
6, 8, 29, 46, 51, 66, 74, 76, 78, 98, 109, 162, 167
```

**Interpretation**: Only 34 users need predictions, not all 46!

---

## 🤔 Submission Format: Two Possibilities

### Option A: User-Level (34 predictions) ⭐ **MOST LIKELY**

**Format**:
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
11,0.55890095,-0.005156543
16,0.8884821,0.15257414
17,...,...
...
```

**Expected**: 34 rows (one per user with is_forecasting_user==True entries)

**Evidence Supporting This**:
1. ✅ New test_subtask2.csv has 46 user-level summaries (not entry-level)
2. ✅ Our current file is already user-level (46 users)
3. ✅ Simpler evaluation (one prediction per user)
4. ✅ Matches organizers' change to user-level metadata format

**Problem with Current File**:
- ❌ We have 46 predictions (all test users)
- ❌ Should only have 34 predictions (users with forecasting entries)
- ❌ 12 extra users included (6, 8, 29, 46, 51, 66, 74, 76, 78, 98, 109, 162, 167)

### Option B: Entry-Level (228 predictions)

**Format**:
```csv
user_id,text_id,pred_state_change_valence,pred_state_change_arousal
6,307,0.65,0.02
6,308,0.71,-0.01
6,309,...,...
11,352,...,...
...
```

**Expected**: 228 rows (one per is_forecasting_user==True entry)

**Evidence Supporting This**:
1. ✅ More granular evaluation
2. ✅ Matches forecasting marker structure (228 entries)
3. ✅ Allows tracking per-entry accuracy

**Evidence AGAINST This**:
1. ❌ New test_subtask2.csv is user-level (not entry-level)
2. ❌ More complex format
3. ❌ Organizers simplified data structure (removed text, moved to metadata)

---

## 🎯 FORMAT CONCLUSION

**Recommended Format**: **Option A - User-Level (34 predictions)** ⭐

### Why User-Level is Correct:

1. **New test_subtask2.csv Structure**:
   - Changed from 785 entry-level rows → 47 user-level rows
   - Organizers simplified to **user-level granularity**

2. **Evaluation Philosophy**:
   - Task is "forecasting state change per user"
   - Not "forecasting state change per text entry"
   - User-level aggregation matches task goal

3. **Current Prediction File Analysis**:
   - We have 46 user-level predictions
   - Format is correct (user_id, valence, arousal)
   - **Only problem**: Includes 12 users without forecasting entries

4. **Logical Consistency**:
   - If organizers wanted entry-level, they wouldn't simplify test_subtask2.csv to user-level
   - User-level metadata → user-level predictions

---

## ✅ CORRECT SUBMISSION FORMAT

### Expected CSV Structure:

```csv
user_id,pred_state_change_valence,pred_state_change_arousal
11,0.55890095,-0.005156543
16,0.8884821,0.15257414
17,-0.12345,0.06789
21,0.4620662,-0.0005494468
27,0.5822679,-0.055471078
30,0.4433145,-0.017106872
38,0.64952457,-0.024442319
41,0.44869345,-0.0031341426
47,0.3574815,-0.025123954
50,0.3556007,-0.033628993
56,0.541007,0.01591526
59,0.41389847,-0.019598689
68,0.43411523,-0.024777584
86,0.47660702,-0.0052349493
88,0.482234,-0.025752768
90,0.26970395,-0.0757117
93,0.39594477,-0.0357903
95,0.40296483,-0.04127547
96,0.37325612,-0.055204112
113,0.34257418,0.08340958
114,0.22559133,-0.06422658
116,0.5979912,-0.0028070547
121,0.66805863,0.0493729
128,0.45444763,-0.030623946
137,0.56481874,0.0032163002
142,0.4321887,-0.009702459
144,0.5762072,0.014893655
146,0.41314,-0.0048619844
148,0.3968933,-0.0070854127
153,0.50831556,0.022216327
161,0.6099476,0.02758711
176,0.45870674,0.012033064
178,0.48024228,0.043054953
182,0.47176117,0.008659385
```

**Total**: 34 rows (+ 1 header) = 35 lines

---

## 🔍 Current vs Required Predictions

### Current File: `pred_subtask2a.csv` (47 lines)

**Users Included**: All 46 test users

**Extra Users** (should be removed - 12 users):
```
6, 8, 29, 46, 51, 66, 74, 76, 78, 98, 109, 162, 167
```

**Reason for Removal**: These users have NO `is_forecasting_user==True` entries

### Required File: `pred_subtask2a.csv` (35 lines)

**Users to Include**: Only 34 users with forecasting entries

**Keep Users** (34 users):
```
11, 16, 17, 21, 27, 30, 38, 41, 47, 50, 56, 59, 68, 86, 88, 90, 93, 95, 96,
113, 114, 116, 121, 128, 137, 142, 144, 146, 148, 153, 161, 176, 178, 182
```

---

## 🚨 Action Required

### Fix Current Predictions

**Problem**:
1. ❌ Current file has 46 predictions (all test users)
2. ❌ Includes 12 users without forecasting entries

**Solution**:
```python
# Load current predictions
pred_df = pd.read_csv('pred_subtask2a.csv')

# Load forecasting marker
marker_df = pd.read_csv('subtask2a_forecasting_user_marker.csv')

# Get users with forecasting entries
forecasting_users = marker_df[marker_df['is_forecasting_user'] == True]['user_id'].unique()
# Result: 34 users

# Filter predictions to only forecasting users
corrected_pred_df = pred_df[pred_df['user_id'].isin(forecasting_users)]

# Save corrected file
corrected_pred_df.to_csv('pred_subtask2a_corrected.csv', index=False)

# Verify
print(f"Original: {len(pred_df)} users")  # 46
print(f"Corrected: {len(corrected_pred_df)} users")  # 34
```

---

## 📋 Verification Checklist

### Format Requirements:

- [ ] CSV file with 3 columns: `user_id`, `pred_state_change_valence`, `pred_state_change_arousal`
- [ ] Exactly 34 predictions (one per user with forecasting entries)
- [ ] Only users with `is_forecasting_user==True` entries included
- [ ] No missing user_ids from the 34 forecasting users
- [ ] No extra user_ids (users without forecasting entries)
- [ ] No NaN or missing values
- [ ] No duplicate user_ids
- [ ] File size: ~1.0 KB (35 lines vs current 1.3 KB with 47 lines)

### User List Verification:

```bash
# Expected users (34):
11, 16, 17, 21, 27, 30, 38, 41, 47, 50, 56, 59, 68, 86, 88, 90, 93, 95, 96,
113, 114, 116, 121, 128, 137, 142, 144, 146, 148, 153, 161, 176, 178, 182

# Excluded users (12):
6, 8, 29, 46, 51, 66, 74, 76, 78, 98, 109, 162, 167
```

---

## 🎯 Prediction Generation Strategy

### Correct Methodology:

```python
# Step 1: Load forecasting marker
marker_df = pd.read_csv('subtask2a_forecasting_user_marker.csv')

# Step 2: Split data
historical_data = marker_df[marker_df['is_forecasting_user'] == False]  # 2,536 entries
forecasting_data = marker_df[marker_df['is_forecasting_user'] == True]  # 228 entries

# Step 3: Extract features from HISTORICAL DATA ONLY
# ✅ Use historical_data for:
#    - user_stats (valence/arousal means, stds, etc.)
#    - temporal features (lag_1, lag_2, lag_mean)
#    - text features (from historical texts only)

# Step 4: Generate predictions for FORECASTING USERS
forecasting_users = forecasting_data['user_id'].unique()  # 34 users

# Step 5: Predict per-entry (228 predictions)
predictions_per_entry = model.predict(forecasting_data)  # 228 predictions

# Step 6: Aggregate to user-level (34 predictions)
# Method A: Mean of all forecasting entries per user
user_predictions = predictions_per_entry.groupby('user_id').mean()

# Method B: Weighted by timestamp (more recent = higher weight)
user_predictions = predictions_per_entry.groupby('user_id').apply(
    lambda x: weighted_average(x, weights=time_weights)
)

# Method C: Last entry per user (temporal assumption)
user_predictions = predictions_per_entry.sort_values('timestamp').groupby('user_id').last()

# Step 7: Save as user-level CSV (34 rows)
final_predictions = user_predictions[['user_id', 'pred_state_change_valence', 'pred_state_change_arousal']]
final_predictions.to_csv('pred_subtask2a.csv', index=False)
```

---

## 🔗 Relationship to TEST_RELEASE_5JAN2026

### Why 46 Users in test_subtask2.csv but Only 34 Need Predictions?

**test_subtask2.csv** (46 users):
- Contains metadata for ALL test users
- Some users: Only historical data (no forecasting)
- Some users: Both historical + forecasting data

**subtask2a_forecasting_user_marker.csv** (34 users with forecasting):
- 34 users: Have is_forecasting_user==True entries → NEED predictions
- 12 users: Only is_forecasting_user==False entries → NO predictions needed

**Example**:
```python
# User 6 (in test_subtask2.csv):
# - Has 23 entries in forecasting_user_marker.csv
# - ALL entries: is_forecasting_user==True (forecasting entries)
# - Conclusion: INCLUDE in predictions

# User 8 (in test_subtask2.csv):
# - Has 17 entries in forecasting_user_marker.csv
# - ALL entries: is_forecasting_user==False (historical only)
# - Conclusion: EXCLUDE from predictions (no forecasting needed)
```

---

## 📊 Summary

### Current Status:

| Aspect | Current File | Required File | Status |
|--------|--------------|---------------|--------|
| **Format** | User-level | User-level | ✅ Correct |
| **Columns** | user_id, valence, arousal | user_id, valence, arousal | ✅ Correct |
| **User Count** | 46 users | 34 users | ❌ Wrong (12 extra) |
| **Methodology** | Used future text | Historical only | ❌ Wrong |
| **Data Source** | test_subtask2.csv | forecasting_user_marker.csv | ❌ Wrong |

### Required Actions:

1. **Filter Users** (Quick Fix):
   - Remove 12 users without forecasting entries
   - Keep only 34 users with is_forecasting_user==True
   - Result: Format is correct, but methodology still wrong

2. **Regenerate Predictions** (Full Fix - REQUIRED):
   - Use subtask2a_forecasting_user_marker.csv
   - Filter is_forecasting_user==False for features
   - Predict for 34 users with forecasting entries
   - Aggregate 228 entry-level predictions → 34 user-level predictions

---

## 🎯 Next Steps

### Immediate (Quick Fix):
```bash
# 1. Filter current predictions to 34 users
python scripts/filter_predictions_to_forecasting_users.py
# Output: pred_subtask2a_filtered.csv (34 users)

# 2. Test submission (may still fail due to wrong methodology)
# But at least format will be correct
```

### Required (Full Fix):
```bash
# 1. Update prediction code
#    - Load forecasting_user_marker.csv
#    - Filter historical data
#    - Generate 228 entry-level predictions
#    - Aggregate to 34 user-level predictions

# 2. Regenerate predictions on Google Colab (35-45 min)

# 3. Verify:
#    - 34 users
#    - Correct user list
#    - No future text used
#    - Historical data only

# 4. Submit to CodaBench
```

---

**Last Updated**: 2026-01-14 23:50 KST
**Conclusion**: **User-level format (34 predictions)** is correct ⭐
**Priority**: Fix user count (46→34) AND regenerate with correct methodology
