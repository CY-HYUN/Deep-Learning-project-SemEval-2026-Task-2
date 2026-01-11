"""
Test Data Verification Script for SemEval 2026 Task 2, Subtask 2a
Verifies the structure and compatibility of downloaded test data.

Author: Hyun Chang-Yong
Date: January 6, 2026
"""

import pandas as pd
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
TEST_DIR = BASE_DIR / 'data' / 'test'
TEST_FILE = TEST_DIR / 'test_subtask2.csv'
MARKER_FILE = TEST_DIR / 'subtask2a_forecasting_user_marker.csv'

def verify_test_data():
    """Verify test data structure and identify prediction targets."""

    print("=" * 80)
    print("SemEval 2026 Task 2 - Test Data Verification")
    print("=" * 80)

    # 1. Check file existence
    print("\n[1] Checking file existence...")
    print(f"Test file: {TEST_FILE}")
    print(f"Exists: {TEST_FILE.exists()}")
    print(f"Marker file: {MARKER_FILE}")
    print(f"Exists: {MARKER_FILE.exists()}")

    if not TEST_FILE.exists() or not MARKER_FILE.exists():
        print("\n❌ ERROR: Test data files not found!")
        return

    # 2. Load test data
    print("\n[2] Loading test data...")
    test_df = pd.read_csv(TEST_FILE)
    marker_df = pd.read_csv(MARKER_FILE)

    print(f"Test data shape: {test_df.shape}")
    print(f"Marker data shape: {marker_df.shape}")

    # 3. Check columns
    print("\n[3] Checking columns...")
    print(f"Test columns: {list(test_df.columns)}")
    print(f"Marker columns: {list(marker_df.columns)}")

    # 4. Check is_forecasting_user field
    print("\n[4] Analyzing is_forecasting_user field...")

    if 'is_forecasting_user' in marker_df.columns:
        print(f"\nMarker file is_forecasting_user distribution:")
        print(marker_df['is_forecasting_user'].value_counts())
        print(f"\nPercentage:")
        print(marker_df['is_forecasting_user'].value_counts(normalize=True) * 100)

    # 5. Check prediction targets (empty state_change values)
    print("\n[5] Identifying prediction targets...")

    if 'state_change_valence' in marker_df.columns and 'state_change_arousal' in marker_df.columns:
        # Count rows with empty state_change values
        empty_valence = marker_df['state_change_valence'].isna()
        empty_arousal = marker_df['state_change_arousal'].isna()
        both_empty = empty_valence & empty_arousal

        print(f"\nRows with empty state_change_valence: {empty_valence.sum()}")
        print(f"Rows with empty state_change_arousal: {empty_arousal.sum()}")
        print(f"Rows with BOTH empty: {both_empty.sum()}")

        # Check is_forecasting_user for these rows
        if 'is_forecasting_user' in marker_df.columns:
            print(f"\nis_forecasting_user values for rows with empty state_change:")
            forecasting_targets = marker_df[both_empty]
            print(forecasting_targets['is_forecasting_user'].value_counts())
            print(f"\n✅ CRITICAL FINDING:")
            print(f"   Prediction targets have is_forecasting_user = {forecasting_targets['is_forecasting_user'].mode()[0] if len(forecasting_targets) > 0 else 'N/A'}")

    # 6. Check user_id distribution
    print("\n[6] Analyzing user_id distribution...")

    if 'user_id' in marker_df.columns:
        unique_users = marker_df['user_id'].nunique()
        print(f"Total unique users: {unique_users}")

        # Find users needing predictions
        if 'state_change_valence' in marker_df.columns:
            forecasting_users = marker_df[marker_df['state_change_valence'].isna()]['user_id'].unique()
            print(f"Users needing predictions: {len(forecasting_users)}")
            print(f"User IDs: {sorted(forecasting_users)[:20]}{'...' if len(forecasting_users) > 20 else ''}")

    # 7. Sample data inspection
    print("\n[7] Sample data inspection...")
    print("\nFirst 5 rows of marker file:")
    print(marker_df.head())

    print("\nRows needing prediction (first 5):")
    if 'state_change_valence' in marker_df.columns:
        needs_prediction = marker_df[marker_df['state_change_valence'].isna()]
        print(needs_prediction.head())

    # 8. Compatibility check with prediction script
    print("\n[8] Prediction script compatibility check...")

    required_cols_test = ['user_id', 'text_id', 'text', 'timestamp', 'valence', 'arousal']
    required_cols_marker = ['user_id', 'text_id', 'state_change_valence', 'state_change_arousal', 'is_forecasting_user']

    test_has_all = all(col in test_df.columns for col in required_cols_test)
    marker_has_all = all(col in marker_df.columns for col in required_cols_marker)

    print(f"\nTest file has all required columns: {test_has_all}")
    if not test_has_all:
        missing = [col for col in required_cols_test if col not in test_df.columns]
        print(f"  Missing: {missing}")

    print(f"Marker file has all required columns: {marker_has_all}")
    if not marker_has_all:
        missing = [col for col in required_cols_marker if col not in marker_df.columns]
        print(f"  Missing: {missing}")

    # 9. Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if 'state_change_valence' in marker_df.columns:
        num_predictions_needed = marker_df['state_change_valence'].isna().sum()
        print(f"✅ Test data loaded successfully")
        print(f"✅ Total predictions needed: {num_predictions_needed}")

        if 'is_forecasting_user' in marker_df.columns:
            forecasting_value = marker_df[marker_df['state_change_valence'].isna()]['is_forecasting_user'].mode()[0] if num_predictions_needed > 0 else 'N/A'
            print(f"✅ Prediction targets have is_forecasting_user = {forecasting_value}")
            print(f"\n⚠️  IMPORTANT: Update prediction script to filter by:")
            print(f"   is_forecasting_user == {forecasting_value}")
            print(f"   OR state_change_valence.isna()")

        print(f"\n✅ Data is ready for prediction generation")

    print("=" * 80)

    return test_df, marker_df

if __name__ == '__main__':
    test_df, marker_df = verify_test_data()
