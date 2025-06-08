#!/usr/bin/env python3
"""
Quick test of the enhanced hybrid reimbursement engine on a sample of cases.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hybrid_reimbursement_engine import HybridReimbursementEngine
from src.feature_engineering import case_to_dataframe, engineer_features
from src.rule_based_corrections import apply_rule_corrections

# Load public cases
with open('data/public_cases.json', 'r') as f:
    public_cases = json.load(f)

# Take a sample of 50 cases for quick testing
import random
random.seed(42)  # For reproducibility
sample_cases = random.sample(public_cases, 50)

# Convert to DataFrame for analysis
data = []
for case in sample_cases:
    data.append({
        'trip_duration_days': case['input']['trip_duration_days'],
        'miles_traveled': case['input']['miles_traveled'],
        'total_receipts_amount': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    })
df = pd.DataFrame(data)

print(f"Testing on {len(df)} sample cases...")

# Initialize hybrid engine with enhanced models
print("Initializing enhanced hybrid engine...")
engine = HybridReimbursementEngine(
    lookup_table_path='models/public_case_lookup.json',
    lightgbm_model_path='models/lightgbm_model_enhanced.joblib',
    random_forest_model_path='models/random_forest_model_enhanced.joblib'
)

# Test on sample cases
print("Testing ML-only predictions (bypassing lookup) and intermediate steps...")
actuals = []
formula_estimates = []
raw_ml_ensemble_preds = []
corrected_ml_preds = []
final_blended_preds = [] # This is what ml_predictions was before

for _, row in df.iterrows():
    # Get actual value
    actual = row['expected']
    actuals.append(actual)

    # --- Replicate logic from engine._predict_with_ml for diagnostics ---
    trip_days = row['trip_duration_days']
    miles = row['miles_traveled']
    receipts_amount = row['total_receipts_amount']

    # 1. Feature Engineering & Formula Estimate
    case_df_single = case_to_dataframe(trip_days, miles, receipts_amount)
    features_df_single = engineer_features(case_df_single)
    # DEBUG: Check for duplicate columns
    if features_df_single.columns.has_duplicates:
        print(f"DEBUG WARNING: Duplicate columns in features_df_single: {features_df_single.columns[features_df_single.columns.duplicated()].tolist()}")
    formula_estimate_val = features_df_single['formula_estimate'].values[0]
    formula_estimates.append(formula_estimate_val)

    # Raw ML ensemble prediction
    # Ensure correct feature order for direct model predictions
    # LightGBM uses 'feature_name()' method (not 'feature_names_in_' attribute like scikit-learn)
    try:
        # In LightGBM, feature_name is a method, not an attribute
        lgbm_feature_order = engine.lightgbm_model.feature_name()
        features_df_single_lgbm = features_df_single[lgbm_feature_order]
        print(f"DEBUG: Using LightGBM feature_name() method: {len(lgbm_feature_order)} features")
    except (AttributeError, TypeError) as e:
        # If method call fails, use the DataFrame as-is
        print(f"DEBUG: Error getting LightGBM feature names: {e} - using DataFrame as-is")
        features_df_single_lgbm = features_df_single

    # RandomForest (scikit-learn) uses 'feature_names_in_'
    if hasattr(engine.random_forest_model, 'feature_names_in_'):
        rf_feature_order = engine.random_forest_model.feature_names_in_
        features_df_single_rf = features_df_single[rf_feature_order]
    else:
        # Fallback if attribute doesn't exist
        print("DEBUG: RandomForest model doesn't have feature_names_in_ attribute - using DataFrame as-is")
        features_df_single_rf = features_df_single

    lgbm_pred_val = engine.lightgbm_model.predict(features_df_single_lgbm)[0]

    # DEBUG: Compare RF columns before prediction
    if hasattr(engine.random_forest_model, 'feature_names_in_'):
        expected_rf_cols = list(engine.random_forest_model.feature_names_in_)
        actual_rf_cols = list(features_df_single_rf.columns)
        if actual_rf_cols != expected_rf_cols:
            print("DEBUG CRITICAL WARNING: RF Column mismatch despite reordering!")
            print(f"  Length Expected: {len(expected_rf_cols)}, Actual: {len(actual_rf_cols)}")
            for i, (e_col, a_col) in enumerate(zip(expected_rf_cols, actual_rf_cols)):
                if e_col != a_col:
                    print(f"  Mismatch at index {i}: Expected '{e_col}', Got '{a_col}'")
                    break # Show first mismatch
            if len(expected_rf_cols) != len(actual_rf_cols):
                print(f"  Set diff (expected - actual): {sorted(list(set(expected_rf_cols) - set(actual_rf_cols)))}")
                print(f"  Set diff (actual - expected): {sorted(list(set(actual_rf_cols) - set(expected_rf_cols)))}")
        else:
            print("DEBUG: RF columns match perfectly after reordering.")
    else:
        print("DEBUG: Cannot compare RF columns - feature_names_in_ not available")
    # else:
    #     print("DEBUG: Columns match and are in order for RF predict call.")

    rf_pred_val = engine.random_forest_model.predict(features_df_single_rf)[0]
    ensemble_pred_val = 0.7 * lgbm_pred_val + 0.3 * rf_pred_val
    raw_ml_ensemble_preds.append(ensemble_pred_val)

    # 3. Corrected ML Prediction
    # apply_rule_corrections expects inputs_df (original inputs), not features_df
    corrected_pred_val = apply_rule_corrections(case_df_single, np.array([ensemble_pred_val]))[0]
    corrected_ml_preds.append(corrected_pred_val)

    # 4. Blend ML with formula (80% ML, 20% Formula)
    blended_pred_val = 0.8 * corrected_pred_val + 0.2 * formula_estimate_val
    final_pred_val = max(0, round(blended_pred_val, 2))
    final_blended_preds.append(final_pred_val)
    
    # Print individual case results
    print(f"Case: {trip_days} days, {miles} miles, ${receipts_amount} receipts")
    print(f"  Expected: ${actual:.2f}")
    print(f"    FormulaEstimate: ${formula_estimate_val:.2f}, Error: ${formula_estimate_val - actual:.2f}")
    print(f"    RawMLPred:       ${ensemble_pred_val:.2f}, Error: ${ensemble_pred_val - actual:.2f}")
    print(f"    CorrectedMLPred: ${corrected_pred_val:.2f}, Error: ${corrected_pred_val - actual:.2f}")
    print(f"    FinalBlended:    ${final_pred_val:.2f}, Error: ${final_pred_val - actual:.2f}")

# Calculate metrics for each stage
stages = {
    "FormulaEstimate": formula_estimates,
    "RawMLEnsemble": raw_ml_ensemble_preds,
    "CorrectedML": corrected_ml_preds,
    "FinalBlended (ML-only)": final_blended_preds
}

for stage_name, predictions in stages.items():
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    exact_matches = sum(abs(a - p) < 1.0 for a, p in zip(actuals, predictions))
    close_matches = sum(abs(a - p) < 10.0 for a, p in zip(actuals, predictions))
    
    print(f"\nMetrics for {stage_name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Exact matches (within $1): {exact_matches}/{len(actuals)} ({exact_matches/len(actuals)*100:.2f}%)")
    print(f"  Close matches (within $10): {close_matches}/{len(actuals)} ({close_matches/len(actuals)*100:.2f}%)")

print("\nQuick test complete!")
