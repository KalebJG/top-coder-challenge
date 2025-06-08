"""
Rule-based corrections for reimbursement predictions.
This module applies post-processing rules to ML predictions to better match legacy logic patterns.
"""
import numpy as np
import pandas as pd

def apply_rule_corrections(inputs_df, predictions):
    """
    Apply rule-based corrections to ML predictions based on identified legacy patterns.
    
    Args:
        inputs_df: DataFrame with input features (trip_duration_days, miles_traveled, total_receipts_amount)
        predictions: Array of ML predictions to correct
        
    Returns:
        Array of corrected predictions
    """
    # Create a copy of predictions to modify
    corrected = predictions.copy()
    
    # Extract key inputs
    trip_days = inputs_df['trip_duration_days'].values
    miles = inputs_df['miles_traveled'].values
    receipts = inputs_df['total_receipts_amount'].values
    
    # ===== SPECIAL CASE CORRECTIONS =====
    
    # 5-day trip penalty (identified in analysis)
    five_day_mask = (trip_days == 5)
    if np.any(five_day_mask):
        # Apply a small penalty to 5-day trips
        corrected[five_day_mask] *= 0.95  # 5% reduction
    
    # 7-day trip bonus (identified in analysis)
    seven_day_mask = (trip_days == 7)
    if np.any(seven_day_mask):
        # Apply a small bonus to 7-day trips
        corrected[seven_day_mask] *= 1.06  # 6% increase
    
    # 2-day trip with medium mileage bonus
    special_2day_med_miles = (trip_days == 2) & (miles > 100) & (miles <= 500)
    if np.any(special_2day_med_miles):
        corrected[special_2day_med_miles] *= 1.15  # 15% increase
    
    # 2-day trip with high mileage bonus
    special_2day_high_miles = (trip_days == 2) & (miles > 1000)
    if np.any(special_2day_high_miles):
        corrected[special_2day_high_miles] *= 1.10  # 10% increase
    
    # 5-day trip with low mileage bonus
    special_5day_low_miles = (trip_days == 5) & (miles <= 100)
    if np.any(special_5day_low_miles):
        corrected[special_5day_low_miles] *= 1.15  # 15% increase
    
    # ===== ROUNDING CORRECTIONS =====
    
    # Round to nearest 5 cents (common in reimbursement systems)
    corrected = np.round(corrected * 20) / 20
    
    return corrected
