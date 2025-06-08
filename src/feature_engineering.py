import numpy as np
import pandas as pd

def case_to_dataframe(trip_duration_days, miles_traveled, total_receipts_amount):
    """Convert a single case to a pandas DataFrame.
    
    Args:
        trip_duration_days: Number of days for the trip
        miles_traveled: Number of miles traveled
        total_receipts_amount: Total amount of receipts
        
    Returns:
        DataFrame with a single row containing the input features
    """
    return pd.DataFrame({
        'trip_duration_days': [trip_duration_days],
        'miles_traveled': [miles_traveled],
        'total_receipts_amount': [total_receipts_amount]
    })

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # ===== PER DIEM FEATURES =====
    # Special day flags based on analysis
    df['is_1_day_trip'] = (df['trip_duration_days'] == 1).astype(int)
    df['is_2_day_trip'] = (df['trip_duration_days'] == 2).astype(int)
    df['is_3_day_trip'] = (df['trip_duration_days'] == 3).astype(int)
    df['is_4_day_trip'] = (df['trip_duration_days'] == 4).astype(int)
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)  # Penalty identified
    df['is_7_day_trip'] = (df['trip_duration_days'] == 7).astype(int)  # Bonus identified
    df['is_14_day_trip'] = (df['trip_duration_days'] == 14).astype(int)
    
    # Per diem tiers
    df['short_trip'] = (df['trip_duration_days'] <= 3).astype(int)
    df['medium_trip'] = ((df['trip_duration_days'] > 3) & (df['trip_duration_days'] <= 7)).astype(int)
    df['long_trip'] = (df['trip_duration_days'] > 7).astype(int)
    
    # Base per diem allowance (based on project memory: ~$100/day)
    df['base_per_diem'] = 100.00      # Feature expected by ML models, now using the flat $100 rate.
    df['base_allowance'] = df['base_per_diem'] * df['trip_duration_days'] # Calculate allowance using the updated base_per_diem.
    
    # ===== MILEAGE RATE TIERS =====
    # Mileage in each tier (based on analysis)
    df['miles_tier1'] = np.minimum(df['miles_traveled'], 100)  # 0-100 miles: ~$19.71/mile
    df['miles_tier2'] = np.maximum(0, np.minimum(df['miles_traveled'] - 100, 150))  # 100-250 miles: ~$6.22/mile
    df['miles_tier3'] = np.maximum(0, np.minimum(df['miles_traveled'] - 250, 250))  # 250-500 miles: ~$3.25/mile
    df['miles_tier4'] = np.maximum(0, np.minimum(df['miles_traveled'] - 500, 250))  # 500-750 miles: ~$2.23/mile
    df['miles_tier5'] = np.maximum(0, np.minimum(df['miles_traveled'] - 750, 250))  # 750-1000 miles: ~$1.77/mile
    df['miles_tier6'] = np.maximum(0, df['miles_traveled'] - 1000)  # 1000+ miles: ~$1.45/mile
    
    # Estimated reimbursement by tier
    df['miles_reimb_t1'] = df['miles_tier1'] * 0.97
    df['miles_reimb_t2'] = df['miles_tier2'] * 0.62
    df['miles_reimb_t3'] = df['miles_tier3'] * 0.325
    df['miles_reimb_t4'] = df['miles_tier4'] * 0.223
    df['miles_reimb_t5'] = df['miles_tier5'] * 0.177
    df['miles_reimb_t6'] = df['miles_tier6'] * 0.145
    
    # Total estimated mileage reimbursement
    df['est_mileage_reimb'] = df['miles_reimb_t1'] + df['miles_reimb_t2'] + df['miles_reimb_t3'] + \
                             df['miles_reimb_t4'] + df['miles_reimb_t5'] + df['miles_reimb_t6']
    
    # ===== RECEIPT UTILIZATION =====
    # Receipt amount in each tier (based on analysis)
    df['receipts_tier1'] = np.minimum(df['total_receipts_amount'], 50)  # 0-50: 27.6x
    df['receipts_tier2'] = np.maximum(0, np.minimum(df['total_receipts_amount'] - 50, 50))  # 50-100: 9.1x
    df['receipts_tier3'] = np.maximum(0, np.minimum(df['total_receipts_amount'] - 100, 150))  # 100-250: 4.9x
    df['receipts_tier4'] = np.maximum(0, np.minimum(df['total_receipts_amount'] - 250, 250))  # 250-500: 2.0x
    df['receipts_tier5'] = np.maximum(0, np.minimum(df['total_receipts_amount'] - 500, 500))  # 500-1000: 1.6x
    df['receipts_tier6'] = np.maximum(0, np.minimum(df['total_receipts_amount'] - 1000, 1000))  # 1000-2000: 1.1x
    df['receipts_tier7'] = np.maximum(0, df['total_receipts_amount'] - 2000)  # 2000+: 0.7x
    
    # Estimated reimbursement by tier
    df['receipts_reimb_t1'] = df['receipts_tier1'] * 2.76
    df['receipts_reimb_t2'] = df['receipts_tier2'] * 0.91
    df['receipts_reimb_t3'] = df['receipts_tier3'] * 0.49
    df['receipts_reimb_t4'] = df['receipts_tier4'] * 0.20
    df['receipts_reimb_t5'] = df['receipts_tier5'] * 0.16
    df['receipts_reimb_t6'] = df['receipts_tier6'] * 0.11
    df['receipts_reimb_t7'] = df['receipts_tier7'] * 0.07
    
    # Total estimated receipt reimbursement
    df['est_receipt_reimb'] = df['receipts_reimb_t1'] + df['receipts_reimb_t2'] + df['receipts_reimb_t3'] + \
                              df['receipts_reimb_t4'] + df['receipts_reimb_t5'] + df['receipts_reimb_t6'] + \
                              df['receipts_reimb_t7']

    # Special handling for 1-day trips for mileage and receipts
    one_day_mask = (df['trip_duration_days'] == 1)
    if one_day_mask.any(): # Apply only if there are any 1-day trips to avoid issues with empty masks
        df.loc[one_day_mask, 'est_mileage_reimb'] = df.loc[one_day_mask, 'miles_traveled'] * 0.5
        df.loc[one_day_mask, 'est_receipt_reimb'] = df.loc[one_day_mask, 'total_receipts_amount'] * 0.2
    
    # ===== COMBINED FORMULA ESTIMATE =====
    # This is a crucial feature for the ML model, representing a strong baseline.
    df['formula_estimate'] = df['base_allowance'] + df['est_mileage_reimb'] + df['est_receipt_reimb']
    
    # ===== SPECIAL INTERACTION CASES =====
    # Create interaction flags for special cases identified in analysis
    df['special_2day_med_miles'] = ((df['trip_duration_days'] == 2) & 
                                   (df['miles_traveled'] > 100) & 
                                   (df['miles_traveled'] <= 500)).astype(int)  # 30% bonus
    
    df['special_2day_high_miles'] = ((df['trip_duration_days'] == 2) & 
                                    (df['miles_traveled'] > 1000)).astype(int)  # 17.5% bonus
    
    df['special_5day_low_miles'] = ((df['trip_duration_days'] == 5) & 
                                   (df['miles_traveled'] <= 100)).astype(int)  # 30.6% bonus
    
    # ===== TRADITIONAL ML FEATURES =====
    # Basic ratios
    df['cost_per_mile'] = df['total_receipts_amount'] / (df['miles_traveled'].replace(0, np.nan))
    df['cost_per_day'] = df['total_receipts_amount'] / (df['trip_duration_days'].replace(0, np.nan))
    df['miles_per_day'] = df['miles_traveled'] / (df['trip_duration_days'].replace(0, np.nan))
    
    # Binned features
    mileage_bins = [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, np.inf]
    receipts_bins = [0, 50, 100, 250, 500, 1000, 2000, 5000, np.inf]
    df['mileage_bin'] = pd.cut(df['miles_traveled'], bins=mileage_bins, labels=False).fillna(0).astype(int)
    df['receipts_bin'] = pd.cut(df['total_receipts_amount'], bins=receipts_bins, labels=False).fillna(0).astype(int)
    
    # Polynomial/interaction features
    df['miles_x_duration'] = df['miles_traveled'] * df['trip_duration_days']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    df['duration_x_receipts'] = df['trip_duration_days'] * df['total_receipts_amount']
    
    # Non-linear transforms
    df['log_miles'] = np.log1p(df['miles_traveled'])
    df['log_receipts'] = np.log1p(df['total_receipts_amount'])
    df['sqrt_miles'] = np.sqrt(df['miles_traveled'])
    df['sqrt_receipts'] = np.sqrt(df['total_receipts_amount'])
    
    # ===== COMBINED ESTIMATE =====
    # This feature attempts to directly model the reimbursement formula
    df['formula_estimate'] = df['base_allowance'] + df['est_mileage_reimb'] + df['est_receipt_reimb']
    
    # Replace any inf/nan with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

def single_case_to_df(trip_duration_days, miles_traveled, total_receipts_amount):
    return pd.DataFrame([{
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }])
