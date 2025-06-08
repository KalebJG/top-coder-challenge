#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data(file_path):
    """Load test cases from JSON file into a pandas DataFrame"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract input and output values
    records = []
    for case in data:
        record = {
            'trip_days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        records.append(record)
    
    return pd.DataFrame(records)

def analyze_per_diem(df):
    """Analyze the per diem component based on trip duration"""
    print("\n=== PER DIEM ANALYSIS ===")
    
    # Group by trip days and analyze cases with minimal miles and receipts
    # This helps isolate the per diem component
    low_impact = df[(df['miles'] < 20) & (df['receipts'] < 5)]
    per_diem_data = low_impact.groupby('trip_days')['reimbursement'].mean().reset_index()
    
    if not per_diem_data.empty:
        print("Estimated per diem rates (from cases with minimal miles and receipts):")
        for _, row in per_diem_data.iterrows():
            print(f"Trip days: {row['trip_days']}, Avg reimbursement: ${row['reimbursement']:.2f}, Per day: ${row['reimbursement']/row['trip_days']:.2f}")
    
    # Check for special case for 5-day trips (mentioned in requirements)
    five_day_trips = df[df['trip_days'] == 5]
    if not five_day_trips.empty:
        print(f"\nFive-day trips average reimbursement: ${five_day_trips['reimbursement'].mean():.2f}")
    
    # Look for patterns in per-day rates across different trip durations
    trip_days_groups = df.groupby('trip_days')
    print("\nAverage reimbursement by trip duration:")
    for days, group in trip_days_groups:
        avg_reimb = group['reimbursement'].mean()
        per_day = avg_reimb / days
        print(f"Trip days: {days}, Avg total: ${avg_reimb:.2f}, Per day: ${per_day:.2f}")

def analyze_mileage(df):
    """Analyze the mileage component of reimbursement"""
    print("\n=== MILEAGE ANALYSIS ===")
    
    # Group by miles ranges to identify potential tiers
    miles_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    df['miles_bin'] = pd.cut(df['miles'], miles_bins)
    
    # For each trip duration, analyze how mileage affects reimbursement
    for days in sorted(df['trip_days'].unique()):
        day_df = df[df['trip_days'] == days]
        if len(day_df) < 5:  # Skip if too few data points
            continue
            
        print(f"\nTrip duration: {days} days")
        # Sort by miles to see the progression
        day_df = day_df.sort_values('miles')
        
        # Calculate the incremental change in reimbursement per mile
        if len(day_df) > 1:
            day_df['miles_diff'] = day_df['miles'].diff()
            day_df['reimb_diff'] = day_df['reimbursement'].diff()
            day_df['rate_per_mile'] = day_df['reimb_diff'] / day_df['miles_diff']
            
            # Group by miles bins and calculate average rate per mile
            miles_rates = day_df.groupby('miles_bin')['rate_per_mile'].mean()
            print("Average rate per mile by distance range:")
            for bin_range, rate in miles_rates.items():
                if not np.isnan(rate):
                    print(f"  Miles {bin_range}: ${rate:.4f}/mile")

def analyze_receipts(df):
    """Analyze how receipt amounts affect reimbursement"""
    print("\n=== RECEIPTS ANALYSIS ===")
    
    # Group by receipt ranges to identify potential tiers
    receipt_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    df['receipt_bin'] = pd.cut(df['receipts'], receipt_bins)
    
    # For consistent trip duration and similar mileage, analyze receipt impact
    for days in sorted(df['trip_days'].unique()):
        day_df = df[df['trip_days'] == days]
        if len(day_df) < 5:  # Skip if too few data points
            continue
            
        # Find groups with similar mileage but different receipt amounts
        miles_bins = [0, 50, 100, 200, 500, float('inf')]
        day_df['miles_group'] = pd.cut(day_df['miles'], miles_bins)
        
        for miles_group, group_df in day_df.groupby('miles_group'):
            if len(group_df) < 5:  # Skip if too few data points
                continue
                
            print(f"\nTrip duration: {days} days, Miles: {miles_group}")
            # Sort by receipt amount
            group_df = group_df.sort_values('receipts')
            
            # Calculate the incremental change in reimbursement per dollar of receipts
            if len(group_df) > 1:
                group_df['receipt_diff'] = group_df['receipts'].diff()
                group_df['reimb_diff'] = group_df['reimbursement'].diff()
                group_df['rate_per_receipt_dollar'] = group_df['reimb_diff'] / group_df['receipt_diff']
                
                # Group by receipt bins and calculate average rate per receipt dollar
                receipt_rates = group_df.groupby('receipt_bin')['rate_per_receipt_dollar'].mean()
                print("Average rate per receipt dollar by amount range:")
                for bin_range, rate in receipt_rates.items():
                    if not np.isnan(rate):
                        print(f"  Receipts {bin_range}: ${rate:.4f}/dollar")

def look_for_special_cases(df):
    """Look for special cases or quirks in the reimbursement calculation"""
    print("\n=== SPECIAL CASES ANALYSIS ===")
    
    # Check for rounding patterns
    cents_distribution = defaultdict(int)
    for reimb in df['reimbursement']:
        cents = int(round((reimb * 100) % 100))
        cents_distribution[cents] += 1
    
    print("Distribution of cents in reimbursement amounts:")
    for cents, count in sorted(cents_distribution.items()):
        if count > 5:  # Only show common patterns
            print(f"  {cents:02d} cents: {count} occurrences ({count/len(df)*100:.1f}%)")
    
    # Check for patterns with specific receipt amounts ending in .49 or .99
    special_receipts = df[(df['receipts'] * 100) % 100 == 49]
    if not special_receipts.empty:
        print("\nCases with receipts ending in .49:")
        for _, row in special_receipts.head(5).iterrows():
            print(f"  Days: {row['trip_days']}, Miles: {row['miles']:.1f}, Receipts: ${row['receipts']:.2f}, Reimbursement: ${row['reimbursement']:.2f}")
    
    special_receipts = df[(df['receipts'] * 100) % 100 == 99]
    if not special_receipts.empty:
        print("\nCases with receipts ending in .99:")
        for _, row in special_receipts.head(5).iterrows():
            print(f"  Days: {row['trip_days']}, Miles: {row['miles']:.1f}, Receipts: ${row['receipts']:.2f}, Reimbursement: ${row['reimbursement']:.2f}")

def main():
    file_path = 'public_cases.json'
    df = load_data(file_path)
    
    print(f"Loaded {len(df)} test cases")
    print("\nBasic statistics:")
    print(df.describe())
    
    # Analyze each component separately
    analyze_per_diem(df)
    analyze_mileage(df)
    analyze_receipts(df)
    look_for_special_cases(df)
    
    # Save analysis results to a file
    with open('analysis_results.txt', 'w') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        print(f"Analysis of {len(df)} test cases")
        print("\nBasic statistics:")
        print(df.describe())
        
        analyze_per_diem(df)
        analyze_mileage(df)
        analyze_receipts(df)
        look_for_special_cases(df)
        
        sys.stdout = original_stdout
    
    print("\nAnalysis results saved to analysis_results.txt")

if __name__ == "__main__":
    main()
