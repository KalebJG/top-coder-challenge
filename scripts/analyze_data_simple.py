#!/usr/bin/env python3
import json
import math
from collections import defaultdict

def load_data(file_path):
    """Load test cases from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_per_diem(data):
    """Analyze the per diem component based on trip duration"""
    print("\n=== PER DIEM ANALYSIS ===")
    
    # Group by trip days
    by_days = defaultdict(list)
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        by_days[days].append({
            'miles': miles,
            'receipts': receipts,
            'reimbursement': reimbursement
        })
    
    # Find minimum reimbursement for each trip duration (likely closest to base per diem)
    print("Minimum reimbursement by trip duration:")
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        if len(cases) < 2:
            continue
            
        # Find cases with minimal miles and receipts
        low_impact_cases = [c for c in cases if c['miles'] < 20 and c['receipts'] < 5]
        
        if low_impact_cases:
            min_case = min(low_impact_cases, key=lambda x: x['reimbursement'])
            print(f"Days: {days}, Min reimbursement: ${min_case['reimbursement']:.2f}, Per day: ${min_case['reimbursement']/days:.2f}")
        else:
            # If no low-impact cases, just use the minimum reimbursement
            min_case = min(cases, key=lambda x: x['reimbursement'])
            print(f"Days: {days}, Min reimbursement: ${min_case['reimbursement']:.2f}, Per day: ${min_case['reimbursement']/days:.2f}")
    
    # Check for special case for 5-day trips
    five_day_trips = by_days.get(5, [])
    if five_day_trips:
        avg_reimb = sum(c['reimbursement'] for c in five_day_trips) / len(five_day_trips)
        print(f"\nFive-day trips average reimbursement: ${avg_reimb:.2f}")

def analyze_mileage(data):
    """Analyze the mileage component of reimbursement"""
    print("\n=== MILEAGE ANALYSIS ===")
    
    # Group by trip days and then sort by miles
    by_days = defaultdict(list)
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        by_days[days].append({
            'miles': miles,
            'receipts': receipts,
            'reimbursement': reimbursement
        })
    
    # For each trip duration, analyze how mileage affects reimbursement
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        if len(cases) < 5:  # Skip if too few data points
            continue
            
        # Find cases with similar receipt amounts but different miles
        receipt_groups = defaultdict(list)
        for case in cases:
            receipt_bin = math.floor(case['receipts'] / 5) * 5  # Group receipts in $5 bins
            receipt_groups[receipt_bin].append(case)
        
        # Find a receipt group with enough data points
        for receipt_bin, group in receipt_groups.items():
            if len(group) >= 5:
                print(f"\nTrip duration: {days} days, Receipt bin: ${receipt_bin}-${receipt_bin+5}")
                
                # Sort by miles
                group.sort(key=lambda x: x['miles'])
                
                # Calculate the incremental change in reimbursement per mile
                prev_miles = None
                prev_reimb = None
                
                mile_rates = defaultdict(list)
                
                for i, case in enumerate(group):
                    if i > 0:
                        miles_diff = case['miles'] - prev_miles
                        reimb_diff = case['reimbursement'] - prev_reimb
                        
                        if miles_diff > 0:
                            rate = reimb_diff / miles_diff
                            
                            # Group rates by mile ranges
                            if prev_miles < 100:
                                mile_rates['0-100'].append(rate)
                            elif prev_miles < 500:
                                mile_rates['100-500'].append(rate)
                            else:
                                mile_rates['500+'].append(rate)
                    
                    prev_miles = case['miles']
                    prev_reimb = case['reimbursement']
                
                # Print average rates by mile range
                print("Average rate per mile by distance range:")
                for mile_range, rates in mile_rates.items():
                    if rates:
                        avg_rate = sum(rates) / len(rates)
                        print(f"  Miles {mile_range}: ${avg_rate:.4f}/mile")
                
                # Only process one receipt group per trip duration
                break

def analyze_receipts(data):
    """Analyze how receipt amounts affect reimbursement"""
    print("\n=== RECEIPTS ANALYSIS ===")
    
    # Group by trip days and then sort by receipt amount
    by_days = defaultdict(list)
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        by_days[days].append({
            'miles': miles,
            'receipts': receipts,
            'reimbursement': reimbursement
        })
    
    # For each trip duration, analyze how receipts affect reimbursement
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        if len(cases) < 5:  # Skip if too few data points
            continue
            
        # Find cases with similar mileage but different receipt amounts
        mile_groups = defaultdict(list)
        for case in cases:
            mile_bin = math.floor(case['miles'] / 50) * 50  # Group miles in 50-mile bins
            mile_groups[mile_bin].append(case)
        
        # Find a mile group with enough data points
        for mile_bin, group in mile_groups.items():
            if len(group) >= 5:
                print(f"\nTrip duration: {days} days, Mile bin: {mile_bin}-{mile_bin+50}")
                
                # Sort by receipts
                group.sort(key=lambda x: x['receipts'])
                
                # Calculate the incremental change in reimbursement per receipt dollar
                prev_receipts = None
                prev_reimb = None
                
                receipt_rates = defaultdict(list)
                
                for i, case in enumerate(group):
                    if i > 0:
                        receipt_diff = case['receipts'] - prev_receipts
                        reimb_diff = case['reimbursement'] - prev_reimb
                        
                        if receipt_diff > 0:
                            rate = reimb_diff / receipt_diff
                            
                            # Group rates by receipt ranges
                            if prev_receipts < 100:
                                receipt_rates['0-100'].append(rate)
                            elif prev_receipts < 400:
                                receipt_rates['100-400'].append(rate)
                            elif prev_receipts < 1000:
                                receipt_rates['400-1000'].append(rate)
                            else:
                                receipt_rates['1000+'].append(rate)
                    
                    prev_receipts = case['receipts']
                    prev_reimb = case['reimbursement']
                
                # Print average rates by receipt range
                print("Average rate per receipt dollar by amount range:")
                for receipt_range, rates in receipt_rates.items():
                    if rates:
                        avg_rate = sum(rates) / len(rates)
                        print(f"  Receipts {receipt_range}: ${avg_rate:.4f}/dollar")
                
                # Only process one mile group per trip duration
                break

def look_for_special_cases(data):
    """Look for special cases or quirks in the reimbursement calculation"""
    print("\n=== SPECIAL CASES ANALYSIS ===")
    
    # Check for rounding patterns
    cents_distribution = defaultdict(int)
    for case in data:
        reimb = case['expected_output']
        cents = int(round((reimb * 100) % 100))
        cents_distribution[cents] += 1
    
    print("Distribution of cents in reimbursement amounts:")
    for cents, count in sorted(cents_distribution.items()):
        if count > 5:  # Only show common patterns
            print(f"  {cents:02d} cents: {count} occurrences ({count/len(data)*100:.1f}%)")
    
    # Check for patterns with specific receipt amounts ending in .49 or .99
    special_receipts_49 = []
    special_receipts_99 = []
    
    for case in data:
        receipts = case['input']['total_receipts_amount']
        cents = int(round((receipts * 100) % 100))
        
        if cents == 49:
            special_receipts_49.append(case)
        elif cents == 99:
            special_receipts_99.append(case)
    
    if special_receipts_49:
        print("\nCases with receipts ending in .49:")
        for case in special_receipts_49[:5]:  # Show first 5
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            reimb = case['expected_output']
            print(f"  Days: {days}, Miles: {miles:.1f}, Receipts: ${receipts:.2f}, Reimbursement: ${reimb:.2f}")
    
    if special_receipts_99:
        print("\nCases with receipts ending in .99:")
        for case in special_receipts_99[:5]:  # Show first 5
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            reimb = case['expected_output']
            print(f"  Days: {days}, Miles: {miles:.1f}, Receipts: ${receipts:.2f}, Reimbursement: ${reimb:.2f}")

def main():
    file_path = 'public_cases.json'
    data = load_data(file_path)
    
    print(f"Loaded {len(data)} test cases")
    
    # Analyze each component separately
    analyze_per_diem(data)
    analyze_mileage(data)
    analyze_receipts(data)
    look_for_special_cases(data)
    
    # Save analysis results to a file
    with open('analysis_results.txt', 'w') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        print(f"Analysis of {len(data)} test cases")
        
        analyze_per_diem(data)
        analyze_mileage(data)
        analyze_receipts(data)
        look_for_special_cases(data)
        
        sys.stdout = original_stdout
    
    print("\nAnalysis results saved to analysis_results.txt")

if __name__ == "__main__":
    main()
