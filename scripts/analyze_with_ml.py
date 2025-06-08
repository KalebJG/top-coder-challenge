#!/usr/bin/env python3
import json
import math
import sys
from collections import defaultdict

def load_data(file_path):
    """Load test cases from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_specific_cases(data):
    """Analyze specific cases to identify patterns"""
    print("\n=== SPECIFIC CASE ANALYSIS ===")
    
    # Analyze cases with same trip days but different miles/receipts
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
    
    # For each trip duration, find cases with similar miles but different receipts
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        if len(cases) < 5:
            continue
        
        # Group by miles (rounded to nearest 10)
        by_miles = defaultdict(list)
        for case in cases:
            mile_bin = round(case['miles'] / 10) * 10
            by_miles[mile_bin].append(case)
        
        # Find a mile bin with multiple cases
        for mile_bin, mile_cases in by_miles.items():
            if len(mile_cases) >= 2:
                print(f"\nTrip duration: {days} days, Miles ~{mile_bin}")
                
                # Sort by receipt amount
                mile_cases.sort(key=lambda x: x['receipts'])
                
                for case in mile_cases:
                    print(f"  Receipts: ${case['receipts']:.2f}, Reimbursement: ${case['reimbursement']:.2f}")
                
                # Calculate differences
                if len(mile_cases) > 1:
                    for i in range(1, len(mile_cases)):
                        receipt_diff = mile_cases[i]['receipts'] - mile_cases[i-1]['receipts']
                        reimb_diff = mile_cases[i]['reimbursement'] - mile_cases[i-1]['reimbursement']
                        if receipt_diff > 0:
                            rate = reimb_diff / receipt_diff
                            print(f"  Receipt diff: ${receipt_diff:.2f}, Reimb diff: ${reimb_diff:.2f}, Rate: {rate:.4f}")
                
                # Only show a few examples per trip duration
                break

def analyze_base_rates(data):
    """Try to determine the base rate for each trip duration"""
    print("\n=== BASE RATE ANALYSIS ===")
    
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
    
    # For each trip duration, estimate the base rate
    base_rates = {}
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        
        # Find cases with minimal miles and receipts
        min_miles_case = min(cases, key=lambda x: x['miles'])
        min_receipts_case = min(cases, key=lambda x: x['receipts'])
        
        # Use the case with both minimal miles and minimal receipts if possible
        base_cases = [c for c in cases if c['miles'] < 20 and c['receipts'] < 5]
        if base_cases:
            base_case = min(base_cases, key=lambda x: x['reimbursement'])
        else:
            # Otherwise use the case with minimal reimbursement
            base_case = min(cases, key=lambda x: x['reimbursement'])
        
        base_rate = base_case['reimbursement']
        base_rates[days] = base_rate
        
        print(f"Days: {days}, Base rate: ${base_rate:.2f}, Per day: ${base_rate/days:.2f}")
        print(f"  From case: Miles={base_case['miles']:.2f}, Receipts=${base_case['receipts']:.2f}")
    
    return base_rates

def analyze_mileage_rates(data, base_rates):
    """Analyze mileage rates after subtracting base rates"""
    print("\n=== MILEAGE RATE ANALYSIS ===")
    
    # Group by trip days
    by_days = defaultdict(list)
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        # Skip if we don't have a base rate for this trip duration
        if days not in base_rates:
            continue
        
        # Calculate the non-base component
        non_base = reimbursement - base_rates[days]
        
        by_days[days].append({
            'miles': miles,
            'receipts': receipts,
            'reimbursement': reimbursement,
            'non_base': non_base
        })
    
    # For each trip duration, analyze mileage rates
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        if len(cases) < 5:
            continue
        
        # Find cases with minimal receipts but varying miles
        low_receipt_cases = [c for c in cases if c['receipts'] < 5]
        if len(low_receipt_cases) >= 2:
            print(f"\nTrip duration: {days} days, Low receipts cases:")
            
            # Sort by miles
            low_receipt_cases.sort(key=lambda x: x['miles'])
            
            for case in low_receipt_cases[:5]:  # Show first 5
                print(f"  Miles: {case['miles']:.2f}, Non-base: ${case['non_base']:.2f}, Rate: ${case['non_base']/case['miles']:.4f}/mile")
            
            # Calculate differences between consecutive cases
            if len(low_receipt_cases) > 1:
                print("\n  Mile rate analysis:")
                
                # Group by mile ranges
                mile_ranges = [(0, 100), (100, 500), (500, float('inf'))]
                for start, end in mile_ranges:
                    range_cases = [c for c in low_receipt_cases if start <= c['miles'] < end]
                    if len(range_cases) >= 2:
                        range_cases.sort(key=lambda x: x['miles'])
                        
                        rates = []
                        for i in range(1, len(range_cases)):
                            mile_diff = range_cases[i]['miles'] - range_cases[i-1]['miles']
                            non_base_diff = range_cases[i]['non_base'] - range_cases[i-1]['non_base']
                            
                            if mile_diff > 0:
                                rate = non_base_diff / mile_diff
                                rates.append(rate)
                        
                        if rates:
                            avg_rate = sum(rates) / len(rates)
                            print(f"  Miles {start}-{end if end != float('inf') else '∞'}: Avg rate ${avg_rate:.4f}/mile")

def analyze_receipt_rates(data, base_rates):
    """Analyze receipt rates after subtracting base rates"""
    print("\n=== RECEIPT RATE ANALYSIS ===")
    
    # Group by trip days
    by_days = defaultdict(list)
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        # Skip if we don't have a base rate for this trip duration
        if days not in base_rates:
            continue
        
        # Calculate the non-base component
        non_base = reimbursement - base_rates[days]
        
        by_days[days].append({
            'miles': miles,
            'receipts': receipts,
            'reimbursement': reimbursement,
            'non_base': non_base
        })
    
    # For each trip duration, analyze receipt rates
    for days in sorted(by_days.keys()):
        cases = by_days[days]
        if len(cases) < 5:
            continue
        
        # Find cases with minimal miles but varying receipts
        low_mile_cases = [c for c in cases if c['miles'] < 20]
        if len(low_mile_cases) >= 2:
            print(f"\nTrip duration: {days} days, Low mileage cases:")
            
            # Sort by receipts
            low_mile_cases.sort(key=lambda x: x['receipts'])
            
            for case in low_mile_cases[:5]:  # Show first 5
                if case['receipts'] > 0:
                    print(f"  Receipts: ${case['receipts']:.2f}, Non-base: ${case['non_base']:.2f}, Rate: ${case['non_base']/case['receipts']:.4f}/dollar")
                else:
                    print(f"  Receipts: ${case['receipts']:.2f}, Non-base: ${case['non_base']:.2f}")
            
            # Calculate differences between consecutive cases
            if len(low_mile_cases) > 1:
                print("\n  Receipt rate analysis:")
                
                # Group by receipt ranges
                receipt_ranges = [(0, 100), (100, 400), (400, 1000), (1000, float('inf'))]
                for start, end in receipt_ranges:
                    range_cases = [c for c in low_mile_cases if start <= c['receipts'] < end]
                    if len(range_cases) >= 2:
                        range_cases.sort(key=lambda x: x['receipts'])
                        
                        rates = []
                        for i in range(1, len(range_cases)):
                            receipt_diff = range_cases[i]['receipts'] - range_cases[i-1]['receipts']
                            non_base_diff = range_cases[i]['non_base'] - range_cases[i-1]['non_base']
                            
                            if receipt_diff > 0:
                                rate = non_base_diff / receipt_diff
                                rates.append(rate)
                        
                        if rates:
                            avg_rate = sum(rates) / len(rates)
                            print(f"  Receipts ${start}-${end if end != float('inf') else '∞'}: Avg rate ${avg_rate:.4f}/dollar")

def analyze_special_cases(data):
    """Analyze special cases and quirks"""
    print("\n=== SPECIAL CASES ANALYSIS ===")
    
    # Check for special handling of receipts ending in .49 or .99
    special_49 = []
    special_99 = []
    
    for case in data:
        receipts = case['input']['total_receipts_amount']
        cents = int(round(receipts * 100)) % 100
        
        if cents == 49:
            special_49.append(case)
        elif cents == 99:
            special_99.append(case)
    
    if special_49:
        print("\nCases with receipts ending in .49:")
        for case in special_49[:5]:  # Show first 5
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            reimb = case['expected_output']
            print(f"  Days: {days}, Miles: {miles:.1f}, Receipts: ${receipts:.2f}, Reimbursement: ${reimb:.2f}")
    
    if special_99:
        print("\nCases with receipts ending in .99:")
        for case in special_99[:5]:  # Show first 5
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            reimb = case['expected_output']
            print(f"  Days: {days}, Miles: {miles:.1f}, Receipts: ${receipts:.2f}, Reimbursement: ${reimb:.2f}")
    
    # Check for special handling of 5-day trips
    five_day_trips = [case for case in data if case['input']['trip_duration_days'] == 5]
    if five_day_trips:
        print("\nFive-day trips analysis:")
        
        # Calculate average reimbursement per day
        total_reimb = sum(case['expected_output'] for case in five_day_trips)
        avg_per_day = total_reimb / (5 * len(five_day_trips))
        
        print(f"  Average reimbursement per day: ${avg_per_day:.2f}")
        
        # Compare with 4-day and 6-day trips
        four_day_trips = [case for case in data if case['input']['trip_duration_days'] == 4]
        six_day_trips = [case for case in data if case['input']['trip_duration_days'] == 6]
        
        if four_day_trips:
            total_4day = sum(case['expected_output'] for case in four_day_trips)
            avg_4day = total_4day / (4 * len(four_day_trips))
            print(f"  4-day trips avg per day: ${avg_4day:.2f}")
        
        if six_day_trips:
            total_6day = sum(case['expected_output'] for case in six_day_trips)
            avg_6day = total_6day / (6 * len(six_day_trips))
            print(f"  6-day trips avg per day: ${avg_6day:.2f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_with_ml.py <path_to_cases_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_data(file_path)
    
    print(f"Loaded {len(data)} test cases")
    
    # Analyze specific cases
    analyze_specific_cases(data)
    
    # Analyze base rates
    base_rates = analyze_base_rates(data)
    
    # Analyze mileage and receipt rates
    analyze_mileage_rates(data, base_rates)
    analyze_receipt_rates(data, base_rates)
    
    # Analyze special cases
    analyze_special_cases(data)

if __name__ == "__main__":
    main()
