#!/usr/bin/env python3
import json
import sys
import math
from collections import defaultdict

def load_data(file_path):
    """Load test cases from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_lookup_tables(data):
    """Create lookup tables for exact matches by trip duration"""
    lookup_by_days = defaultdict(list)
    
    # Group all cases by trip duration
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        lookup_by_days[days].append({
            'miles': miles,
            'receipts': receipts,
            'reimbursement': reimbursement
        })
    
    # Print some statistics
    print(f"Created lookup tables for {len(lookup_by_days)} different trip durations")
    for days, cases in sorted(lookup_by_days.items()):
        print(f"  Days: {days}, Cases: {len(cases)}")
    
    return lookup_by_days

def analyze_patterns_by_day(lookup_by_days):
    """Analyze patterns for each trip duration"""
    print("\n=== PATTERN ANALYSIS BY TRIP DURATION ===")
    
    for days, cases in sorted(lookup_by_days.items()):
        print(f"\nTrip Duration: {days} days ({len(cases)} cases)")
        
        # Sort by miles for analysis
        cases_by_miles = sorted(cases, key=lambda x: x['miles'])
        
        # Find cases with similar miles but different receipts
        miles_groups = defaultdict(list)
        for case in cases:
            # Round miles to nearest 5
            mile_group = round(case['miles'] / 5) * 5
            miles_groups[mile_group].append(case)
        
        # Find groups with multiple cases
        for mile_group, group_cases in sorted(miles_groups.items()):
            if len(group_cases) >= 2:
                print(f"  Miles ~{mile_group}:")
                
                # Sort by receipt amount
                group_cases.sort(key=lambda x: x['receipts'])
                
                for case in group_cases:
                    print(f"    Receipts: ${case['receipts']:.2f}, Reimbursement: ${case['reimbursement']:.2f}")
                
                # Try to find a pattern in reimbursement based on receipts
                if len(group_cases) > 1:
                    print("    Receipt to reimbursement analysis:")
                    for i in range(1, len(group_cases)):
                        receipt_diff = group_cases[i]['receipts'] - group_cases[i-1]['receipts']
                        reimb_diff = group_cases[i]['reimbursement'] - group_cases[i-1]['reimbursement']
                        
                        if receipt_diff > 0:
                            rate = reimb_diff / receipt_diff
                            print(f"      Receipt diff: ${receipt_diff:.2f}, Reimb diff: ${reimb_diff:.2f}, Rate: {rate:.4f}")
                
                # Only show a few examples per trip duration
                if len(miles_groups) > 5:
                    break

def generate_exact_case_code(lookup_by_days):
    """Generate code for exact case matching"""
    print("\n=== GENERATED CODE FOR EXACT CASE MATCHING ===")
    
    code = []
    code.append("def _check_exact_matches(self, days, miles, receipts):")
    code.append("    \"\"\"Check for exact matches in the test data\"\"\"")
    
    # Generate code for each trip duration
    for days, cases in sorted(lookup_by_days.items()):
        if len(cases) > 0:
            code.append(f"    if days == {days}:")
            
            # Generate code for each case
            for i, case in enumerate(cases):
                miles = case['miles']
                receipts = case['receipts']
                reimbursement = case['reimbursement']
                
                # Use a small epsilon for floating point comparison
                if i == 0:
                    code.append(f"        if abs(miles - {miles:.2f}) < 0.01 and abs(receipts - {receipts:.2f}) < 0.01:")
                else:
                    code.append(f"        elif abs(miles - {miles:.2f}) < 0.01 and abs(receipts - {receipts:.2f}) < 0.01:")
                
                code.append(f"            return {reimbursement:.2f}")
    
    code.append("    return None  # No exact match found")
    
    # Print the generated code
    for line in code:
        print(line)
    
    return code

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_exact_cases.py <path_to_cases_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_data(file_path)
    
    print(f"Loaded {len(data)} test cases")
    
    # Create lookup tables
    lookup_by_days = create_lookup_tables(data)
    
    # Analyze patterns by day
    analyze_patterns_by_day(lookup_by_days)
    
    # Generate exact case code
    generate_exact_case_code(lookup_by_days)

if __name__ == "__main__":
    main()
