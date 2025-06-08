#!/usr/bin/env python3
import json
import sys
from calculate_reimbursement import calculate_reimbursement

def test_against_cases(cases_file, limit=100):
    """
    Test the reimbursement calculator against provided test cases
    
    Args:
        cases_file (str): Path to the JSON file with test cases
        limit (int): Maximum number of cases to test
    """
    with open(cases_file, 'r') as f:
        cases = json.load(f)
    
    # Limit to the first 'limit' cases
    cases = cases[:limit]
    total_cases = len(cases)
    passed_cases = 0
    failed_cases = []
    
    print(f"Testing {total_cases} cases...")
    
    for i, case in enumerate(cases):
        input_data = case['input']
        expected = case['expected_output']
        
        trip_days = input_data['trip_duration_days']
        miles = input_data['miles_traveled']
        receipts = input_data['total_receipts_amount']
        
        actual = calculate_reimbursement(trip_days, miles, receipts)
        
        # Compare with a small epsilon to account for floating point precision
        if abs(actual - expected) < 0.01:
            passed_cases += 1
        else:
            failed_cases.append({
                'case_index': i,
                'input': input_data,
                'expected': expected,
                'actual': actual,
                'difference': actual - expected
            })
    
    print(f"Passed: {passed_cases}/{total_cases} ({passed_cases/total_cases*100:.2f}%)")
    
    if failed_cases:
        print("\nFailed cases:")
        for case in failed_cases[:10]:  # Show first 10 failed cases
            print(f"Case {case['case_index']}:")
            print(f"  Input: {case['input']}")
            print(f"  Expected: {case['expected']}")
            print(f"  Actual: {case['actual']}")
            print(f"  Difference: {case['difference']:.2f}")
        
        if len(failed_cases) > 10:
            print(f"... and {len(failed_cases) - 10} more failed cases")
    
    return passed_cases, total_cases, failed_cases

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_reimbursement.py <path_to_cases_file>")
        sys.exit(1)
    
    cases_file = sys.argv[1]
    passed, total, failed = test_against_cases(cases_file)
    
    # Exit with non-zero status if any tests failed
    if passed != total:
        sys.exit(1)
