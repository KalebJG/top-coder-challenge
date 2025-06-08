#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

def load_test_cases(limit=None):
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    return test_cases[:limit] if limit else test_cases

def run_test_case(case):
    input_data = case['input']
    expected = case['expected_output']
    
    cmd = [
        'python3', 'calculate_reimbursement.py',
        str(input_data['trip_duration_days']),
        str(input_data['miles_traveled']),
        str(input_data['total_receipts_amount'])
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        actual = float(result.stdout.strip())
        error = abs(actual - expected)
        return {
            'input': input_data,
            'expected': expected,
            'actual': actual,
            'error': error,
            'success': math.isclose(actual, expected, abs_tol=0.01)
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running test case: {e}")
        return None

def main():
    test_cases = load_test_cases()
    total_cases = len(test_cases)
    successful = 0
    errors = []
    
    print(f"Testing all {total_cases} cases...\n")
    
    for i, case in enumerate(test_cases, 1):
        if i % 100 == 0 or i == 1 or i == total_cases:
            print(f"Processed {i}/{total_cases} cases")
        
        result = run_test_case(case)
        if result:
            if result['success']:
                successful += 1
            else:
                errors.append({
                    'input': result['input'],
                    'expected': result['expected'],
                    'actual': result['actual'],
                    'error': result['error']
                })
    
    # Sort errors by magnitude
    errors.sort(key=lambda x: x['error'], reverse=True)
    
    print("\n=== Test Results ===")
    print(f"Total test cases: {total_cases}")
    print(f"Successful runs: {successful} ({successful/total_cases*100:.1f}%)")
    
    if errors:
        avg_error = sum(e['error'] for e in errors) / len(errors)
        print(f"Average error: ${avg_error:.2f}")
        print(f"Maximum error: ${errors[0]['error']:.2f}")
        
        print("\nTop 5 errors:")
        for i, error in enumerate(errors[:5], 1):
            print(f"{i}. Input: {error['input']}")
            print(f"   Expected: ${error['expected']:.2f}, Got: ${error['actual']:.2f}, "
                  f"Error: ${error['error']:.2f}")
    
    # Save detailed errors to file
    with open('test_errors.json', 'w') as f:
        json.dump({
            'total_cases': total_cases,
            'successful': successful,
            'errors': errors
        }, f, indent=2)

if __name__ == "__main__":
    import math
    main()
