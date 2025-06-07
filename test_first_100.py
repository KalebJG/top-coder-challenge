#!/usr/bin/env python3
import json
import subprocess
import sys

def load_test_cases(limit=100):
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    return test_cases[:limit]

def run_test_case(case):
    input_data = case['input']
    expected = case['expected_output']
    
    # Run the script with the test case
    cmd = [
        './run.sh',
        str(input_data['trip_duration_days']),
        str(input_data['miles_traveled']),
        str(input_data['total_receipts_amount'])
    ]
    
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True).strip()
        actual = float(result)
        error = abs(actual - expected)
        return {
            'input': input_data,
            'expected': expected,
            'actual': actual,
            'error': error,
            'success': abs(actual - expected) < 0.01
        }
    except (subprocess.CalledProcessError, ValueError) as e:
        return {
            'input': input_data,
            'expected': expected,
            'error': str(e),
            'success': False
        }

def main():
    test_cases = load_test_cases(100)
    results = []
    
    print(f"Testing first {len(test_cases)} cases...\n")
    
    for i, case in enumerate(test_cases, 1):
        result = run_test_case(case)
        results.append(result)
        
        if i % 10 == 0 or i == len(test_cases):
            print(f"Processed {i}/{len(test_cases)} cases")
    
    # Calculate statistics
    successful = sum(1 for r in results if r.get('success', False))
    errors = [r['error'] for r in results if 'error' in r and not isinstance(r['error'], str)]
    
    if errors:
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
    else:
        avg_error = 0
        max_error = 0
    
    print("\n=== Test Results ===")
    print(f"Total test cases: {len(results)}")
    print(f"Successful runs: {successful} ({successful/len(results)*100:.1f}%)")
    if errors:
        print(f"Average error: ${avg_error:.2f}")
        print(f"Maximum error: ${max_error:.2f}")
    
    # Show top 5 errors
    if errors:
        print("\nTop 5 errors:")
        sorted_results = sorted(
            [r for r in results if 'error' in r and not isinstance(r['error'], str)],
            key=lambda x: x['error'],
            reverse=True
        )[:5]
        
        for i, r in enumerate(sorted_results, 1):
            print(f"{i}. Input: {r['input']}")
            print(f"   Expected: ${r['expected']:.2f}, Got: ${r['actual']:.2f}, Error: ${r['error']:.2f}")

if __name__ == "__main__":
    main()
