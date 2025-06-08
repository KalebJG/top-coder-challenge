#!/usr/bin/env python3
"""
Test the accuracy of the ML reimbursement calculator against test cases.
"""
import json
import sys
from calculate_reimbursement import calculate_reimbursement

def load_test_cases():
    """Load test cases from public_cases.json."""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def test_ml_accuracy(use_ml=True):
    """Test the ML model's accuracy on the test cases."""
    test_cases = load_test_cases()
    total_cases = len(test_cases)
    correct = 0
    errors = []
    
    print(f"Testing {total_cases} cases with {'ML' if use_ml else 'legacy'} implementation...")
    
    for i, case in enumerate(test_cases, 1):
        if i % 100 == 0 or i == 1 or i == total_cases:
            print(f"Processed {i}/{total_cases} cases")
        
        input_data = case['input']
        expected = case['expected_output']
        
        try:
            # Get prediction
            predicted = calculate_reimbursement(
                input_data['trip_duration_days'],
                input_data['miles_traveled'],
                input_data['total_receipts_amount'],
                use_ml=use_ml
            )
            
            # Check if prediction matches expected (within 1 cent tolerance for floating point)
            if abs(predicted - expected) < 0.01:
                correct += 1
            else:
                errors.append({
                    'input': input_data,
                    'expected': expected,
                    'predicted': predicted,
                    'error': abs(predicted - expected)
                })
                
        except Exception as e:
            print(f"Error processing case {i}: {e}", file=sys.stderr)
    
    # Sort errors by magnitude
    errors.sort(key=lambda x: x['error'], reverse=True)
    
    # Print results
    accuracy = (correct / total_cases) * 100
    print(f"\n=== Test Results ===")
    print(f"Total test cases: {total_cases}")
    print(f"Correct predictions: {correct} ({accuracy:.2f}%)")
    
    if errors:
        avg_error = sum(e['error'] for e in errors) / len(errors)
        print(f"Average error: ${avg_error:.2f}")
        print(f"Maximum error: ${errors[0]['error']:.2f}")
        
        print("\nTop 5 errors:")
        for i, error in enumerate(errors[:5], 1):
            print(f"{i}. Input: {error['input']}")
            print(f"   Expected: ${error['expected']:.2f}, Predicted: ${error['predicted']:.2f}, "
                  f"Error: ${error['error']:.2f}")
    
    return accuracy, errors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ML reimbursement calculator accuracy')
    parser.add_argument('--legacy', action='store_true', help='Test legacy implementation instead of ML')
    args = parser.parse_args()
    
    test_ml_accuracy(use_ml=not args.legacy)
