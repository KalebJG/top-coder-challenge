#!/usr/bin/env python3
"""
Test the accuracy of the ML Reimbursement Engine against test cases.
"""
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml_reimbursement_engine import EnhancedMLReimbursementEngine

def load_test_cases(filepath='public_cases.json'):
    """Load test cases from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_predictions(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mean_absolute_percentage_error': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'accuracy_within_10_percent': np.mean(np.abs((y_true - y_pred) / y_true) <= 0.1) * 100,
        'accuracy_within_20_percent': np.mean(np.abs((y_true - y_pred) / y_true) <= 0.2) * 100,
    }
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics in a readable format."""
    print("\n=== Model Evaluation Metrics ===")
    print(f"Mean Absolute Error (MAE): ${metrics['mae']:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${metrics['rmse']:.2f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Mean Absolute Percentage Error: {metrics['mean_absolute_percentage_error']:.2f}%")
    print(f"Accuracy within 10%: {metrics['accuracy_within_10_percent']:.2f}%")
    print(f"Accuracy within 20%: {metrics['accuracy_within_20_percent']:.2f}%")

def main():
    engine = EnhancedMLReimbursementEngine(model_type='dt')
    model_file = 'reimbursement_model.joblib'
    if os.path.exists(model_file):
        engine.load_model(model_file)
    
    # Load test cases
    print("Loading test cases...")
    test_cases = load_test_cases()
    
    # Prepare data for evaluation
    y_true = []
    y_pred = []
    results = []
    
    print("\nRunning predictions on test cases...")
    for i, case in enumerate(test_cases, 1):
        input_data = case['input']
        expected = case['expected_output']
        
        # Make prediction
        try:
            predicted = engine.predict(
                trip_duration_days=input_data['trip_duration_days'],
                miles_traveled=input_data['miles_traveled'],
                total_receipts_amount=input_data['total_receipts_amount']
            )
            
            # Store results
            y_true.append(expected)
            y_pred.append(predicted)
            results.append({
                'case': i,
                'days': input_data['trip_duration_days'],
                'miles': input_data['miles_traveled'],
                'receipts': input_data['total_receipts_amount'],
                'expected': expected,
                'predicted': predicted,
                'difference': predicted - expected,
                'percent_error': ((predicted - expected) / expected) * 100 if expected != 0 else float('inf')
            })
            
        except Exception as e:
            print(f"Error processing case {i}: {e}")
    
    # Convert to numpy arrays for calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate and print metrics
    metrics = evaluate_predictions(y_true, y_pred)
    print_metrics(metrics)
    
    # Create and save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('prediction_results.csv', index=False)
    print("\nDetailed results saved to 'prediction_results.csv'")
    
    # Print some examples
    print("\n=== Example Predictions ===")
    print(results_df.head().to_string(index=False))
    
    # Print worst predictions
    print("\n=== Worst Underpredictions ===")
    print(results_df.nlargest(5, 'difference').to_string(index=False))
    
    print("\n=== Worst Overpredictions ===")
    print(results_df.nsmallest(5, 'difference').to_string(index=False))

if __name__ == "__main__":
    main()
