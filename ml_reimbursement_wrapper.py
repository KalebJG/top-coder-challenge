#!/usr/bin/env python3
"""
ML-based Reimbursement Calculator Wrapper

This module provides a wrapper around the ML model that matches the interface
of the original reimbursement calculator.
"""
import os
import sys
from ml_reimbursement_engine import MLReimbursementEngine

class MLReimbursementCalculator:
    """Wrapper class for ML-based reimbursement calculation."""
    
    def __init__(self, model_path='reimbursement_model.joblib'):
        """Initialize the calculator with the trained ML model.
        
        Args:
            model_path (str): Path to the trained model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                "Please train the model first by running ml_reimbursement_engine.py"
            )
        
        self.engine = MLReimbursementEngine()
        self.engine.load_model(model_path)
    
    def calculate_reimbursement(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """Calculate reimbursement using the ML model.
        
        Args:
            trip_duration_days (int): Number of days for the trip
            miles_traveled (float): Miles traveled
            total_receipts_amount (float): Total receipts amount
            
        Returns:
            float: Rounded reimbursement amount with 2 decimal places
        """
        try:
            # Get prediction from the ML model
            amount = self.engine.predict(
                trip_duration_days,
                miles_traveled,
                total_receipts_amount
            )
            
            # Round to 2 decimal places (currency standard)
            return round(amount, 2)
            
        except Exception as e:
            print(f"Error calculating reimbursement: {e}", file=sys.stderr)
            raise

def main():
    """Main function to run the calculator from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate travel reimbursement using ML model')
    parser.add_argument('trip_duration_days', type=int, help='Number of days for the trip')
    parser.add_argument('miles_traveled', type=float, help='Miles traveled')
    parser.add_argument('total_receipts_amount', type=float, help='Total receipts amount')
    
    args = parser.parse_args()
    
    try:
        calculator = MLReimbursementCalculator()
        result = calculator.calculate_reimbursement(
            args.trip_duration_days,
            args.miles_traveled,
            args.total_receipts_amount
        )
        print(f"{result:.2f}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
