import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml_reimbursement_engine import EnhancedMLReimbursementEngine

if len(sys.argv) != 4:
    print("Usage: python calculate_reimbursement_cli.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
    sys.exit(1)

trip_duration_days = int(sys.argv[1])
miles_traveled = float(sys.argv[2])
total_receipts_amount = float(sys.argv[3])

engine = EnhancedMLReimbursementEngine()
engine.load_model('models/reimbursement_model.joblib')

amount = engine.predict(trip_duration_days, miles_traveled, total_receipts_amount)
print(amount)
