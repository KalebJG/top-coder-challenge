#!/bin/bash

# Black Box Challenge - Reimbursement Calculator
# This script calculates the reimbursement amount based on trip details
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Use Python implementation
python3 calculate_reimbursement.py "$1" "$2" "$3"
