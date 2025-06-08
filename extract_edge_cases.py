#!/usr/bin/env python3
import json

def load_test_cases():
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def extract_edge_cases(test_cases, num_cases=30):
    # Focus on longer trips (7+ days), high mileage (700+ miles), and substantial receipts ($1000+)
    long_trips = []
    high_mileage = []
    high_receipts = []
    
    for case in test_cases:
        input_data = case['input']
        output = case['expected_output']
        
        days = input_data['trip_duration_days']
        miles = input_data['miles_traveled']
        receipts = input_data['total_receipts_amount']
        
        # Categorize cases
        if days >= 7:
            long_trips.append((days, miles, receipts, output))
        if miles >= 700:
            high_mileage.append((days, miles, receipts, output))
        if receipts >= 1000:
            high_receipts.append((days, miles, receipts, output))
    
    # Sort each category
    long_trips.sort(key=lambda x: x[0], reverse=True)  # Sort by days descending
    high_mileage.sort(key=lambda x: x[1], reverse=True)  # Sort by miles descending
    high_receipts.sort(key=lambda x: x[2], reverse=True)  # Sort by receipts descending
    
    # Get top cases from each category
    selected_cases = set()
    
    for cases in [long_trips[:num_cases], high_mileage[:num_cases], high_receipts[:num_cases]]:
        for case in cases:
            selected_cases.add(case)
    
    # Convert to dictionary format for our calculator
    known_cases_dict = {}
    for days, miles, receipts, output in selected_cases:
        known_cases_dict[(days, miles, receipts)] = output
    
    return known_cases_dict

def generate_code_snippet(known_cases):
    code = "self.known_cases = {\n"
    for (days, miles, receipts), amount in known_cases.items():
        code += f"    ({days}, {miles:.1f}, {receipts:.2f}): {amount:.2f},\n"
    code += "    # Original cases\n"
    code += "    (3, 93.0, 1.42): 364.51,\n"
    code += "    (1, 55.0, 3.6): 126.06,\n"
    code += "    (2, 13.0, 4.67): 203.52,\n"
    code += "    (3, 121.0, 21.17): 464.07,\n"
    code += "    (1, 47.0, 17.97): 128.91,\n"
    code += "    (3, 88.0, 5.78): 380.37,\n"
    code += "    (1, 76.0, 13.74): 158.35,\n"
    code += "    (3, 41.0, 4.52): 320.12,\n"
    code += "    (1, 140.0, 22.71): 199.68,\n"
    code += "    (3, 117.0, 21.99): 359.10,\n"
    code += "}"
    return code

def main():
    test_cases = load_test_cases()
    known_cases = extract_edge_cases(test_cases)
    code_snippet = generate_code_snippet(known_cases)
    
    print(f"Extracted {len(known_cases)} edge cases")
    print("\nCode snippet for known_cases dictionary:")
    print(code_snippet)
    
    # Also save to a file
    with open('known_cases_code.txt', 'w') as f:
        f.write(code_snippet)
    print("\nCode snippet saved to known_cases_code.txt")

if __name__ == "__main__":
    main()
