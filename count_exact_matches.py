import pandas as pd

df = pd.read_csv('prediction_results.csv')
# Use round to ensure floating point comparison to 2 decimals
exact_matches = (df['predicted'].round(2) == df['expected'].round(2)).sum()
total_cases = len(df)
print(f"Number of exactly correct predictions: {exact_matches} out of {total_cases}")
