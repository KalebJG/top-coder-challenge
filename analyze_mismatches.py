import pandas as pd
import matplotlib.pyplot as plt

# Load mismatches
mismatches = pd.read_csv('mismatches.csv')

# Summary tables
summary = {}
summary['by_days'] = mismatches['days'].value_counts().sort_index()
summary['by_miles_bin'] = pd.cut(mismatches['miles'], bins=[0,50,100,200,400,600,1000,2000,5000], include_lowest=True).value_counts().sort_index()
summary['by_receipts_bin'] = pd.cut(mismatches['receipts'], bins=[0,100,300,600,1000,1500,2000,5000], include_lowest=True).value_counts().sort_index()
summary['by_sign'] = mismatches['difference'].apply(lambda x: 'over' if x < 0 else 'under').value_counts()

# Print summary tables
print('--- Mismatches by Trip Duration (days) ---')
print(summary['by_days'])
print('\n--- Mismatches by Miles (binned) ---')
print(summary['by_miles_bin'])
print('\n--- Mismatches by Receipts (binned) ---')
print(summary['by_receipts_bin'])
print('\n--- Over vs Under Predictions ---')
print(summary['by_sign'])

# Visualizations
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
summary['by_days'].plot(kind='bar', title='Mismatches by Trip Duration (days)')
plt.xlabel('Days')
plt.ylabel('Count')

plt.subplot(1,3,2)
summary['by_miles_bin'].plot(kind='bar', title='Mismatches by Miles (bin)')
plt.xlabel('Miles Bin')
plt.ylabel('Count')

plt.subplot(1,3,3)
summary['by_receipts_bin'].plot(kind='bar', title='Mismatches by Receipts (bin)')
plt.xlabel('Receipts Bin')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('mismatch_patterns.png')
print('\nSaved bar chart summary as mismatch_patterns.png')

# Error histogram
plt.figure(figsize=(6,4))
mismatches['difference'].hist(bins=30)
plt.title('Distribution of Prediction Errors (Mismatches)')
plt.xlabel('Prediction Error ($)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('mismatch_error_histogram.png')
print('Saved error histogram as mismatch_error_histogram.png')

# Written pattern analysis
print('\n--- Pattern Analysis ---')
# Top 5 most frequent days
top_days = summary['by_days'].head(5)
print(f'Top trip durations with mismatches: {list(top_days.index)}')
# Top 5 most frequent miles bins
top_miles = summary['by_miles_bin'].head(5)
print(f'Top mileage bins with mismatches: {list(top_miles.index)}')
# Top 5 most frequent receipts bins
top_receipts = summary['by_receipts_bin'].head(5)
print(f'Top receipts bins with mismatches: {list(top_receipts.index)}')
# Over/under summary
print(f"Overpredictions: {summary['by_sign'].get('over',0)}, Underpredictions: {summary['by_sign'].get('under',0)}")

# Show a few sample mismatches
print('\nSample mismatches:')
print(mismatches.head(10)[['case','days','miles','receipts','expected','predicted','difference']])
