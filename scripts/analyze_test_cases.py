#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_test_cases():
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def analyze_basic_stats(test_cases):
    # Convert to DataFrame for easier analysis
    data = []
    for case in test_cases:
        data.append({
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'output': case['expected_output']
        })
    
    df = pd.DataFrame(data)
    
    # Calculate basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Calculate correlation matrix
    print("\n=== Correlation Matrix ===")
    print(df.corr())
    
    # Calculate reimbursement per day
    df['per_day'] = df['output'] / df['days']
    
    # Group by days and calculate average reimbursement
    print("\n=== Average Reimbursement by Trip Duration ===")
    print(df.groupby('days')['output'].agg(['mean', 'median', 'count']))
    
    return df

def plot_relationships(df):
    # Create scatter plots for different relationships
    plt.figure(figsize=(15, 10))
    
    # Days vs Output
    plt.subplot(2, 2, 1)
    plt.scatter(df['days'], df['output'], alpha=0.5)
    plt.title('Trip Duration vs Reimbursement')
    plt.xlabel('Days')
    plt.ylabel('Reimbursement')
    
    # Miles vs Output
    plt.subplot(2, 2, 2)
    plt.scatter(df['miles'], df['output'], alpha=0.5)
    plt.title('Miles Traveled vs Reimbursement')
    plt.xlabel('Miles')
    plt.ylabel('Reimbursement')
    
    # Receipts vs Output
    plt.subplot(2, 2, 3)
    plt.scatter(df['receipts'], df['output'], alpha=0.5)
    plt.title('Receipts vs Reimbursement')
    plt.xlabel('Receipts ($)')
    plt.ylabel('Reimbursement')
    
    # Days vs Per Day Reimbursement
    plt.subplot(2, 2, 4)
    plt.scatter(df['days'], df['per_day'], alpha=0.5)
    plt.title('Trip Duration vs Per Day Reimbursement')
    plt.xlabel('Days')
    plt.ylabel('Reimbursement per Day')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png')
    print("\nSaved data analysis plots to data_analysis.png")

def cluster_analysis(df):
    # Prepare data for clustering
    X = df[['days', 'miles', 'receipts']].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Use elbow method to find optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure()
    plt.plot(range(1, 11), wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow_curve.png')
    
    # Based on the elbow curve, choose k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster characteristics
    print("\n=== Cluster Analysis ===")
    cluster_stats = df.groupby('cluster').agg({
        'days': ['mean', 'count'],
        'miles': 'mean',
        'receipts': 'mean',
        'output': 'mean'
    })
    print(cluster_stats)
    
    return df

def main():
    print("Analyzing test cases...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")
    
    df = analyze_basic_stats(test_cases)
    plot_relationships(df)
    df_with_clusters = cluster_analysis(df)
    
    # Save the clustered data for further analysis
    df_with_clusters.to_csv('clustered_test_cases.csv', index=False)
    print("\nSaved clustered data to clustered_test_cases.csv")

if __name__ == "__main__":
    main()
