import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """Load and explore the transaction data"""
    
    # Load the data
    df = pd.read_csv('../data/transactions.csv')
    
    print("=== BASIC DATA INFO ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Total transactions: {len(df)}")
    print(f"Features: {list(df.columns)}")
    
    print("\n=== FRAUD DISTRIBUTION ===")
    fraud_counts = df['is_fraud'].value_counts()
    print(fraud_counts)
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())
    
    return df

def analyze_fraud_patterns(df):
    """Analyze patterns in fraudulent transactions"""
    
    print("\n=== FRAUD ANALYSIS ===")
    
    # Amount analysis
    print("\nAmount Analysis:")
    print("Legitimate transactions - Amount stats:")
    print(df[df['is_fraud'] == 0]['amount'].describe())
    print("\nFraudulent transactions - Amount stats:")
    print(df[df['is_fraud'] == 1]['amount'].describe())
    
    # Category analysis
    print("\nFraud by Category:")
    category_fraud = df.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).round(3)
    category_fraud.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
    print(category_fraud.sort_values('fraud_rate', ascending=False))
    
    # Time analysis
    print("\nFraud by Hour:")
    hour_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).round(3)
    hour_fraud.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
    print(hour_fraud.sort_values('fraud_rate', ascending=False).head(10))
    
    # Location risk analysis
    print("\nFraud by Location Risk:")
    location_fraud = df.groupby('location_risk')['is_fraud'].agg(['count', 'sum', 'mean']).round(3)
    location_fraud.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
    print(location_fraud.sort_values('fraud_rate', ascending=False))

def create_visualizations(df):
    """Create visualizations to understand the data"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fraud Detection Data Analysis', fontsize=16)
    
    # 1. Fraud distribution
    fraud_counts = df['is_fraud'].value_counts()
    axes[0,0].pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%')
    axes[0,0].set_title('Overall Fraud Distribution')
    
    # 2. Amount distribution by fraud
    axes[0,1].hist([df[df['is_fraud']==0]['amount'], df[df['is_fraud']==1]['amount']], 
                   bins=50, alpha=0.7, label=['Legitimate', 'Fraud'])
    axes[0,1].set_xlabel('Amount')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Amount Distribution')
    axes[0,1].legend()
    axes[0,1].set_xlim(0, 2000)  # Focus on smaller amounts
    
    # 3. Fraud by category
    category_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    axes[0,2].bar(range(len(category_fraud)), category_fraud.values)
    axes[0,2].set_xticks(range(len(category_fraud)))
    axes[0,2].set_xticklabels(category_fraud.index, rotation=45)
    axes[0,2].set_title('Fraud Rate by Category')
    axes[0,2].set_ylabel('Fraud Rate')
    
    # 4. Fraud by hour
    hour_fraud = df.groupby('hour')['is_fraud'].mean()
    axes[1,0].plot(hour_fraud.index, hour_fraud.values, marker='o')
    axes[1,0].set_xlabel('Hour of Day')
    axes[1,0].set_ylabel('Fraud Rate')
    axes[1,0].set_title('Fraud Rate by Hour')
    axes[1,0].grid(True)
    
    # 5. Account age vs fraud
    axes[1,1].boxplot([df[df['is_fraud']==0]['account_age_days'], 
                       df[df['is_fraud']==1]['account_age_days']])
    axes[1,1].set_xticklabels(['Legitimate', 'Fraud'])
    axes[1,1].set_ylabel('Account Age (days)')
    axes[1,1].set_title('Account Age Distribution')
    
    # 6. Location risk
    location_fraud = df.groupby('location_risk')['is_fraud'].mean()
    axes[1,2].bar(location_fraud.index, location_fraud.values)
    axes[1,2].set_title('Fraud Rate by Location Risk')
    axes[1,2].set_ylabel('Fraud Rate')
    
    plt.tight_layout()
    plt.savefig('../data/fraud_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as '../data/fraud_analysis.png'")

if __name__ == "__main__":
    print("Starting data exploration...")
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze fraud patterns
    analyze_fraud_patterns(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\nData exploration complete!")
    print("\nKey insights:")
    print("1. Check which categories have highest fraud rates")
    print("2. Look at amount patterns between fraud/legitimate")
    print("3. Notice time patterns in fraudulent transactions")
    print("4. See how location risk affects fraud rates")