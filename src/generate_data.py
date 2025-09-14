import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_transaction_data(num_transactions=10000):
    """Generate realistic transaction data for fraud detection"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic data ranges
    merchants = ['Amazon', 'Walmart', 'Starbucks', 'Shell Gas', 'McDonalds', 
                'Target', 'Best Buy', 'Grocery Store', 'ATM Withdrawal', 
                'Online Casino', 'Crypto Exchange', 'Unknown Merchant']
    
    categories = ['retail', 'food', 'gas', 'entertainment', 'gambling', 
                 'crypto', 'grocery', 'electronics', 'cash', 'other']
    
    # Generate base data
    data = []
    
    for i in range(num_transactions):
        # Random transaction details
        merchant = random.choice(merchants)
        category = random.choice(categories)
        
        # Amount based on category (some categories have higher amounts)
        if category in ['gambling', 'crypto']:
            amount = np.random.exponential(500) + 10
        elif category == 'cash':
            amount = random.choice([20, 40, 60, 80, 100, 200])
        else:
            amount = np.random.exponential(50) + 1
        
        # Time features
        days_ago = random.randint(0, 30)
        base_time = datetime.now() - timedelta(days=days_ago)
        hour = random.randint(0, 23)
        transaction_time = base_time.replace(hour=hour, minute=random.randint(0, 59))
        
        # User features
        user_id = f"user_{random.randint(1000, 9999)}"
        account_age_days = random.randint(1, 1000)
        
        # Location risk (simplified)
        location_risk = random.choice(['low', 'medium', 'high'])
        
        transaction = {
            'transaction_id': f'txn_{i:06d}',
            'user_id': user_id,
            'amount': round(amount, 2),
            'merchant': merchant,
            'category': category,
            'timestamp': transaction_time,
            'hour': hour,
            'day_of_week': transaction_time.weekday(),
            'account_age_days': account_age_days,
            'location_risk': location_risk
        }
        
        data.append(transaction)
    
    return pd.DataFrame(data)

def add_fraud_labels(df):
    """Add fraud labels based on realistic patterns"""
    
    fraud_indicators = []
    
    for _, row in df.iterrows():
        fraud_score = 0
        
        # High amount transactions
        if row['amount'] > 1000:
            fraud_score += 0.3
        if row['amount'] > 5000:
            fraud_score += 0.4
            
        # Suspicious categories
        if row['category'] in ['gambling', 'crypto']:
            fraud_score += 0.3
            
        # Unusual times
        if row['hour'] < 6 or row['hour'] > 23:
            fraud_score += 0.2
            
        # New accounts
        if row['account_age_days'] < 30:
            fraud_score += 0.2
            
        # High risk locations
        if row['location_risk'] == 'high':
            fraud_score += 0.4
            
        # Add some randomness
        fraud_score += random.uniform(-0.1, 0.1)
        
        # Determine if fraudulent (threshold at 0.5)
        is_fraud = 1 if fraud_score > 0.5 else 0
        fraud_indicators.append(is_fraud)
    
    df['is_fraud'] = fraud_indicators
    return df

if __name__ == "__main__":
    print("Generating transaction data...")
    
    # Generate data
    transactions = generate_transaction_data(10000)
    
    # Add fraud labels
    transactions = add_fraud_labels(transactions)
    
    # Save to CSV
    transactions.to_csv('../data/transactions.csv', index=False)
    
    # Print summary
    print(f"\nGenerated {len(transactions)} transactions")
    print(f"Fraud transactions: {transactions['is_fraud'].sum()}")
    print(f"Legitimate transactions: {(transactions['is_fraud'] == 0).sum()}")
    print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")
    
    print("\nFirst 5 rows:")
    print(transactions.head())
    
    print(f"\nData saved to '../data/transactions.csv'")