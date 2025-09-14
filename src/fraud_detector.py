import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

class FraudDetector:
    """Real-time fraud detection system"""
    
    def __init__(self):
        """Initialize the fraud detector by loading trained models"""
        
        print("Loading fraud detection models...")
        
        # Load the trained model
        self.model = joblib.load('models/best_fraud_model.pkl')

        # Load preprocessing components
        self.scaler = joblib.load('models/scaler.pkl')
        self.merchant_encoder = joblib.load('models/merchant_encoder.pkl')
        self.category_encoder = joblib.load('models/category_encoder.pkl')
        self.location_encoder = joblib.load('models/location_encoder.pkl')

        # Load model metadata
        self.metadata = joblib.load('models/model_metadata.pkl')
        
        print(f"Loaded {self.metadata['model_type']} model")
        print(f"Model AUC Score: {self.metadata['auc_score']:.4f}")
        print("Fraud detector ready!")
    
    def preprocess_transaction(self, transaction):
        """Preprocess a single transaction for prediction"""
        
        # Convert to DataFrame for easier processing
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        else:
            df = transaction.copy()
        
        # Parse timestamp if it's a string
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Encode categorical variables
        try:
            df['merchant_encoded'] = self.merchant_encoder.transform(df['merchant'])
        except ValueError:
            # Handle unknown merchants
            df['merchant_encoded'] = -1
            
        try:
            df['category_encoded'] = self.category_encoder.transform(df['category'])
        except ValueError:
            # Handle unknown categories
            df['category_encoded'] = -1
            
        try:
            df['location_risk_encoded'] = self.location_encoder.transform(df['location_risk'])
        except ValueError:
            # Handle unknown location risks
            df['location_risk_encoded'] = 0  # Default to low risk
        
        # Create amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_high_amount'] = (df['amount'] > 1000).astype(int)  # Using a fixed threshold
        
        # Account age features
        df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
        
        # Select the same features used in training
        feature_columns = [
            'amount', 'amount_log', 'is_high_amount',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'account_age_days', 'is_new_account',
            'merchant_encoded', 'category_encoded', 'location_risk_encoded'
        ]
        
        return df[feature_columns]
    
    def predict_fraud(self, transaction):
        """Predict if a transaction is fraudulent"""
        
        # Preprocess the transaction
        features = self.preprocess_transaction(transaction)
        
        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        fraud_probability = self.model.predict_proba(features)[0][1]
        
        # Determine risk level
        if fraud_probability >= 0.8:
            risk_level = "HIGH"
        elif fraud_probability >= 0.5:
            risk_level = "MEDIUM"
        elif fraud_probability >= 0.3:
            risk_level = "LOW"
        else:
            risk_level = "VERY LOW"
        
        result = {
            'is_fraud': bool(prediction),
            'fraud_probability': float(fraud_probability),
            'risk_level': risk_level,
            'recommendation': 'BLOCK' if prediction else 'APPROVE'
        }
        
        return result
    
    def batch_predict(self, transactions):
        """Predict fraud for multiple transactions"""
        
        results = []
        for transaction in transactions:
            result = self.predict_fraud(transaction)
            result['transaction_id'] = transaction.get('transaction_id', 'unknown')
            results.append(result)
        
        return results

def test_fraud_detector():
    """Test the fraud detector with sample transactions"""
    
    # Initialize detector
    detector = FraudDetector()
    
    print("\n=== TESTING FRAUD DETECTOR ===")
    
    # Test transactions (mix of normal and suspicious)
    test_transactions = [
        {
            'transaction_id': 'test_001',
            'user_id': 'user_1234',
            'amount': 50.00,
            'merchant': 'Starbucks',
            'category': 'food',
            'timestamp': '2024-01-15 14:30:00',
            'account_age_days': 365,
            'location_risk': 'low'
        },
        {
            'transaction_id': 'test_002',
            'user_id': 'user_5678',
            'amount': 5000.00,
            'merchant': 'Online Casino',
            'category': 'gambling',
            'timestamp': '2024-01-15 02:15:00',
            'account_age_days': 5,
            'location_risk': 'high'
        },
        {
            'transaction_id': 'test_003',
            'user_id': 'user_9999',
            'amount': 1200.00,
            'merchant': 'Crypto Exchange',
            'category': 'crypto',
            'timestamp': '2024-01-15 23:45:00',
            'account_age_days': 15,
            'location_risk': 'medium'
        },
        {
            'transaction_id': 'test_004',
            'user_id': 'user_1111',
            'amount': 75.00,
            'merchant': 'Walmart',
            'category': 'retail',
            'timestamp': '2024-01-15 10:00:00',
            'account_age_days': 500,
            'location_risk': 'low'
        }
    ]
    
    # Test each transaction
    for i, transaction in enumerate(test_transactions, 1):
        print(f"\n--- Test Transaction {i} ---")
        print(f"Amount: ${transaction['amount']}")
        print(f"Merchant: {transaction['merchant']}")
        print(f"Category: {transaction['category']}")
        print(f"Time: {transaction['timestamp']}")
        print(f"Account Age: {transaction['account_age_days']} days")
        print(f"Location Risk: {transaction['location_risk']}")
        
        # Get prediction
        result = detector.predict_fraud(transaction)
        
        print(f"\n🔍 FRAUD ANALYSIS:")
        print(f"Fraud Probability: {result['fraud_probability']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
        if result['is_fraud']:
            print("⚠️  SUSPICIOUS TRANSACTION - REQUIRES REVIEW")
        else:
            print("✅ TRANSACTION APPROVED")

if __name__ == "__main__":
    test_fraud_detector()