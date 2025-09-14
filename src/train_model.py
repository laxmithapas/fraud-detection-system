import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def prepare_features(df):
    """Prepare features for machine learning"""
    
    print("Preparing features...")
    
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Convert timestamp to useful features
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
    df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)).astype(int)
    
    # Encode categorical variables
    le_merchant = LabelEncoder()
    le_category = LabelEncoder()
    le_location = LabelEncoder()
    
    df_processed['merchant_encoded'] = le_merchant.fit_transform(df_processed['merchant'])
    df_processed['category_encoded'] = le_category.fit_transform(df_processed['category'])
    df_processed['location_risk_encoded'] = le_location.fit_transform(df_processed['location_risk'])
    
    # Create amount-based features
    df_processed['amount_log'] = np.log1p(df_processed['amount'])  # Log transform for skewed amounts
    df_processed['is_high_amount'] = (df_processed['amount'] > df_processed['amount'].quantile(0.95)).astype(int)
    
    # Account age features
    df_processed['is_new_account'] = (df_processed['account_age_days'] < 30).astype(int)
    
    # Select features for the model
    feature_columns = [
        'amount', 'amount_log', 'is_high_amount',
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'account_age_days', 'is_new_account',
        'merchant_encoded', 'category_encoded', 'location_risk_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['is_fraud']
    
    # Save encoders for later use
    if not os.path.exists('../models'):
        os.makedirs('../models')
    
    joblib.dump(le_merchant, '../models/merchant_encoder.pkl')
    joblib.dump(le_category, '../models/category_encoder.pkl')
    joblib.dump(le_location, '../models/location_encoder.pkl')
    
    print(f"Features prepared: {feature_columns}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_columns

def train_models(X, y):
    """Train multiple models and compare performance"""
    
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, '../models/scaler.pkl')
    
    models = {}
    results = {}
    
    print("\n=== TRAINING MODELS ===")
    
    # 1. Random Forest
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'auc_score': roc_auc_score(y_test, rf_proba)
    }
    
    # 2. Logistic Regression
    print("2. Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'auc_score': roc_auc_score(y_test, lr_proba)
    }
    
    return models, results, X_test, y_test, X_test_scaled

def evaluate_models(models, results, X_test, y_test, X_test_scaled):
    """Evaluate and compare model performance"""
    
    print("\n=== MODEL EVALUATION ===")
    
    for model_name, result in results.items():
        print(f"\n--- {model_name.upper()} RESULTS ---")
        print(f"AUC Score: {result['auc_score']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, result['predictions']))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, result['predictions'])
        print(cm)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
    
    # Choose best model based on AUC score
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_model = models[best_model_name]
    
    print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
    print(f"AUC Score: {results[best_model_name]['auc_score']:.4f}")
    
    # Save best model
    joblib.dump(best_model, '../models/best_fraud_model.pkl')
    
    # Save model metadata
    metadata = {
        'model_type': best_model_name,
        'auc_score': results[best_model_name]['auc_score'],
        'features': list(X_test.columns)
    }
    joblib.dump(metadata, '../models/model_metadata.pkl')
    
    print(f"\nBest model saved as '../models/best_fraud_model.pkl'")
    
    return best_model_name, best_model

def show_feature_importance(model, feature_columns, model_name):
    """Show feature importance for tree-based models"""
    
    if hasattr(model, 'feature_importances_'):
        print(f"\n=== FEATURE IMPORTANCE ({model_name}) ===")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance)
        
        # Save feature importance
        feature_importance.to_csv('../models/feature_importance.csv', index=False)
        print("\nFeature importance saved to '../models/feature_importance.csv'")

if __name__ == "__main__":
    print("Starting fraud detection model training...")
    
    # Load data
    df = pd.read_csv('../data/transactions.csv')
    
    # Prepare features
    X, y, feature_columns = prepare_features(df)
    
    # Train models
    models, results, X_test, y_test, X_test_scaled = train_models(X, y)
    
    # Evaluate models
    best_model_name, best_model = evaluate_models(models, results, X_test, y_test, X_test_scaled)
    
    # Show feature importance
    show_feature_importance(best_model, feature_columns, best_model_name)
    
    print("\nModel training complete!")
    print(f"Best model: {best_model_name}")
    print("Files saved in '../models/' directory")