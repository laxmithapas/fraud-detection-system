import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import random
import threading
from fraud_detector import FraudDetector
import json
import os

class FraudMonitoringDashboard:
    """Real-time fraud monitoring dashboard"""
    
    def __init__(self):
        self.detector = FraudDetector()
        self.transaction_stream = []
        self.fraud_alerts = []
        self.is_running = False
        
        # Statistics tracking
        self.stats = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'legitimate_transactions': 0,
            'total_amount_processed': 0.0,
            'total_fraud_amount': 0.0,
            'avg_fraud_probability': 0.0,
            'transactions_per_minute': 0,
            'last_update': datetime.now()
        }
        
        # Configure matplotlib for real-time plotting
        plt.ion()  # Turn on interactive mode
        
    def generate_random_transaction(self):
        """Generate a realistic random transaction"""
        
        merchants = ['Amazon', 'Walmart', 'Starbucks', 'Shell Gas', 'McDonalds', 
                    'Target', 'Best Buy', 'Grocery Store', 'ATM Withdrawal', 
                    'Online Casino', 'Crypto Exchange', 'Unknown Merchant']
        
        categories = ['retail', 'food', 'gas', 'entertainment', 'gambling', 
                     'crypto', 'grocery', 'electronics', 'cash', 'other']
        
        # Generate transaction with varying fraud likelihood
        merchant = random.choice(merchants)
        category = random.choice(categories)
        
        # Bias amounts based on category for more realistic fraud patterns
        if category in ['gambling', 'crypto']:
            amount = np.random.exponential(800) + 100  # Higher amounts
        elif category == 'cash':
            amount = random.choice([20, 40, 60, 80, 100, 200, 500])
        else:
            amount = np.random.exponential(75) + 1
        
        # Time-based patterns (more fraud at night)
        current_hour = datetime.now().hour
        if random.random() < 0.3:  # 30% chance of unusual time
            hour = random.choice([1, 2, 3, 23, 0])  # Night hours
        else:
            hour = random.randint(6, 22)  # Normal hours
        
        transaction_time = datetime.now().replace(hour=hour, minute=random.randint(0, 59))
        
        transaction = {
            'transaction_id': f'live_{int(time.time())}_{random.randint(1000, 9999)}',
            'user_id': f'user_{random.randint(1000, 9999)}',
            'amount': round(amount, 2),
            'merchant': merchant,
            'category': category,
            'timestamp': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'account_age_days': random.randint(1, 1000),
            'location_risk': random.choices(['low', 'medium', 'high'], weights=[70, 20, 10])[0]
        }
        
        return transaction
    
    def process_transaction(self, transaction):
        """Process a single transaction through fraud detection"""
        
        # Get fraud prediction
        result = self.detector.predict_fraud(transaction)
        
        # Add transaction to stream
        transaction_record = {
            **transaction,
            **result,
            'processed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.transaction_stream.append(transaction_record)
        
        # Keep only last 100 transactions for memory management
        if len(self.transaction_stream) > 100:
            self.transaction_stream = self.transaction_stream[-100:]
        
        # Add to fraud alerts if suspicious
        if result['fraud_probability'] >= 0.5:
            self.fraud_alerts.append(transaction_record)
            if len(self.fraud_alerts) > 50:  # Keep last 50 alerts
                self.fraud_alerts = self.fraud_alerts[-50:]
        
        # Update statistics
        self.update_statistics(transaction_record)
        
        return transaction_record
    
    def update_statistics(self, transaction):
        """Update real-time statistics"""
        
        self.stats['total_transactions'] += 1
        self.stats['total_amount_processed'] += transaction['amount']
        
        if transaction['is_fraud']:
            self.stats['fraud_detected'] += 1
            self.stats['total_fraud_amount'] += transaction['amount']
        else:
            self.stats['legitimate_transactions'] += 1
        
        # Calculate average fraud probability
        recent_transactions = self.transaction_stream[-20:]  # Last 20 transactions
        if recent_transactions:
            self.stats['avg_fraud_probability'] = np.mean([t['fraud_probability'] for t in recent_transactions])
        
        # Calculate transactions per minute
        now = datetime.now()
        time_diff = (now - self.stats['last_update']).total_seconds() / 60
        if time_diff > 0:
            self.stats['transactions_per_minute'] = len(recent_transactions) / time_diff
        
        self.stats['last_update'] = now
    
    def display_dashboard(self):
        """Display the monitoring dashboard"""
        
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔐" + "="*70 + "🔐")
        print("🚨        REAL-TIME FRAUD DETECTION MONITORING DASHBOARD        🚨")
        print("🔐" + "="*70 + "🔐")
        
        # Current time
        print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚡ Status: {'MONITORING' if self.is_running else 'STOPPED'}")
        
        print("\n📊 REAL-TIME STATISTICS")
        print("-" * 50)
        print(f"🔢 Total Transactions Processed: {self.stats['total_transactions']:,}")
        print(f"🚨 Fraud Detected: {self.stats['fraud_detected']:,}")
        print(f"✅ Legitimate Transactions: {self.stats['legitimate_transactions']:,}")
        
        # Calculate fraud rate
        if self.stats['total_transactions'] > 0:
            fraud_rate = (self.stats['fraud_detected'] / self.stats['total_transactions']) * 100
            print(f"📈 Current Fraud Rate: {fraud_rate:.1f}%")
        
        print(f"💰 Total Amount Processed: ${self.stats['total_amount_processed']:,.2f}")
        print(f"💸 Fraudulent Amount: ${self.stats['total_fraud_amount']:,.2f}")
        print(f"🎯 Avg Fraud Probability: {self.stats['avg_fraud_probability']:.1%}")
        print(f"⚡ Transactions/Min: {self.stats['transactions_per_minute']:.1f}")
        
        # Recent transactions
        print("\n📋 LAST 10 TRANSACTIONS")
        print("-" * 80)
        print(f"{'ID':<15} {'Amount':<10} {'Merchant':<15} {'Fraud%':<8} {'Status':<8}")
        print("-" * 80)
        
        recent = self.transaction_stream[-10:] if self.transaction_stream else []
        for txn in reversed(recent):
            status = "🚨BLOCK" if txn['is_fraud'] else "✅PASS"
            txn_id = txn['transaction_id'][-12:]  # Show last 12 chars
            print(f"{txn_id:<15} ${txn['amount']:<9.2f} {txn['merchant'][:14]:<15} {txn['fraud_probability']:.1%} {status:<8}")
        
        # Fraud alerts
        if self.fraud_alerts:
            print("\n🚨 RECENT FRAUD ALERTS")
            print("-" * 50)
            recent_alerts = self.fraud_alerts[-5:]  # Show last 5 alerts
            for alert in reversed(recent_alerts):
                print(f"⚠️  ${alert['amount']:,.2f} at {alert['merchant']} - Risk: {alert['fraud_probability']:.1%}")
        
        print("\n" + "="*70)
        print("Press Ctrl+C to stop monitoring")
        print("="*70)
    
    def run_simulation(self, transactions_per_minute=10):
        """Run the fraud detection simulation"""
        
        print("🚀 Starting fraud detection monitoring...")
        print(f"📊 Processing {transactions_per_minute} transactions per minute")
        print("🔍 Real-time fraud detection active")
        
        self.is_running = True
        delay = 60.0 / transactions_per_minute  # Delay between transactions
        
        try:
            while self.is_running:
                # Generate and process transaction
                transaction = self.generate_random_transaction()
                processed_txn = self.process_transaction(transaction)
                
                # Display dashboard
                self.display_dashboard()
                
                # Log fraud alerts to console
                if processed_txn['is_fraud']:
                    print(f"\n🚨 FRAUD ALERT: {processed_txn['transaction_id']}")
                    print(f"   Amount: ${processed_txn['amount']:,.2f}")
                    print(f"   Merchant: {processed_txn['merchant']}")
                    print(f"   Fraud Probability: {processed_txn['fraud_probability']:.1%}")
                
                # Wait before next transaction
                time.sleep(delay)
                
        except KeyboardInterrupt:
            self.is_running = False
            print("\n\n🛑 Monitoring stopped by user")
            self.show_final_summary()
    
    def show_final_summary(self):
        """Show final monitoring summary"""
        
        print("\n" + "="*50)
        print("📊 MONITORING SESSION SUMMARY")
        print("="*50)
        
        print(f"⏱️  Session Duration: Active until stopped")
        print(f"🔢 Total Transactions: {self.stats['total_transactions']:,}")
        print(f"🚨 Fraud Detected: {self.stats['fraud_detected']:,}")
        print(f"💰 Amount Processed: ${self.stats['total_amount_processed']:,.2f}")
        
        if self.stats['total_transactions'] > 0:
            fraud_rate = (self.stats['fraud_detected'] / self.stats['total_transactions']) * 100
            print(f"📈 Overall Fraud Rate: {fraud_rate:.1f}%")
        
        # Save session data
        session_data = {
            'session_summary': self.stats,
            'transactions': self.transaction_stream,
            'fraud_alerts': self.fraud_alerts,
            'session_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f"../data/monitoring_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"💾 Session data saved to: {filename}")
        print("\n✅ Monitoring dashboard closed successfully!")

def main():
    """Main function to run the monitoring dashboard"""
    
    dashboard = FraudMonitoringDashboard()
    
    print("🔐 Fraud Detection Monitoring Dashboard")
    print("=" * 40)
    print("This dashboard will simulate real-time transaction processing")
    print("and show fraud detection results in real-time.")
    
    try:
        # Ask user for monitoring parameters
        print("\n⚙️  Configuration:")
        
        while True:
            try:
                tpm = input("Transactions per minute (default 6): ").strip()
                if not tpm:
                    tpm = 6
                else:
                    tpm = int(tpm)
                break
            except ValueError:
                print("Please enter a valid number")
        
        print(f"\n🚀 Starting monitoring with {tpm} transactions per minute...")
        print("📊 Dashboard will update in real-time")
        print("🛑 Press Ctrl+C to stop monitoring\n")
        
        input("Press Enter to start monitoring...")
        
        # Start the simulation
        dashboard.run_simulation(transactions_per_minute=tpm)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()