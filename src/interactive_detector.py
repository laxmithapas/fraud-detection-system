import pandas as pd
from datetime import datetime
from fraud_detector import FraudDetector
import json

class InteractiveFraudDetector:
    """Interactive interface for fraud detection"""
    
    def __init__(self):
        self.detector = FraudDetector()
        self.transaction_history = []
    
    def get_transaction_input(self):
        """Get transaction details from user input"""
        
        print("\n" + "="*50)
        print("🔍 FRAUD DETECTION SYSTEM")
        print("="*50)
        print("Enter transaction details:")
        
        # Get transaction details
        transaction = {}
        
        # Transaction ID
        transaction['transaction_id'] = input("Transaction ID (or press Enter for auto): ").strip()
        if not transaction['transaction_id']:
            transaction['transaction_id'] = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # User ID
        transaction['user_id'] = input("User ID: ").strip() or "user_unknown"
        
        # Amount
        while True:
            try:
                amount_str = input("Transaction Amount ($): ").strip()
                transaction['amount'] = float(amount_str)
                break
            except ValueError:
                print("Please enter a valid number for amount")
        
        # Merchant
        print("\nAvailable merchants:")
        merchants = ['Amazon', 'Walmart', 'Starbucks', 'Shell Gas', 'McDonalds', 
                    'Target', 'Best Buy', 'Grocery Store', 'ATM Withdrawal', 
                    'Online Casino', 'Crypto Exchange', 'Unknown Merchant']
        
        for i, merchant in enumerate(merchants, 1):
            print(f"{i}. {merchant}")
        
        while True:
            try:
                merchant_choice = input("Select merchant (1-12) or type custom name: ").strip()
                if merchant_choice.isdigit() and 1 <= int(merchant_choice) <= 12:
                    transaction['merchant'] = merchants[int(merchant_choice) - 1]
                    break
                else:
                    transaction['merchant'] = merchant_choice
                    break
            except:
                print("Please enter a valid choice")
        
        # Category
        print("\nAvailable categories:")
        categories = ['retail', 'food', 'gas', 'entertainment', 'gambling', 
                     'crypto', 'grocery', 'electronics', 'cash', 'other']
        
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        while True:
            try:
                cat_choice = input("Select category (1-10): ").strip()
                if cat_choice.isdigit() and 1 <= int(cat_choice) <= 10:
                    transaction['category'] = categories[int(cat_choice) - 1]
                    break
                else:
                    print("Please enter a number between 1-10")
            except:
                print("Please enter a valid choice")
        
        # Timestamp
        timestamp_input = input("Transaction time (YYYY-MM-DD HH:MM or press Enter for now): ").strip()
        if not timestamp_input:
            transaction['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            try:
                # Try to parse the input
                if len(timestamp_input) == 16:  # YYYY-MM-DD HH:MM
                    datetime.strptime(timestamp_input, '%Y-%m-%d %H:%M')
                    transaction['timestamp'] = timestamp_input + ':00'
                else:
                    transaction['timestamp'] = timestamp_input
            except ValueError:
                print("Invalid time format, using current time")
                transaction['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Account age
        while True:
            try:
                age_input = input("Account age in days (or press Enter for 100): ").strip()
                if not age_input:
                    transaction['account_age_days'] = 100
                else:
                    transaction['account_age_days'] = int(age_input)
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Location risk
        print("\nLocation risk levels:")
        print("1. low")
        print("2. medium") 
        print("3. high")
        
        while True:
            try:
                risk_choice = input("Select location risk (1-3): ").strip()
                risk_levels = ['low', 'medium', 'high']
                if risk_choice.isdigit() and 1 <= int(risk_choice) <= 3:
                    transaction['location_risk'] = risk_levels[int(risk_choice) - 1]
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except:
                print("Please enter a valid choice")
        
        return transaction
    
    def analyze_transaction(self, transaction):
        """Analyze the transaction and display results"""
        
        print("\n" + "="*50)
        print("📊 TRANSACTION ANALYSIS")
        print("="*50)
        
        # Display transaction details
        print("Transaction Details:")
        print(f"  ID: {transaction['transaction_id']}")
        print(f"  User: {transaction['user_id']}")
        print(f"  Amount: ${transaction['amount']:,.2f}")
        print(f"  Merchant: {transaction['merchant']}")
        print(f"  Category: {transaction['category']}")
        print(f"  Time: {transaction['timestamp']}")
        print(f"  Account Age: {transaction['account_age_days']} days")
        print(f"  Location Risk: {transaction['location_risk']}")
        
        # Get fraud prediction
        result = self.detector.predict_fraud(transaction)
        
        print("\n🔍 FRAUD ANALYSIS RESULTS:")
        print("-" * 30)
        print(f"Fraud Probability: {result['fraud_probability']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
        # Visual indicator
        if result['fraud_probability'] >= 0.8:
            print("🚨 HIGH RISK - BLOCK TRANSACTION")
            status_color = "RED"
        elif result['fraud_probability'] >= 0.5:
            print("⚠️  MEDIUM RISK - MANUAL REVIEW REQUIRED")
            status_color = "YELLOW"
        elif result['fraud_probability'] >= 0.3:
            print("⚡ LOW RISK - MONITOR TRANSACTION")
            status_color = "ORANGE"
        else:
            print("✅ VERY LOW RISK - APPROVE TRANSACTION")
            status_color = "GREEN"
        
        # Add to history
        transaction_record = {
            **transaction,
            **result,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.transaction_history.append(transaction_record)
        
        return result
    
    def show_history(self):
        """Show transaction history"""
        
        if not self.transaction_history:
            print("\nNo transaction history available.")
            return
        
        print("\n" + "="*50)
        print("📋 TRANSACTION HISTORY")
        print("="*50)
        
        for i, txn in enumerate(self.transaction_history, 1):
            print(f"\n{i}. Transaction {txn['transaction_id']}")
            print(f"   Amount: ${txn['amount']:,.2f} | Merchant: {txn['merchant']}")
            print(f"   Fraud Probability: {txn['fraud_probability']:.1%} | Status: {txn['recommendation']}")
            print(f"   Analyzed: {txn['analysis_time']}")
    
    def save_history(self):
        """Save transaction history to file"""
        
        if not self.transaction_history:
            print("No history to save.")
            return
        
        filename = f"../data/fraud_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.transaction_history, f, indent=2, default=str)
        
        print(f"\nTransaction history saved to: {filename}")
    
    def run(self):
        """Main interactive loop"""
        
        print("🔐 Welcome to the Interactive Fraud Detection System!")
        print("This system will analyze transactions in real-time for fraud patterns.")
        
        while True:
            print("\n" + "="*50)
            print("MAIN MENU")
            print("="*50)
            print("1. Analyze New Transaction")
            print("2. View Transaction History")
            print("3. Save History to File")
            print("4. Exit")
            
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == '1':
                try:
                    transaction = self.get_transaction_input()
                    self.analyze_transaction(transaction)
                    
                    input("\nPress Enter to continue...")
                    
                except KeyboardInterrupt:
                    print("\n\nReturning to main menu...")
                except Exception as e:
                    print(f"\nError processing transaction: {e}")
                    input("Press Enter to continue...")
            
            elif choice == '2':
                self.show_history()
                input("\nPress Enter to continue...")
            
            elif choice == '3':
                self.save_history()
                input("Press Enter to continue...")
            
            elif choice == '4':
                print("\n👋 Thank you for using the Fraud Detection System!")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    app = InteractiveFraudDetector()
    app.run()