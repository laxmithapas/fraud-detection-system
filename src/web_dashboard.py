from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import threading
import time
import random
from datetime import datetime
from fraud_detector import FraudDetector
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fraud_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebDashboard:
    def __init__(self):
        self.detector = FraudDetector()
        self.is_monitoring = False
        self.stats = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'legitimate_transactions': 0,
            'total_amount': 0.0,
            'fraud_amount': 0.0,
            'fraud_rate': 0.0,
            'avg_amount': 0.0
        }
        self.recent_transactions = []
        self.fraud_alerts = []
        self.monitoring_thread = None

    def generate_transaction(self):
        """Generate a random transaction"""
        merchants = ['Amazon', 'Walmart', 'Starbucks', 'Shell Gas', 'McDonalds', 
                    'Target', 'Best Buy', 'Grocery Store', 'ATM Withdrawal', 
                    'Online Casino', 'Crypto Exchange', 'Gas Station']
        
        categories = ['retail', 'food', 'gas', 'entertainment', 'gambling', 
                     'crypto', 'grocery', 'electronics', 'cash', 'other']
        
        merchant = random.choice(merchants)
        category = random.choice(categories)
        
        # Generate amounts with fraud bias
        if category in ['gambling', 'crypto']:
            amount = np.random.exponential(500) + 50
        else:
            amount = np.random.exponential(75) + 1
        
        return {
            'transaction_id': f'web_{int(time.time())}_{random.randint(1000, 9999)}',
            'user_id': f'user_{random.randint(1000, 9999)}',
            'amount': round(amount, 2),
            'merchant': merchant,
            'category': category,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'account_age_days': random.randint(1, 800),
            'location_risk': random.choices(['low', 'medium', 'high'], weights=[70, 20, 10])[0]
        }

    def process_transaction(self, transaction):
        """Process transaction and update stats"""
        result = self.detector.predict_fraud(transaction)
        
        # Create transaction record
        txn_record = {
            **transaction,
            **result,
            'processed_at': datetime.now().strftime('%H:%M:%S')
        }
        
        # Update statistics
        self.stats['total_transactions'] += 1
        self.stats['total_amount'] += transaction['amount']
        
        if result['is_fraud']:
            self.stats['fraud_detected'] += 1
            self.stats['fraud_amount'] += transaction['amount']
            self.fraud_alerts.append(txn_record)
            if len(self.fraud_alerts) > 20:
                self.fraud_alerts = self.fraud_alerts[-20:]
        else:
            self.stats['legitimate_transactions'] += 1
        
        # Calculate rates
        if self.stats['total_transactions'] > 0:
            self.stats['fraud_rate'] = (self.stats['fraud_detected'] / self.stats['total_transactions']) * 100
            self.stats['avg_amount'] = self.stats['total_amount'] / self.stats['total_transactions']
        
        # Add to recent transactions
        self.recent_transactions.append(txn_record)
        if len(self.recent_transactions) > 50:
            self.recent_transactions = self.recent_transactions[-50:]
        
        return txn_record

    def monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            transaction = self.generate_transaction()
            processed_txn = self.process_transaction(transaction)
            
            # Emit real-time updates
            socketio.emit('new_transaction', processed_txn)
            socketio.emit('stats_update', self.stats)
            
            if processed_txn['is_fraud']:
                socketio.emit('fraud_alert', processed_txn)
            
            time.sleep(2)  # Process transaction every 2 seconds

# Initialize dashboard
dashboard = WebDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(dashboard.stats)

@app.route('/api/transactions')
def get_transactions():
    """Get recent transactions"""
    return jsonify(dashboard.recent_transactions[-10:])

@app.route('/api/alerts')
def get_alerts():
    """Get fraud alerts"""
    return jsonify(dashboard.fraud_alerts[-5:])

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    """Analyze a single transaction"""
    try:
        transaction = request.json
        result = dashboard.detector.predict_fraud(transaction)
        return jsonify({
            'success': True,
            'result': result,
            'transaction': transaction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('start_monitoring')
def start_monitoring():
    """Start real-time monitoring"""
    if not dashboard.is_monitoring:
        dashboard.is_monitoring = True
        dashboard.monitoring_thread = threading.Thread(target=dashboard.monitoring_loop)
        dashboard.monitoring_thread.daemon = True
        dashboard.monitoring_thread.start()
        emit('monitoring_status', {'status': 'started'})

@socketio.on('stop_monitoring')
def stop_monitoring():
    """Stop real-time monitoring"""
    dashboard.is_monitoring = False
    emit('monitoring_status', {'status': 'stopped'})

@socketio.on('reset_stats')
def reset_stats():
    """Reset all statistics"""
    dashboard.stats = {
        'total_transactions': 0,
        'fraud_detected': 0,
        'legitimate_transactions': 0,
        'total_amount': 0.0,
        'fraud_amount': 0.0,
        'fraud_rate': 0.0,
        'avg_amount': 0.0
    }
    dashboard.recent_transactions = []
    dashboard.fraud_alerts = []
    emit('stats_update', dashboard.stats)
    emit('transactions_cleared')

if __name__ == '__main__':
    print("🚀 Starting Fraud Detection Web Dashboard...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)