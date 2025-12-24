# sentiment_analyzer_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import uuid
import datetime
from datetime import datetime, timedelta
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import joblib
import re
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analyzer with Active Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .uncertain-sample {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFC107;
        margin: 0.5rem 0;
    }
    .certain-sample {
        background-color: #D4EDDA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28A745;
        margin: 0.5rem 0;
    }
    .feedback-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 0.5rem 0;
    }
    .query-result {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9C27B0;
        margin: 0.5rem 0;
    }
    .sentiment-positive { color: #28A745; font-weight: bold; }
    .sentiment-negative { color: #DC3545; font-weight: bold; }
    .sentiment-neutral { color: #6C757D; font-weight: bold; }
    .confidence-high { color: #28A745; }
    .confidence-medium { color: #FFC107; }
    .confidence-low { color: #DC3545; }
    </style>
""", unsafe_allow_html=True)

# Data storage paths
DATA_DIR = "data"
MODELS_DIR = "models"
FEEDBACK_DIR = "feedback"
USERS_FILE = "users.json"
LOGS_FILE = "activity_logs.json"
FEEDBACK_FILE = "feedback_data.json"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'users' not in st.session_state:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                st.session_state.users = json.load(f)
        else:
            st.session_state.users = {}
            # Create default admin user
            st.session_state.users['admin'] = {
                'password': hashlib.sha256('admin123'.encode()).hexdigest(),
                'role': 'admin',
                'email': 'admin@system.com',
                'created_at': datetime.now().isoformat()
            }
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'workspaces' not in st.session_state:
        st.session_state.workspaces = {}
    
    if 'logs' not in st.session_state:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'r') as f:
                st.session_state.logs = json.load(f)
        else:
            st.session_state.logs = []
    
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    
    if 'models' not in st.session_state:
        st.session_state.models = {}
    
    if 'feedback_data' not in st.session_state:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                st.session_state.feedback_data = json.load(f)
        else:
            st.session_state.feedback_data = {
                'queries': [],
                'corrections': [],
                'user_feedback': []
            }
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

init_session_state()

# Helper functions
def log_activity(action, details, user=None):
    """Log user activity"""
    if user is None:
        user = st.session_state.current_user if st.session_state.current_user else "Anonymous"
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user,
        'action': action,
        'details': details
    }
    st.session_state.logs.append(log_entry)
    
    # Save to file
    with open(LOGS_FILE, 'w') as f:
        json.dump(st.session_state.logs, f, indent=2)

def save_users():
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(st.session_state.users, f, indent=2)

def save_feedback():
    """Save feedback data to file"""
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(st.session_state.feedback_data, f, indent=2)

def create_workspace(name, owner):
    """Create a new workspace"""
    workspace_id = str(uuid.uuid4())[:8]
    st.session_state.workspaces[workspace_id] = {
        'name': name,
        'owner': owner,
        'created_at': datetime.now().isoformat(),
        'datasets': [],
        'models': [],
        'members': [owner]
    }
    log_activity(f"Created workspace '{name}'", f"Workspace ID: {workspace_id}", owner)
    return workspace_id

# Authentication functions
def login(username, password):
    """User login"""
    if username in st.session_state.users:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        if st.session_state.users[username]['password'] == hashed_pw:
            st.session_state.current_user = username
            log_activity("User logged in", f"Username: {username}")
            return True
    return False

def register(username, password, email):
    """Register new user"""
    if username in st.session_state.users:
        return False, "Username already exists"
    
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    st.session_state.users[username] = {
        'password': hashed_pw,
        'role': 'user',
        'email': email,
        'created_at': datetime.now().isoformat()
    }
    save_users()
    log_activity("New user registered", f"Username: {username}")
    return True, "Registration successful"

# ML Model functions with Aspect Extraction
class SentimentModel:
    def __init__(self, model_name="default"):
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.is_trained = False
        self.classes_ = None
        self.model_name = model_name
        self.training_date = None
        
        # Define common aspects and their keywords
        self.aspect_keywords = {
            'product': ['product', 'item', 'goods', 'merchandise', 'purchase'],
            'quality': ['quality', 'durable', 'sturdy', 'flimsy', 'material', 'build'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value'],
            'service': ['service', 'support', 'customer', 'help', 'assistance'],
            'delivery': ['delivery', 'shipping', 'arrived', 'shipped', 'package'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient'],
            'design': ['design', 'look', 'appearance', 'style', 'color'],
            'usability': ['easy', 'difficult', 'user-friendly', 'complicated', 'simple']
        }
    
    def extract_aspects(self, text):
        """Extract aspects from text based on keywords"""
        aspects = []
        text_lower = text.lower()
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    aspects.append(aspect)
                    break  # Found one keyword for this aspect
        
        return list(set(aspects))  # Remove duplicates
    
    def train(self, X, y):
        """Train the model"""
        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)
        self.is_trained = True
        self.classes_ = self.classifier.classes_
        self.training_date = datetime.now().isoformat()
        return X_vec
    
    def predict_with_details(self, text):
        """Predict sentiment with detailed information"""
        if not self.is_trained:
            # Return mock predictions for demo
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text
            
            results = []
            for t in texts:
                # Mock sentiment prediction
                if "good" in t.lower() or "great" in t.lower() or "excellent" in t.lower():
                    pred = "positive"
                    confidence = 75.0
                elif "bad" in t.lower() or "poor" in t.lower() or "terrible" in t.lower():
                    pred = "negative"
                    confidence = 70.0
                else:
                    pred = "neutral"
                    confidence = 60.0
                
                # Extract aspects
                aspects = self.extract_aspects(t)
                
                results.append({
                    'text': t,
                    'predicted_sentiment': pred,
                    'confidence': confidence,
                    'aspects': aspects,
                    'probabilities': {'positive': 40, 'negative': 30, 'neutral': 30},
                    'all_predictions': [('positive', 40), ('negative', 30), ('neutral', 30)]
                })
            
            return results if len(results) > 1 else results[0]
        
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        X_vec = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X_vec)
        probabilities = self.classifier.predict_proba(X_vec)
        
        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
            confidence = np.max(prob)
            pred_class = self.classes_[np.argmax(prob)]
            confidence_percent = float(confidence * 100)
            
            # Extract aspects
            aspects = self.extract_aspects(text)
            
            # Get probability for each class
            class_probs = {}
            for idx, cls in enumerate(self.classes_):
                class_probs[cls] = float(prob[idx] * 100)
            
            results.append({
                'text': text,
                'predicted_sentiment': pred_class,
                'confidence': confidence_percent,
                'aspects': aspects,
                'probabilities': class_probs,
                'all_predictions': list(zip(self.classes_, [float(p*100) for p in prob]))
            })
        
        return results if len(results) > 1 else results[0]
    
    def predict(self, X):
        """Predict sentiment (legacy method)"""
        results = self.predict_with_details(X)
        if isinstance(results, list):
            predictions = [r['predicted_sentiment'] for r in results]
            confidences = [r['confidence'] / 100 for r in results]
            return predictions, confidences
        else:
            return [results['predicted_sentiment']], [results['confidence'] / 100]
    
    def save(self, path):
        """Save model to file"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_trained': self.is_trained,
            'classes_': self.classes_,
            'model_name': self.model_name,
            'training_date': self.training_date
        }
        joblib.dump(model_data, path)
    
    def load(self, path):
        """Load model from file"""
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        self.classes_ = model_data['classes_']
        self.model_name = model_data.get('model_name', 'default')
        self.training_date = model_data.get('training_date')

# Sample datasets
def load_sample_dataset():
    """Load sample dataset for demo"""
    data = {
        'text': [
            "This product is amazing! I love the quality and design.",
            "Very disappointed with the customer service and delivery time.",
            "Average product, nothing special but good price.",
            "Best purchase ever! The performance exceeded my expectations.",
            "Waste of money, product doesn't work as advertised.",
            "Good value for the price, but shipping was slow.",
            "Terrible customer service and poor quality materials.",
            "Exceeded my expectations in terms of usability and performance.",
            "Okay product, could be better design but works fine.",
            "Absolutely love it! Will buy again, great service and quality.",
            "The price is too high for what you get.",
            "Excellent performance and very user-friendly interface.",
            "Poor build quality but decent service.",
            "Fast delivery and good packaging.",
            "Not worth the money, performance is disappointing."
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                     'positive', 'negative', 'positive', 'neutral', 'positive',
                     'negative', 'positive', 'negative', 'positive', 'negative'],
        'aspect': ['product,quality,design', 'service,delivery', 'product,price', 
                  'performance', 'product', 'price,delivery', 'service,quality',
                  'usability,performance', 'design', 'service,quality',
                  'price', 'performance,usability', 'quality,service',
                  'delivery', 'price,performance']
    }
    return pd.DataFrame(data)

def generate_sample_csv():
    """Generate a sample CSV for download"""
    df = load_sample_dataset()
    return df.to_csv(index=False)

def create_mock_model_files():
    """Create some mock model files for demonstration"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Create a simple mock model if none exist
    if len([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]) == 0:
        model = SentimentModel(model_name="demo_model")
        
        # Train with sample data
        df = load_sample_dataset()
        X = df['text'].tolist()
        y = df['sentiment'].tolist()
        model.train(X, y)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, "demo_model_20240101.pkl")
        model.save(model_path)
        
        # Create a few more mock models
        for i in range(2):
            model2 = SentimentModel(model_name=f"model_v{i+1}")
            model2.train(X, y)
            model2.save(os.path.join(MODELS_DIR, f"model_v{i+1}_20240101.pkl"))

# Feedback Management Functions
def save_feedback_entry(query, predicted_sentiment, corrected_sentiment=None, 
                       user_feedback=None, aspects=None, confidence=None):
    """Save feedback entry"""
    feedback_id = str(uuid.uuid4())[:8]
    
    entry = {
        'id': feedback_id,
        'timestamp': datetime.now().isoformat(),
        'user': st.session_state.current_user,
        'query': query,
        'predicted_sentiment': predicted_sentiment,
        'corrected_sentiment': corrected_sentiment,
        'user_feedback': user_feedback,
        'aspects': aspects if aspects else [],
        'confidence': confidence,
        'is_corrected': corrected_sentiment is not None
    }
    
    # Save query to history
    st.session_state.query_history.append({
        'query': query,
        'prediction': predicted_sentiment,
        'timestamp': datetime.now().isoformat()
    })
    
    # Save to feedback data
    st.session_state.feedback_data['queries'].append(entry)
    
    # If correction was made, add to corrections
    if corrected_sentiment is not None:
        correction_entry = entry.copy()
        correction_entry['correction_type'] = 'manual_correction'
        st.session_state.feedback_data['corrections'].append(correction_entry)
    
    # Save to file
    save_feedback()
    
    log_activity("Feedback submitted", f"Query: {query[:50]}...", st.session_state.current_user)
    
    return feedback_id

def get_feedback_stats():
    """Get feedback statistics"""
    total_queries = len(st.session_state.feedback_data['queries'])
    total_corrections = len(st.session_state.feedback_data['corrections'])
    
    if total_queries > 0:
        correction_rate = (total_corrections / total_queries) * 100
    else:
        correction_rate = 0
    
    # Sentiment distribution
    sentiments = defaultdict(int)
    for query in st.session_state.feedback_data['queries']:
        pred = query['predicted_sentiment']
        sentiments[pred] += 1
    
    return {
        'total_queries': total_queries,
        'total_corrections': total_corrections,
        'correction_rate': correction_rate,
        'sentiment_distribution': dict(sentiments)
    }

# ============================================
# ADMIN PANEL PAGE
# ============================================
def admin_panel_page():
    st.markdown("<h1 class='main-header'>👥 Admin Panel</h1>", unsafe_allow_html=True)
    
    tabs = st.tabs(["👤 Users Management", "📊 System Stats", "📝 Activity Logs", "⚙️ System Settings"])
    
    with tabs[0]:  # Users Management
        st.markdown("<h2 class='sub-header'>User Management</h2>", unsafe_allow_html=True)
        
        # Display users in a table
        if st.session_state.users:
            users_list = []
            for username, user_data in st.session_state.users.items():
                users_list.append({
                    'Username': username,
                    'Role': user_data.get('role', 'user'),
                    'Email': user_data.get('email', 'N/A'),
                    'Created': user_data.get('created_at', 'N/A')
                })
            
            users_df = pd.DataFrame(users_list)
            st.dataframe(users_df, use_container_width=True)
        else:
            st.info("No users found")
        
        # User actions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Remove User")
            if st.session_state.users:
                user_to_remove = st.selectbox(
                    "Select user to remove",
                    list(st.session_state.users.keys())
                )
                
                if st.button("🗑️ Remove User", key="remove_user"):
                    if user_to_remove != 'admin' and user_to_remove != st.session_state.current_user:
                        del st.session_state.users[user_to_remove]
                        save_users()
                        log_activity("User removed", f"Removed user: {user_to_remove}")
                        st.success(f"User '{user_to_remove}' removed successfully!")
                        st.rerun()
                    else:
                        st.error("Cannot remove admin or current user!")
        
        with col2:
            st.markdown("### Reset Password")
            if st.session_state.users:
                user_to_reset = st.selectbox(
                    "Select user for password reset",
                    list(st.session_state.users.keys())
                )
                new_password = st.text_input("New Password", type="password", key="new_pass")
                confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass")
                
                if st.button("🔐 Reset Password", key="reset_pass"):
                    if new_password and new_password == confirm_password:
                        hashed_pw = hashlib.sha256(new_password.encode()).hexdigest()
                        st.session_state.users[user_to_reset]['password'] = hashed_pw
                        save_users()
                        log_activity("Password reset", f"User: {user_to_reset}")
                        st.success(f"Password for '{user_to_reset}' reset successfully!")
                    else:
                        st.error("Passwords don't match or are empty!")
        
        # Add new user
        st.markdown("### Add New User")
        with st.form("add_user_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_username = st.text_input("Username")
            with col2:
                new_user_email = st.text_input("Email")
            with col3:
                new_user_role = st.selectbox("Role", ["user", "admin"])
            
            new_user_password = st.text_input("Password", type="password")
            
            if st.form_submit_button("➕ Add User"):
                if new_username and new_user_password:
                    if new_username not in st.session_state.users:
                        hashed_pw = hashlib.sha256(new_user_password.encode()).hexdigest()
                        st.session_state.users[new_username] = {
                            'password': hashed_pw,
                            'role': new_user_role,
                            'email': new_user_email,
                            'created_at': datetime.now().isoformat()
                        }
                        save_users()
                        log_activity("User added by admin", f"Username: {new_username}")
                        st.success(f"User '{new_username}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Username already exists!")
                else:
                    st.error("Username and password are required!")
    
    with tabs[1]:  # System Stats
        st.markdown("<h2 class='sub-header'>System Statistics</h2>", unsafe_allow_html=True)
        
        # Calculate stats
        total_users = len(st.session_state.users)
        total_workspaces = len(st.session_state.workspaces)
        
        # Count models
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
            total_models = len(model_files)
        except:
            total_models = 0
        
        total_logs = len(st.session_state.logs)
        feedback_stats = get_feedback_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", total_users)
        
        with col2:
            st.metric("Total Workspaces", total_workspaces)
        
        with col3:
            st.metric("Saved Models", total_models)
        
        with col4:
            st.metric("Activity Logs", total_logs)
        
        # Additional stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", feedback_stats['total_queries'])
        
        with col2:
            st.metric("Total Corrections", feedback_stats['total_corrections'])
        
        with col3:
            st.metric("Correction Rate", f"{feedback_stats['correction_rate']:.1f}%")
        
        # System health
        st.markdown("### System Health")
        health_cols = st.columns(3)
        
        with health_cols[0]:
            # Disk usage (simulated)
            st.progress(65, text="Disk Usage: 65%")
        
        with health_cols[1]:
            # Memory usage (simulated)
            st.progress(42, text="Memory Usage: 42%")
        
        with health_cols[2]:
            # CPU usage (simulated)
            st.progress(28, text="CPU Usage: 28%")
    
    with tabs[2]:  # Activity Logs
        st.markdown("<h2 class='sub-header'>Activity History</h2>", unsafe_allow_html=True)
        
        if st.session_state.logs:
            # Convert to DataFrame
            logs_df = pd.DataFrame(st.session_state.logs)
            logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
            logs_df = logs_df.sort_values('timestamp', ascending=False)
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Date filter
                min_date = logs_df['timestamp'].min().date()
                max_date = logs_df['timestamp'].max().date()
                selected_date = st.date_input(
                    "Filter by date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col2:
                # User filter
                all_users = ["All"] + list(logs_df['user'].unique())
                selected_user = st.selectbox("Filter by user", all_users)
            
            with col3:
                # Action filter
                all_actions = ["All"] + list(logs_df['action'].unique())
                selected_action = st.selectbox("Filter by action", all_actions)
            
            # Apply filters
            filtered_logs = logs_df.copy()
            
            if selected_date:
                filtered_logs = filtered_logs[
                    filtered_logs['timestamp'].dt.date == selected_date
                ]
            
            if selected_user != "All":
                filtered_logs = filtered_logs[filtered_logs['user'] == selected_user]
            
            if selected_action != "All":
                filtered_logs = filtered_logs[filtered_logs['action'] == selected_action]
            
            # Display logs
            st.write(f"**Showing {len(filtered_logs)} of {len(logs_df)} logs**")
            
            # Display in expandable sections
            for idx, log in filtered_logs.head(50).iterrows():  # Show first 50
                with st.expander(f"{log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {log['user']} - {log['action']}"):
                    st.write(f"**User:** {log['user']}")
                    st.write(f"**Action:** {log['action']}")
                    st.write(f"**Details:** {log['details']}")
                    st.write(f"**Timestamp:** {log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Export logs
            if st.button("📥 Export Logs as CSV"):
                csv = filtered_logs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"activity_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("No activity logs found")
    
    with tabs[3]:  # System Settings
        st.markdown("<h2 class='sub-header'>System Configuration</h2>", unsafe_allow_html=True)
        
        # Configuration sections
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            st.markdown("### Active Learning Settings")
            
            # Active learning threshold
            al_threshold = st.slider(
                "Confidence Threshold for Active Learning",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Samples with confidence below this threshold will be marked for review"
            )
            
            # Batch size for active learning
            batch_size = st.number_input(
                "Batch Size for Review",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of uncertain samples to show at once"
            )
            
            # Model settings
            st.markdown("### Model Settings")
            default_model_type = st.selectbox(
                "Default Model Type",
                ["Random Forest", "SVM", "Naive Bayes", "Gradient Boosting"],
                index=0
            )
            
            retrain_frequency = st.selectbox(
                "Auto-retrain Frequency",
                ["Never", "Daily", "Weekly", "After 10 corrections", "After 50 corrections"],
                index=0
            )
        
        with settings_col2:
            st.markdown("### System Preferences")
            
            # Data retention
            retention_days = st.slider(
                "Data Retention Period (days)",
                min_value=7,
                max_value=365,
                value=90,
                help="How long to keep user data and logs"
            )
            
            # Email notifications
            email_notifications = st.checkbox(
                "Enable Email Notifications",
                value=False,
                help="Send notifications for system events"
            )
            
            if email_notifications:
                smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", 587)
                notification_email = st.text_input("Notification Email")
            
            # Backup settings
            st.markdown("### Backup Settings")
            auto_backup = st.checkbox("Enable Auto Backup", value=True)
            
            if auto_backup:
                backup_frequency = st.selectbox(
                    "Backup Frequency",
                    ["Daily", "Weekly", "Monthly"],
                    index=0
                )
        
        # Save settings
        if st.button("💾 Save All Settings", type="primary"):
            settings = {
                'al_threshold': al_threshold,
                'batch_size': batch_size,
                'default_model_type': default_model_type,
                'retrain_frequency': retrain_frequency,
                'retention_days': retention_days,
                'email_notifications': email_notifications,
                'auto_backup': auto_backup,
                'backup_frequency': backup_frequency if auto_backup else None,
                'last_updated': datetime.now().isoformat(),
                'updated_by': st.session_state.current_user
            }
            
            # Save settings to file
            settings_file = "system_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            log_activity("System settings updated", f"Updated by {st.session_state.current_user}")
            st.success("✅ Settings saved successfully!")
            
            # Show summary
            with st.expander("Settings Summary"):
                st.json(settings)

# ============================================
# WORKSPACES PAGE
# ============================================
def workspaces_page():
    st.markdown("<h1 class='main-header'>📁 Workspaces Management</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3>Your Workspaces</h3>", unsafe_allow_html=True)
        
        if not st.session_state.workspaces:
            st.info("You don't have any workspaces yet. Create one to get started!")
        else:
            for ws_id, ws_data in st.session_state.workspaces.items():
                with st.expander(f"📂 {ws_data['name']} (ID: {ws_id})"):
                    st.write(f"**Owner:** {ws_data['owner']}")
                    st.write(f"**Created:** {ws_data['created_at']}")
                    st.write(f"**Members:** {', '.join(ws_data['members'])}")
                    st.write(f"**Datasets:** {len(ws_data['datasets'])}")
                    st.write(f"**Models:** {len(ws_data['models'])}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button(f"View Datasets", key=f"view_{ws_id}"):
                            st.session_state.current_workspace = ws_id
                    with col_b:
                        if st.button(f"Upload Data", key=f"upload_{ws_id}"):
                            st.info("Upload functionality would be implemented here")
                    with col_c:
                        if st.button(f"Delete", key=f"del_{ws_id}"):
                            if ws_data['owner'] == st.session_state.current_user or st.session_state.users[st.session_state.current_user]['role'] == 'admin':
                                del st.session_state.workspaces[ws_id]
                                log_activity("Workspace deleted", f"Workspace ID: {ws_id}")
                                st.success(f"Workspace deleted!")
                                st.rerun()
                            else:
                                st.error("Only workspace owner or admin can delete workspace!")
    
    with col2:
        st.markdown("<h3>Create New Workspace</h3>", unsafe_allow_html=True)
        
        with st.form("new_workspace"):
            ws_name = st.text_input("Workspace Name")
            ws_description = st.text_area("Description")
            
            if st.form_submit_button("Create Workspace"):
                if ws_name:
                    ws_id = create_workspace(ws_name, st.session_state.current_user)
                    st.success(f"✅ Workspace '{ws_name}' created! ID: {ws_id}")
                    st.rerun()
                else:
                    st.error("Please enter a workspace name")

# ============================================
# MODEL MANAGEMENT PAGE
# ============================================
def model_management_page():
    st.markdown("<h1 class='main-header'>🤖 Model Management</h1>", unsafe_allow_html=True)
    
    # Create mock models if none exist
    create_mock_model_files()
    
    # List existing models
    st.markdown("<h2 class='sub-header'>📚 Model Versions</h2>", unsafe_allow_html=True)
    
    # Check for saved models
    try:
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        model_files.sort(reverse=True)
    except:
        model_files = []
    
    if model_files:
        st.info(f"Found {len(model_files)} saved models")
        for model_file in model_files:
            with st.expander(f"🧠 {model_file}"):
                model_path = os.path.join(MODELS_DIR, model_file)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Load", key=f"load_{model_file}"):
                        model = SentimentModel()
                        try:
                            model.load(model_path)
                            st.session_state.current_model = model
                            st.success(f"Model '{model_file}' loaded successfully!")
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
                
                with col2:
                    if st.button(f"Download", key=f"down_{model_file}"):
                        try:
                            with open(model_path, 'rb') as f:
                                st.download_button(
                                    label="Download Model",
                                    data=f.read(),
                                    file_name=model_file,
                                    mime="application/octet-stream",
                                    key=f"download_btn_{model_file}"
                                )
                        except Exception as e:
                            st.error(f"Error downloading: {e}")
                
                with col3:
                    if st.button(f"Delete", key=f"del_{model_file}"):
                        try:
                            os.remove(model_path)
                            st.success(f"Model '{model_file}' deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting: {e}")
    else:
        st.info("No models found. Train a model first!")
    
    # Train new model
    st.markdown("<h2 class='sub-header'>🆕 Train New Model</h2>", unsafe_allow_html=True)
    
    # Provide sample dataset
    st.markdown("### Sample Datasets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Sample Dataset 1"):
            df = load_sample_dataset()
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sample_sentiment_data.csv",
                mime="text/csv"
            )
    
    with col2:
        # Generate larger sample dataset
        sample_reviews = [
            "The product quality is excellent and worth every penny",
            "Very disappointed with the poor customer service",
            "Average performance, not bad but not great either",
            "Outstanding value for money, highly recommended",
            "The worst purchase I've ever made, complete waste",
            "Good product but delivery took too long",
            "Excellent features and user-friendly interface",
            "Overpriced for what you actually get",
            "Reliable and durable, good investment",
            "Flimsy construction, broke after a week"
        ]
        sample_sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative',
                           'positive', 'positive', 'negative', 'positive', 'negative']
        
        larger_df = pd.DataFrame({
            'text': sample_reviews * 5,  # Repeat to make larger
            'sentiment': sample_sentiments * 5
        })
        
        if st.button("📥 Download Sample Dataset 2"):
            csv = larger_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="larger_sample_data.csv",
                mime="text/csv"
            )
    
    with col3:
        st.info("Upload your own CSV file below")
    
    uploaded_file = st.file_uploader("Upload training data (CSV)", type=['csv'], key="train_upload")
    
    # Show sample data if no upload
    if uploaded_file is None:
        st.markdown("**Sample Data Preview:**")
        df_preview = load_sample_dataset()
        st.dataframe(df_preview, use_container_width=True)
        df = df_preview
    else:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} samples from uploaded file")
    
    if 'text' in df.columns and 'sentiment' in df.columns:
        with st.form("train_model_form"):
            test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
            
            if st.form_submit_button("🚀 Train Model"):
                with st.spinner("Training model..."):
                    # Prepare data
                    X = df['text'].fillna('').astype(str).tolist()
                    y = df['sentiment'].fillna('neutral').astype(str).tolist()
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Train model
                    model = SentimentModel(model_name=f"model_{datetime.now().strftime('%Y%m%d')}")
                    model.train(X_train, y_train)
                    
                    # Evaluate
                    predictions, confidence = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions, average='weighted')
                    
                    # Save model
                    model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    model_path = os.path.join(MODELS_DIR, model_name)
                    model.save(model_path)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("F1 Score", f"{f1:.2%}")
                    with col3:
                        st.metric("Training Samples", len(X_train))
                    
                    st.session_state.current_model = model
                    log_activity("New model trained", 
                               f"Accuracy: {accuracy:.2%}, Samples: {len(X)}")
                    
                    st.success(f"✅ Model trained and saved as '{model_name}'")
                    
                    # Show classification report
                    with st.expander("📊 Detailed Classification Report"):
                        report = classification_report(y_test, predictions, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_test, predictions)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
    else:
        st.error("Dataset must contain 'text' and 'sentiment' columns")

# ============================================
# ANALYTICS PAGE
# ============================================
def analytics_page():
    st.markdown("<h1 class='main-header'>📈 Advanced Analytics</h1>", unsafe_allow_html=True)
    
    # Load sample data
    df = load_sample_dataset()
    
    # Create interactive dashboard
    st.markdown("<h2 class='sub-header'>Interactive Dashboard</h2>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(datetime.now().date() - timedelta(days=30), datetime.now().date())
        )
    
    with col2:
        selected_aspects = st.multiselect(
            "Select Aspects",
            options=['product', 'quality', 'price', 'service', 'delivery', 'performance', 'design', 'usability'],
            default=['product', 'quality', 'price']
        )
    
    with col3:
        selected_sentiments = st.multiselect(
            "Select Sentiments",
            options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Distribution', 'Trend Over Time',
                       'Aspect Comparison', 'Confidence Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'histogram'}]]
    )
    
    # Pie chart - Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    fig.add_trace(
        go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, 
               marker_colors=['green', 'red', 'blue']),
        row=1, col=1
    )
    
    # Line chart - Simulated trend
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    trend_data = pd.DataFrame({
        'date': dates,
        'positive': np.random.randint(10, 50, 30),
        'negative': np.random.randint(5, 30, 30),
        'neutral': np.random.randint(2, 20, 30)
    })
    
    for sentiment in ['positive', 'negative', 'neutral']:
        fig.add_trace(
            go.Scatter(x=trend_data['date'], y=trend_data[sentiment],
                      mode='lines+markers', name=sentiment.capitalize()),
            row=1, col=2
        )
    
    # Bar chart - Aspect comparison
    # Explode aspects for analysis
    df['aspect_list'] = df['aspect'].str.split(',')
    exploded_df = df.explode('aspect_list')
    aspect_counts = exploded_df.groupby(['aspect_list', 'sentiment']).size().unstack(fill_value=0)
    
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in aspect_counts.columns:
            fig.add_trace(
                go.Bar(x=aspect_counts.index, y=aspect_counts[sentiment], name=sentiment.capitalize()),
                row=2, col=1
            )
    
    # Histogram - Confidence distribution
    confidence_scores = np.random.normal(70, 15, 1000)
    confidence_scores = np.clip(confidence_scores, 0, 100)
    fig.add_trace(
        go.Histogram(x=confidence_scores, nbinsx=20, name='Confidence'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, showlegend=True, title_text="Sentiment Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("<h3>Export Analytics</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Charts as PNG"):
            st.info("Chart export would save as PNG file")
    
    with col2:
        # Create sample data for export
        sample_data = pd.DataFrame({
            'Date': dates,
            'Positive_Count': trend_data['positive'],
            'Negative_Count': trend_data['negative'],
            'Neutral_Count': trend_data['neutral']
        })
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📈 Export Data as CSV",
            data=csv,
            file_name="sentiment_analytics.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("📋 Generate Report"):
            # Generate a sample report
            report_text = f"""
            # Sentiment Analysis Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ## Summary Statistics
            - Total Samples: {len(df)}
            - Positive Sentiments: {len(df[df['sentiment'] == 'positive'])}
            - Negative Sentiments: {len(df[df['sentiment'] == 'negative'])}
            - Neutral Sentiments: {len(df[df['sentiment'] == 'neutral'])}
            
            ## Aspect Analysis
            {exploded_df['aspect_list'].value_counts().to_string()}
            
            ## Recommendations
            1. Focus on improving aspects with negative sentiment
            2. Consider collecting more data for better insights
            3. Monitor sentiment trends regularly
            """
            
            st.text_area("Generated Report", report_text, height=300)
            st.success("Report generated successfully!")

# ============================================
# SETTINGS PAGE
# ============================================
def settings_page():
    st.markdown("<h1 class='main-header'>⚙️ User Settings</h1>", unsafe_allow_html=True)
    
    current_user = st.session_state.current_user
    user_data = st.session_state.users[current_user]
    
    with st.form("user_settings"):
        st.write(f"**Username:** {current_user}")
        st.write(f"**Role:** {user_data['role']}")
        
        new_email = st.text_input("Email Address", value=user_data.get('email', ''))
        
        # Password change
        st.markdown("### Change Password")
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("💾 Save Changes"):
            # Update email
            if new_email != user_data.get('email', ''):
                st.session_state.users[current_user]['email'] = new_email
                save_users()
                st.success("Email updated successfully!")
            
            # Update password if provided
            if new_password:
                if new_password != confirm_password:
                    st.error("New passwords don't match!")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    # Verify current password
                    current_hash = hashlib.sha256(current_password.encode()).hexdigest()
                    if current_hash == user_data['password']:
                        new_hash = hashlib.sha256(new_password.encode()).hexdigest()
                        st.session_state.users[current_user]['password'] = new_hash
                        save_users()
                        log_activity("Password changed", "User updated their password")
                        st.success("Password updated successfully!")
                    else:
                        st.error("Current password is incorrect")

# ============================================
# AUTHENTICATION PAGES
# ============================================
def auth_page():
    st.markdown("<h1 class='main-header'>🔐 Sentiment Analyzer System</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if login(username, password):
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Register")
            
            if submit_register:
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = register(new_username, new_password, new_email)
                    if success:
                        st.success(message)
                        st.info("Please login with your new credentials")
                    else:
                        st.error(message)

# ============================================
# QUERY & FEEDBACK PAGE
# ============================================
def query_feedback_page():
    st.markdown("<h1 class='main-header'>🔍 Query Search & Feedback</h1>", unsafe_allow_html=True)
    
    tabs = st.tabs(["🔎 Sentiment Analysis", "📝 Feedback Review", "📊 Feedback Analytics", "📋 Query History"])
    
    with tabs[0]:  # Sentiment Analysis
        st.markdown("<h2 class='sub-header'>Analyze Text Sentiment</h2>", unsafe_allow_html=True)
        
        # Input options
        input_method = st.radio("Input Method:", ["Single Text", "Batch Upload"])
        
        if input_method == "Single Text":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query_text = st.text_area(
                    "Enter text to analyze:",
                    placeholder="Enter your text here...\nExample: 'The product quality is excellent but delivery was slow.'",
                    height=100
                )
            
            with col2:
                st.write("")
                st.write("")
                analyze_button = st.button("🚀 Analyze Sentiment", type="primary")
            
            if analyze_button and query_text:
                with st.spinner("Analyzing sentiment..."):
                    if st.session_state.current_model is None:
                        # Create a demo model if none exists
                        st.session_state.current_model = SentimentModel(model_name="demo_model")
                        # Train with sample data
                        df = load_sample_dataset()
                        X = df['text'].tolist()
                        y = df['sentiment'].tolist()
                        st.session_state.current_model.train(X, y)
                    
                    # Analyze the text
                    result = st.session_state.current_model.predict_with_details(query_text)
                    
                    # Display results
                    st.markdown("<div class='query-result'>", unsafe_allow_html=True)
                    st.markdown(f"**📝 Query:** {query_text}")
                    
                    # Sentiment with color coding
                    sentiment_class = f"sentiment-{result['predicted_sentiment']}"
                    sentiment_html = f"<span class='{sentiment_class}'>📊 Predicted Sentiment: {result['predicted_sentiment'].upper()}</span>"
                    st.markdown(sentiment_html, unsafe_allow_html=True)
                    
                    # Confidence with color coding
                    confidence = result['confidence']
                    if confidence >= 70:
                        conf_class = "confidence-high"
                    elif confidence >= 50:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    
                    confidence_html = f"<span class='{conf_class}'>🎯 Confidence: {confidence:.1f}%</span>"
                    st.markdown(confidence_html, unsafe_allow_html=True)
                    
                    # Aspects
                    if result['aspects']:
                        st.markdown(f"**🏷️ Detected Aspects:** {', '.join(result['aspects'])}")
                    
                    # Probability distribution
                    st.markdown("**📈 Probability Distribution:**")
                    probs_df = pd.DataFrame({
                        'Sentiment': list(result['probabilities'].keys()),
                        'Probability (%)': list(result['probabilities'].values())
                    })
                    st.dataframe(probs_df, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Feedback form
                    st.markdown("<h3>✏️ Provide Feedback</h3>", unsafe_allow_html=True)
                    
                    with st.form(f"feedback_form_{hash(query_text)}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            corrected_sentiment = st.selectbox(
                                "Correct Sentiment (if wrong):",
                                ["keep_prediction", "positive", "negative", "neutral"],
                                format_func=lambda x: "Keep Prediction" if x == "keep_prediction" else x.capitalize()
                            )
                        
                        with col2:
                            additional_aspects = st.multiselect(
                                "Add Missing Aspects:",
                                ["product", "quality", "price", "service", "delivery", 
                                 "performance", "design", "usability"]
                            )
                        
                        user_feedback = st.text_area(
                            "Additional Feedback/Comments:",
                            placeholder="Any additional comments or suggestions..."
                        )
                        
                        submit_feedback = st.form_submit_button("💾 Save Feedback")
                        
                        if submit_feedback:
                            # Combine aspects
                            all_aspects = list(set(result['aspects'] + additional_aspects))
                            
                            # Save feedback
                            if corrected_sentiment == "keep_prediction":
                                corrected = None
                            else:
                                corrected = corrected_sentiment
                            
                            save_feedback_entry(
                                query=query_text,
                                predicted_sentiment=result['predicted_sentiment'],
                                corrected_sentiment=corrected,
                                user_feedback=user_feedback,
                                aspects=all_aspects,
                                confidence=result['confidence']
                            )
                            
                            st.success("✅ Feedback saved successfully!")
                            
                            if corrected_sentiment != "keep_prediction":
                                st.info("⚠️ Correction saved. The model will be updated during next retraining.")
        
        else:  # Batch Upload
            uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                if 'text' in df.columns:
                    st.success(f"✅ Loaded {len(df)} texts")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🔍 Analyze All Texts"):
                        if st.session_state.current_model is None:
                            st.session_state.current_model = SentimentModel(model_name="demo_model")
                            sample_df = load_sample_dataset()
                            X_sample = sample_df['text'].tolist()
                            y_sample = sample_df['sentiment'].tolist()
                            st.session_state.current_model.train(X_sample, y_sample)
                        
                        with st.spinner(f"Analyzing {len(df)} texts..."):
                            results = []
                            for text in df['text'].head(20):  # Limit to first 20 for demo
                                result = st.session_state.current_model.predict_with_details(str(text))
                                results.append({
                                    'text': text[:100] + "..." if len(str(text)) > 100 else text,
                                    'sentiment': result['predicted_sentiment'],
                                    'confidence': result['confidence'],
                                    'aspects': ', '.join(result['aspects'])
                                })
                            
                            results_df = pd.DataFrame(results)
                            st.markdown("<h3>📊 Analysis Results</h3>", unsafe_allow_html=True)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("CSV file must contain a 'text' column")
            else:
                st.info("👆 Upload a CSV file or use single text input")
    
    with tabs[1]:  # Feedback Review
        st.markdown("<h2 class='sub-header'>📝 Review & Manage Feedback</h2>", unsafe_allow_html=True)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Filter by:",
                ["All Feedback", "Only Corrections", "Only Uncorrected"]
            )
        
        with col2:
            filter_sentiment = st.selectbox(
                "Filter by Sentiment:",
                ["All", "positive", "negative", "neutral"]
            )
        
        with col3:
            all_users = set([f.get('user', 'Unknown') for f in st.session_state.feedback_data['queries']])
            filter_user = st.selectbox(
                "Filter by User:",
                ["All"] + list(all_users)
            )
        
        # Apply filters
        feedback_list = st.session_state.feedback_data['queries']
        
        if filter_type == "Only Corrections":
            feedback_list = [f for f in feedback_list if f['is_corrected']]
        elif filter_type == "Only Uncorrected":
            feedback_list = [f for f in feedback_list if not f['is_corrected']]
        
        if filter_sentiment != "All":
            feedback_list = [f for f in feedback_list if f['predicted_sentiment'] == filter_sentiment]
        
        if filter_user != "All":
            feedback_list = [f for f in feedback_list if f.get('user') == filter_user]
        
        # Display feedback
        st.write(f"**Total Feedback Items:** {len(feedback_list)}")
        
        if feedback_list:
            for i, feedback in enumerate(reversed(feedback_list[-20:])):  # Show last 20
                with st.expander(f"📝 Feedback {i+1}: {feedback['query'][:50]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Query:** {feedback['query']}")
                        st.write(f"**Predicted:** {feedback['predicted_sentiment']}")
                        st.write(f"**Confidence:** {feedback.get('confidence', 'N/A')}%")
                        st.write(f"**Aspects:** {', '.join(feedback.get('aspects', []))}")
                    
                    with col2:
                        st.write(f"**User:** {feedback.get('user', 'Unknown')}")
                        st.write(f"**Time:** {feedback['timestamp']}")
                        if feedback['is_corrected']:
                            st.success(f"**Corrected to:** {feedback['corrected_sentiment']}")
                        else:
                            st.info("**No correction made**")
                        
                        if feedback.get('user_feedback'):
                            st.write(f"**Comments:** {feedback['user_feedback']}")
                    
                    # Action buttons
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button(f"Use for Training", key=f"train_{feedback['id']}_{i}"):
                            st.info("This feedback will be included in next training cycle")
                    with col_b:
                        if st.button(f"Delete", key=f"delete_{feedback['id']}_{i}"):
                            # Remove from feedback data
                            st.session_state.feedback_data['queries'] = [
                                f for f in st.session_state.feedback_data['queries'] 
                                if f['id'] != feedback['id']
                            ]
                            save_feedback()
                            st.success("Feedback deleted!")
                            st.rerun()
        else:
            st.info("No feedback found with current filters.")
    
    with tabs[2]:  # Feedback Analytics
        st.markdown("<h2 class='sub-header'>📊 Feedback Analytics</h2>", unsafe_allow_html=True)
        
        stats = get_feedback_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", stats['total_queries'])
        
        with col2:
            st.metric("Total Corrections", stats['total_corrections'])
        
        with col3:
            st.metric("Correction Rate", f"{stats['correction_rate']:.1f}%")
        
        with col4:
            st.metric("Active Users", len(set([f.get('user', 'Unknown') 
                                              for f in st.session_state.feedback_data['queries']])))
        
        # Visualizations
        if stats['total_queries'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution chart
                sentiments_df = pd.DataFrame({
                    'Sentiment': list(stats['sentiment_distribution'].keys()),
                    'Count': list(stats['sentiment_distribution'].values())
                })
                
                fig1 = px.pie(sentiments_df, values='Count', names='Sentiment',
                             title='Sentiment Distribution in Queries',
                             color='Sentiment',
                             color_discrete_map={'positive': 'green', 
                                               'negative': 'red', 
                                               'neutral': 'blue'})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Create simulated time series data for demonstration
                if stats['total_queries'] > 0:
                    dates = pd.date_range(end=datetime.now(), periods=min(30, stats['total_queries']), freq='D')
                    daily_counts = np.random.randint(1, 10, len(dates))
                    daily_df = pd.DataFrame({'Date': dates, 'Count': daily_counts})
                    
                    fig2 = px.line(daily_df, x='Date', y='Count',
                                  title='Daily Query Volume (Last 30 Days)',
                                  markers=True)
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Export feedback data
            st.markdown("### 📥 Export Feedback Data")
            if st.button("Export All Feedback as CSV"):
                feedback_df = pd.DataFrame(st.session_state.feedback_data['queries'])
                csv = feedback_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="feedback_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No feedback data available for analytics.")
    
    with tabs[3]:  # Query History
        st.markdown("<h2 class='sub-header'>📋 Recent Query History</h2>", unsafe_allow_html=True)
        
        if st.session_state.query_history:
            # Display recent queries
            history_df = pd.DataFrame(st.session_state.query_history[-50:])  # Last 50 queries
            
            # Convert timestamp
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['time'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                history_df[['query', 'prediction', 'time']].sort_values('time', ascending=False),
                use_container_width=True
            )
            
            # Search in history
            st.markdown("### 🔍 Search in Query History")
            search_term = st.text_input("Search queries:", key="search_history")
            
            if search_term:
                filtered = history_df[history_df['query'].str.contains(search_term, case=False, na=False)]
                st.write(f"Found {len(filtered)} matching queries:")
                st.dataframe(filtered[['query', 'prediction', 'time']], use_container_width=True)
        else:
            st.info("No query history available.")

# ============================================
# ACTIVE LEARNING PAGE
# ============================================
def active_learning_page():
    st.markdown("<h1 class='main-header'>🤖 Active Learning Module</h1>", unsafe_allow_html=True)
    
    st.info("""
    **Active Learning Workflow:**
    1. Upload your dataset
    2. Model predicts sentiment with confidence scores
    3. Low confidence samples (<50%) are shown for review
    4. You correct the predictions
    5. Model retrains with your corrections
    6. Model becomes smarter over time
    """)
    
    # Check if we have feedback data for training
    if len(st.session_state.feedback_data['corrections']) > 0:
        st.success(f"✅ You have {len(st.session_state.feedback_data['corrections'])} corrections available for training!")
        
        if st.button("🔄 Retrain Model with Feedback"):
            with st.spinner("Retraining model with feedback corrections..."):
                # Prepare training data from feedback
                training_data = []
                for correction in st.session_state.feedback_data['corrections']:
                    training_data.append({
                        'text': correction['query'],
                        'sentiment': correction['corrected_sentiment']
                    })
                
                # Add sample data if needed
                if len(training_data) < 10:
                    df_sample = load_sample_dataset()
                    for _, row in df_sample.iterrows():
                        training_data.append({
                            'text': row['text'],
                            'sentiment': row['sentiment']
                        })
                
                # Convert to DataFrame
                train_df = pd.DataFrame(training_data)
                
                # Train model
                if st.session_state.current_model is None:
                    st.session_state.current_model = SentimentModel(model_name="feedback_trained")
                
                X = train_df['text'].tolist()
                y = train_df['sentiment'].tolist()
                
                st.session_state.current_model.train(X, y)
                
                # Save model
                model_name = f"model_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                model_path = os.path.join(MODELS_DIR, model_name)
                st.session_state.current_model.save(model_path)
                
                log_activity("Model retrained with feedback", 
                           f"Using {len(training_data)} samples including {len(st.session_state.feedback_data['corrections'])} corrections")
                
                st.success(f"✅ Model retrained successfully with {len(training_data)} samples!")
                st.rerun()
    
    # File upload for traditional active learning
    st.markdown("<h2 class='sub-header'>📤 Upload Dataset for Active Learning</h2>", unsafe_allow_html=True)
    
    # Provide sample dataset
    st.markdown("### Sample Dataset")
    sample_csv = generate_sample_csv()
    st.download_button(
        label="📥 Download Sample Dataset",
        data=sample_csv,
        file_name="active_learning_sample.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="al_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded: {len(df)} samples")
        
        # Check required columns
        if 'text' not in df.columns:
            st.error("Dataset must contain 'text' column")
            return
        
        # Initialize or load model
        if st.session_state.current_model is None:
            st.session_state.current_model = SentimentModel()
            # Train with sample data if not trained
            if not st.session_state.current_model.is_trained:
                sample_df = load_sample_dataset()
                X_sample = sample_df['text'].tolist()
                y_sample = sample_df['sentiment'].tolist()
                st.session_state.current_model.train(X_sample, y_sample)
        
        # Make predictions on a subset (for demo)
        sample_size = min(20, len(df))
        df_sample = df.head(sample_size).copy()
        
        # Make predictions
        with st.spinner("Making predictions..."):
            results = []
            for text in df_sample['text']:
                result = st.session_state.current_model.predict_with_details(str(text))
                results.append({
                    'text': text,
                    'predicted_sentiment': result['predicted_sentiment'],
                    'confidence': result['confidence'],
                    'aspects': result['aspects']
                })
        
        results_df = pd.DataFrame(results)
        
        # Show uncertain samples
        st.markdown("<h2 class='sub-header'>❓ Uncertain Samples for Review</h2>", unsafe_allow_html=True)
        
        uncertain_mask = results_df['confidence'] < 50
        uncertain_df = results_df[uncertain_mask].copy()
        
        if len(uncertain_df) > 0:
            st.warning(f"Found {len(uncertain_df)} uncertain samples (confidence < 50%)")
            
            # Create editable table for uncertain samples
            corrections = []
            for idx, row in uncertain_df.iterrows():
                with st.container():
                    st.markdown(f"<div class='uncertain-sample'>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Text:** {row['text']}")
                        st.write(f"**Predicted:** {row['predicted_sentiment']}")
                        st.write(f"**Confidence:** {row['confidence']:.1f}%")
                        st.write(f"**Aspects:** {', '.join(row['aspects'])}")
                    
                    with col2:
                        correction = st.selectbox(
                            f"Correct sentiment:",
                            ["positive", "negative", "neutral"],
                            key=f"al_corr_{idx}",
                            index=["positive", "negative", "neutral"].index(row['predicted_sentiment']) 
                                  if row['predicted_sentiment'] in ["positive", "negative", "neutral"] else 0
                        )
                        corrections.append((row['text'], correction, row['predicted_sentiment'], row['aspects'], row['confidence']))
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("💾 Save All Corrections", key="save_al_corrections"):
                for text, correction, original, aspects, confidence in corrections:
                    save_feedback_entry(
                        query=text,
                        predicted_sentiment=original,
                        corrected_sentiment=correction,
                        aspects=aspects,
                        confidence=confidence
                    )
                
                st.success(f"✅ {len(corrections)} corrections saved to feedback!")
                st.info("Go to 'Model Management' to retrain with these corrections.")
        else:
            st.success("🎉 No uncertain samples found! Model is confident in all predictions.")
        
        # Show all predictions
        st.markdown("<h2 class='sub-header'>📊 All Predictions</h2>", unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)
    
    else:
        st.info("👆 Upload a CSV file OR use existing feedback corrections above")

# ============================================
# DASHBOARD PAGE
# ============================================
def dashboard_page():
    st.markdown("<h1 class='main-header'>📊 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Load sample data for demo
    df = load_sample_dataset()
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Reviews", len(df))
    
    with col2:
        pos_count = len(df[df['sentiment'] == 'positive'])
        st.metric("Positive", pos_count)
    
    with col3:
        neg_count = len(df[df['sentiment'] == 'negative'])
        st.metric("Negative", neg_count)
    
    with col4:
        neu_count = len(df[df['sentiment'] == 'neutral'])
        st.metric("Neutral", neu_count)
    
    with col5:
        stats = get_feedback_stats()
        st.metric("User Queries", stats['total_queries'])
    
    # Quick query box
    st.markdown("<h3>🔍 Quick Sentiment Check</h3>", unsafe_allow_html=True)
    quick_query = st.text_input("Enter text for quick analysis:", 
                                placeholder="Type here and press Enter...",
                                key="quick_query")
    
    if quick_query:
        if st.session_state.current_model is None:
            # Create and train a demo model
            st.session_state.current_model = SentimentModel(model_name="demo_model")
            sample_df = load_sample_dataset()
            X_sample = sample_df['text'].tolist()
            y_sample = sample_df['sentiment'].tolist()
            st.session_state.current_model.train(X_sample, y_sample)
        
        result = st.session_state.current_model.predict_with_details(quick_query)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment", result['predicted_sentiment'].upper())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with col3:
            st.metric("Aspects", len(result['aspects']))
    
    # Visualization
    st.markdown("<h2 class='sub-header'>📈 Sentiment Trends</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Distribution", "Trend Over Time", "Aspect Analysis"])
    
    with tab1:
        fig1 = px.pie(df, names='sentiment', title='Sentiment Distribution',
                     color='sentiment', color_discrete_map={'positive': 'green', 
                                                          'negative': 'red', 
                                                          'neutral': 'blue'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Create dates for the sample data
        dates = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        df_with_dates = df.copy()
        df_with_dates['date'] = dates
        
        df_trend = df_with_dates.groupby([df_with_dates['date'].dt.date, 'sentiment']).size().reset_index(name='count')
        fig2 = px.line(df_trend, x='date', y='count', color='sentiment',
                      title='Sentiment Trends Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Explode aspects
        df['aspect_list'] = df['aspect'].str.split(',')
        exploded_df = df.explode('aspect_list')
        fig3 = px.bar(exploded_df, x='aspect_list', color='sentiment', barmode='group',
                     title='Sentiment by Aspect')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Data table
    st.markdown("<h2 class='sub-header'>📋 Recent Reviews</h2>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

# ============================================
# MAIN DASHBOARD NAVIGATION
# ============================================
def main_dashboard():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {st.session_state.current_user}")
        
        if st.session_state.users[st.session_state.current_user]['role'] == 'admin':
            st.markdown("**Admin Privileges** ✅")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Query & Feedback", "🤖 Active Learning", 
             "📁 Workspaces", "🧠 Model Management", "👥 Admin Panel", 
             "📈 Analytics", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.current_model and st.session_state.current_model.is_trained:
            st.info(f"🤖 Model: {st.session_state.current_model.model_name}")
        
        stats = get_feedback_stats()
        st.caption(f"📝 Queries: {stats['total_queries']}")
        st.caption(f"✏️ Corrections: {stats['total_corrections']}")
        
        st.markdown("---")
        
        if st.button("🚪 Logout"):
            log_activity("User logged out", f"Username: {st.session_state.current_user}")
            st.session_state.current_user = None
            st.rerun()
    
    # Page routing
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "🔍 Query & Feedback":
        query_feedback_page()
    elif page == "🤖 Active Learning":
        active_learning_page()
    elif page == "📁 Workspaces":
        workspaces_page()
    elif page == "🧠 Model Management":
        model_management_page()
    elif page == "👥 Admin Panel":
        if st.session_state.users[st.session_state.current_user]['role'] == 'admin':
            admin_panel_page()
        else:
            st.error("⛔ Access Denied: Admin privileges required")
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

# ============================================
# MAIN APPLICATION FLOW
# ============================================
def main():
    # Create directories and files if they don't exist
    create_mock_model_files()
    
    if st.session_state.current_user is None:
        auth_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()