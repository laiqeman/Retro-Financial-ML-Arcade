import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Financial ML Arcade",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #00ff00;
    }
    .stButton>button {
        background-color: #000000;
        color: #00ff00;
        border: 2px solid #00ff00;
        border-radius: 0;
        font-family: 'Courier New', monospace;
    }
    .stButton>button:hover {
        background-color: #1a1a1a;
        border: 2px solid #ff00ff;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #00ff00;
        border: 2px solid #00ff00;
    }
    .stSelectbox>div>div>select {
        background-color: #1a1a1a;
        color: #00ff00;
        border: 2px solid #00ff00;
    }
    .stSlider>div>div>div {
        background-color: #1a1a1a;
    }
    .stSlider>div>div>div>div {
        background-color: #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'loan_data' not in st.session_state:
    st.session_state.loan_data = None

# Welcome Interface
def show_welcome():
    st.title("🎮 Financial ML Arcade")
    st.markdown("""
    Welcome to the Financial ML Arcade! Choose your data source and let's start the machine learning journey.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Upload Kaggle Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Dataset loaded successfully!")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.subheader("📈 Yahoo Finance Data")
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)")
        if symbol:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period="1mo")
                st.session_state.data = data
                st.success(f"Data for {symbol} loaded successfully!")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

# Data Preprocessing
def preprocess_data():
    if st.session_state.data is None:
        st.warning("Please load data first!")
        return
    
    st.subheader("🧹 Data Preprocessing")
    
    # Handle missing values
    missing_values = st.session_state.data.isnull().sum()
    st.write("Missing Values:", missing_values)
    
    # Fill missing values
    if st.button("Clean Missing Values"):
        st.session_state.data = st.session_state.data.fillna(method='ffill')
        st.success("Missing values cleaned!")
    
    # Remove outliers
    if st.button("Remove Outliers"):
        for column in st.session_state.data.select_dtypes(include=[np.number]).columns:
            Q1 = st.session_state.data[column].quantile(0.25)
            Q3 = st.session_state.data[column].quantile(0.75)
            IQR = Q3 - Q1
            st.session_state.data = st.session_state.data[
                (st.session_state.data[column] >= Q1 - 1.5 * IQR) &
                (st.session_state.data[column] <= Q3 + 1.5 * IQR)
            ]
        st.success("Outliers removed!")

# Feature Engineering
def feature_engineering():
    if st.session_state.data is None:
        st.warning("Please load data first!")
        return
    
    st.subheader("⚙️ Feature Engineering")
    
    # Select features
    numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns
    selected_features = st.multiselect("Select Features", numeric_columns)
    
    if selected_features:
        X = st.session_state.data[selected_features]
        y = st.session_state.data['Close'] if 'Close' in st.session_state.data.columns else st.session_state.data.iloc[:, -1]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.session_state.X = X_scaled
        st.session_state.y = y
        st.success("Features selected and scaled!")

# Model Training
def train_model():
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("Please complete feature engineering first!")
        return
    
    st.subheader("🎯 Model Training")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.X, st.session_state.y, test_size=0.2, random_state=42
    )
    
    # Train model
    if st.button("Train Linear Regression Model"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.session_state.model = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        st.session_state.predictions = y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.success("Model trained successfully!")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name='Actual', line=dict(color='#00ff00')))
        fig.add_trace(go.Scatter(y=y_pred, name='Predicted', line=dict(color='#ff00ff')))
        fig.update_layout(
            title="Actual vs Predicted Values",
            template="plotly_dark",
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a'
        )
        st.plotly_chart(fig)

# Loan Calculator
def loan_calculator():
    st.title("💰 Smart Loan Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        interest_rate = st.slider("Annual Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.0, step=0.1)
        loan_term = st.slider("Loan Term (Years)", min_value=1, max_value=30, value=5)
        
        # Calculate monthly payment
        monthly_rate = interest_rate / 100 / 12
        num_payments = loan_term * 12
        monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        st.metric("Monthly Payment", f"${monthly_payment:.2f}")
        st.metric("Total Interest", f"${(monthly_payment * num_payments - loan_amount):.2f}")
        st.metric("Total Payment", f"${(monthly_payment * num_payments):.2f}")
    
    with col2:
        st.subheader("Payment Schedule")
        # Create payment schedule
        schedule = []
        balance = loan_amount
        
        for month in range(1, num_payments + 1):
            interest_payment = balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
            balance -= principal_payment
            
            schedule.append({
                'Month': month,
                'Payment': monthly_payment,
                'Principal': principal_payment,
                'Interest': interest_payment,
                'Balance': max(0, balance)
            })
        
        schedule_df = pd.DataFrame(schedule)
        st.dataframe(schedule_df.style.format({
            'Payment': '${:.2f}',
            'Principal': '${:.2f}',
            'Interest': '${:.2f}',
            'Balance': '${:.2f}'
        }))
        
        # Plot payment breakdown
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Principal', 'Interest'],
            y=[loan_amount, monthly_payment * num_payments - loan_amount],
            marker_color=['#00ff00', '#ff00ff']
        ))
        fig.update_layout(
            title="Payment Breakdown",
            template="plotly_dark",
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a'
        )
        st.plotly_chart(fig)

# Smart Financial System
def smart_financial_system():
    st.title("🧠 Smart Financial System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income & Expenses")
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        
        st.subheader("Fixed Expenses")
        rent = st.number_input("Rent/Mortgage", min_value=0, value=1500)
        utilities = st.number_input("Utilities", min_value=0, value=200)
        insurance = st.number_input("Insurance", min_value=0, value=300)
        
        st.subheader("Variable Expenses")
        groceries = st.number_input("Groceries", min_value=0, value=400)
        transportation = st.number_input("Transportation", min_value=0, value=300)
        entertainment = st.number_input("Entertainment", min_value=0, value=200)
        
        total_expenses = rent + utilities + insurance + groceries + transportation + entertainment
        savings = monthly_income - total_expenses
        
        st.metric("Total Expenses", f"${total_expenses}")
        st.metric("Monthly Savings", f"${savings}")
        
        # Calculate savings rate
        savings_rate = (savings / monthly_income) * 100
        st.metric("Savings Rate", f"{savings_rate:.1f}%")
    
    with col2:
        st.subheader("Financial Health Analysis")
        
        # Create expense breakdown
        expenses = {
            'Rent/Mortgage': rent,
            'Utilities': utilities,
            'Insurance': insurance,
            'Groceries': groceries,
            'Transportation': transportation,
            'Entertainment': entertainment
        }
        
        # Plot expense breakdown
        fig = go.Figure(data=[go.Pie(
            labels=list(expenses.keys()),
            values=list(expenses.values()),
            hole=.3,
            marker_colors=['#00ff00', '#ff00ff', '#00ffff', '#ffff00', '#ff0000', '#0000ff']
        )])
        fig.update_layout(
            title="Expense Breakdown",
            template="plotly_dark",
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a'
        )
        st.plotly_chart(fig)
        
        # Financial health recommendations
        st.subheader("Recommendations")
        if savings_rate < 20:
            st.warning("⚠️ Your savings rate is below the recommended 20%. Consider reducing expenses.")
        if rent > monthly_income * 0.3:
            st.warning("⚠️ Your housing costs exceed 30% of your income. Consider finding more affordable housing.")
        if entertainment > monthly_income * 0.1:
            st.warning("⚠️ Your entertainment expenses are high. Consider setting a stricter budget.")
        
        if savings_rate >= 20:
            st.success("✅ Great job! Your savings rate is healthy.")
        if rent <= monthly_income * 0.3:
            st.success("✅ Your housing costs are within a healthy range.")
        if entertainment <= monthly_income * 0.1:
            st.success("✅ Your entertainment expenses are well-managed.")

# Main App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Welcome",
        "Loan Calculator",
        "Smart Financial System",
        "Preprocessing",
        "Feature Engineering",
        "Model Training"
    ])
    
    if page == "Welcome":
        show_welcome()
    elif page == "Loan Calculator":
        loan_calculator()
    elif page == "Smart Financial System":
        smart_financial_system()
    elif page == "Preprocessing":
        preprocess_data()
    elif page == "Feature Engineering":
        feature_engineering()
    elif page == "Model Training":
        train_model()

if __name__ == "__main__":
    main() 