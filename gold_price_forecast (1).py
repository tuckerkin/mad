
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load data
@st.cache
def load_data():
    gold_data = pd.read_csv('gold_prices.csv', parse_dates=['Date'], index_col='Date')
    inflation_data = pd.read_csv('inflation.csv', parse_dates=['Date'], index_col='Date')
    interest_rate_data = pd.read_csv('interest_rates.csv', parse_dates=['Date'], index_col='Date')
    data = gold_data.join([inflation_data, interest_rate_data], how='inner')
    return data

data = load_data()

# Sidebar for user input
st.sidebar.header('User Input')
years = st.sidebar.slider('Select years for training data', 1, 10, 5)

# Show raw data
st.subheader('Historical Gold Prices and Economic Data')
st.write(data.tail())

# Preprocessing
data['Gold_Lag_1d'] = data['Gold_Price'].shift(1)
data.dropna(inplace=True)

# Features and target
X = data[['Gold_Lag_1d', 'Inflation_Rate', 'Interest_Rate']]
y = data['Gold_Price']

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Plot actual vs predicted prices
st.subheader('Actual vs Predicted Prices')
fig, ax = plt.subplots()
ax.plot(data.index, y, label='Actual Price', color='blue')
ax.plot(data.index, predictions, label='Predicted Price', color='red')
ax.legend()
st.pyplot(fig)

# Forecast future prices (example)
st.subheader('Forecast Future Prices')
future_prices = model.predict(X[-30:])  # Predicting on last 30 days for simplicity
st.write(future_prices)
