import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set the stock symbol and the time period for fetching data
stock_symbol = 'HCC.NS'  
start_date = '2020-01-01'
end_date = '2024-11-01'

# Fetch historical stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Prepare the data
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.timestamp)  # Convert to timestamp

# Features and target variable
X = data[['Date']]  # Features (in this case, just the date)
y = data['Close']   # Target variable (closing price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred, color='red', label='Predicted Prices')
plt.plot(data['Date'], data['Close'], color='green', label='Historical Prices')
plt.title(f'Stock Price Prediction for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()

# Print the model's coefficients
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')