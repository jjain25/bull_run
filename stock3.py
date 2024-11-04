import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set the title of the app
st.title('Stock Price Prediction App')

# Input for stock symbol
stock_symbol = st.text_input('Enter Stock Symbol:', 'HCC.NS')

# Input for date range
start_date = st.date_input('Start Date', pd.to_datetime('2021-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-11-01'))

# Input for alpha value for Exponential Smoothing
alpha = st.slider('Select Alpha Value for Exponential Smoothing:', 0.01, 1.0, 0.2)

# Input for forecast period
forecast_periods = st.slider('Select Forecast Periods:', 1, 365, 30)

# Fetch historical stock data
if st.button('Fetch Data'):
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for the given stock symbol.")
    else:
        # Prepare the data
        data['Date'] = data.index
        data['Date'] = pd.to_datetime(data['Date'])

        # Prepare features and target variable for Linear Regression
        data['Date_ordinal'] = data['Date'].map(pd.Timestamp.timestamp)
        X = data[['Date_ordinal']]
        y = data['Close']

        # Split the data into training and testing sets for Linear Regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Create and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate predictions for both training and testing sets for a continuous line
        X_full = np.array(range(int(data['Date_ordinal'].min()), int(data['Date_ordinal'].max()) + 1)).reshape(-1, 1)
        y_pred_full = model.predict(X_full)

        # Convert ordinal values back to dates for plotting
        X_full_dates = pd.to_datetime(X_full.flatten(), unit='s')

        # Exponential Smoothing with forecast
        model_es = ExponentialSmoothing(data['Close'], trend='add', seasonal=None, damped_trend=False)
        model_es_fit = model_es.fit(smoothing_level=alpha)

        # Forecast future values
        forecast = model_es_fit.forecast(steps=forecast_periods)
        
        # Convert forecast to a pandas Series with a date index for plotting
        forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='B')
        forecast_series = pd.Series(forecast, index=forecast_dates)

        # Plotting the results
        fig, axs = plt.subplots(4, 1, figsize=(14, 28))

        # Plot historical and predicted prices (Linear Regression)
        axs[0].plot(data['Date'], data['Close'], color='green', label='Historical Prices', linewidth=2)
        axs[0].plot(X_full_dates, y_pred_full, 'r-', label='Regression Line (Linear Regression)', linewidth=2)
        axs[0].scatter(data['Date'].iloc[len(X_train):], y_test, color='blue', label='Actual Prices', alpha=0.5)
        axs[0].scatter(data['Date'].iloc[len(X_train):], model.predict(X_test), color='red', label='Predicted Prices (Linear Regression)', alpha=0.5)
        axs[0].set_title(f'Stock Price Prediction for {stock_symbol} (Linear Regression)')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Stock Price')
        axs[0].legend()
        axs[0].grid()

        # Plot Exponential Smoothing with Forecast
        axs[1].plot(data['Date'], model_es_fit.fittedvalues, color='orange', label=f'Exponential Smoothing (Alpha={alpha})', linewidth=2)
        axs[1].set_title('Exponential Smoothing of Stock Prices')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Stock Price')
        axs[1].grid()
        axs[1].legend()

        # Plot forecasted values on Exponential Smoothing graph
        axs[2].plot(data['Date'], model_es_fit.fittedvalues, color='orange', label='Fitted Values (Exponential Smoothing)', linewidth=2)
        axs[2].plot(forecast_series.index, forecast_series, color='purple', label='Forecasted Prices', linestyle='--', linewidth=2)
        axs[2].set_title('Exponential Smoothing with Forecasted Prices')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Stock Price')
        axs[2].grid()
        axs[2].legend()

        # Comparison of Historical and Forecasted Prices
        axs[3].plot(data['Date'], data['Close'], color='green', label='Historical Prices', linewidth=2)
        axs[3].plot(data['Date'], model_es_fit.fittedvalues, color='orange', label='Exponential Smoothing', linewidth=2)
        axs[3].plot(forecast_series.index, forecast_series, color='purple', label='Forecasted Prices', linestyle='--', linewidth=2)
        axs[3].set_title('Comparison of Historical and Forecasted Prices')
        axs[3].set_xlabel('Date')
        axs[3].set_ylabel('Stock Price')
        axs[3].legend()
        axs[3].grid()

        # Show the plot in Streamlit
        st.pyplot(fig)

        # Interpretation of Linear Regression Coefficients
        st.write("### Linear Regression Model Interpretation")
        st.write(f"Linear Regression Coefficient (Slope): {model.coef_[0]}")
        st.write(f"Linear Regression Intercept: {model.intercept_}")
        st.write("The coefficient indicates the average change in the stock's closing price with respect to time. A positive value suggests an increasing trend over time, while a negative value indicates a decreasing trend.")

        # Interpretation of Exponential Smoothing
        st.write("### Exponential Smoothing Interpretation")
        st.write(f"Exponential Smoothing Model Summary: {model_es_fit.summary()}")
        st.write("The Exponential Smoothing model uses historical data to smooth out fluctuations and provide a forecast. The alpha parameter controls the smoothing effect. Higher values of alpha give more weight to recent observations, capturing short-term trends, while lower values consider a longer-term trend.")
