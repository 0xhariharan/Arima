import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima

def load_data(ticker_symbol):
    # Fetch the stock data using yfinance
    stock_data = yf.Ticker(ticker_symbol)
    stock_history = stock_data.history(start="2001-01-01", actions=False)[["Open", "High", "Low", "Close"]]
    
    # Convert datetime index to date datatype
    stock_history.index = pd.to_datetime(stock_history.index).date
    stock_history['Date'] = stock_history.index
        
    final_df = stock_history[["Close"]]
    
    # Set index and column names to None
    final_df.index.name = None
    final_df.columns.name = None
    
    return final_df

def main():
    st.title('Stock Price Forecasting with ARIMA Model')

    # Text input for stock ticker symbol
    ticker_symbol = st.text_input('Enter Ticker Symbol (e.g., TCS.NS, AAPL)', 'TCS.NS')

    if ticker_symbol:
        final_df = load_data(ticker_symbol)

        # Split the data into training and testing sets
        n_train = len(final_df) - 90  # Use last 90 days as test data
        train = final_df["Close"][:n_train]
        test = final_df["Close"][n_train:]

        # Automatically determine the optimal p, d, q parameters using auto_arima
        st.write("Finding the optimal ARIMA model...")
        model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
        
        # Fit the model on the training data
        model_fit = model.fit(train)
        
        # Make predictions on the test set
        predictions = model_fit.predict(n_periods=len(test))

        # Calculate MAPE (Mean Absolute Percentage Error) for the model performance
        mape = np.mean(np.abs((test - predictions) / test)) * 100
        st.write(f"Mean Absolute Percentage Error (MAPE) on the test set: {mape:.2f}%")

        # Forecast future prices (next 90 days)
        forecast = model_fit.predict(n_periods=90)
        forecast_dates = pd.date_range(final_df.index[-1], periods=91)[1:]
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
        
        # Set index and column names to None for forecast_df
        forecast_df.index.name = None
        forecast_df.columns.name = None

        # Plot the actual data, predictions, and forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(final_df.index, final_df["Close"], label='Actual')
        ax.plot(test.index, predictions, label='Predictions', color='red')
        ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='green')
        ax.legend()

        st.pyplot(fig)

        # Display the top 10 forecasted prices
        st.write("Predictions:")    
        st.write(forecast_df[['Date', 'Forecast']].head(10).set_index('Date').round(2))  # Display only 'Date' and 'Forecast' columns

if __name__ == '__main__':
    main()
