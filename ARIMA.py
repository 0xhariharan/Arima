import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima  # Using pmdarima for automatic ARIMA model order selection

def load_data(ticker_symbol):
    stock_data = yf.Ticker(ticker_symbol)
    stock_history = stock_data.history(period="1y", actions=False)[["Open", "High", "Low", "Close"]]
    
    # Convert datetime index to date datatype
    stock_history.index = pd.to_datetime(stock_history.index).date
    stock_history['Date'] = stock_history.index  # No need to convert datetime to date 
    final_df = stock_history[["Close"]]
    
    # Set index and column names to None
    final_df.index.name = None
    final_df.columns.name = None
    
    return final_df

def main():
    st.title('Stock Price Forecasting with ARIMA Model')

    ticker_symbol = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL, MSFT, TCS.NS)')

    if ticker_symbol:
        final_df = load_data(ticker_symbol)

        st.write(f"Showing data for {ticker_symbol}")
        st.write(final_df.tail())

        # Use auto_arima from pmdarima to automatically select the best ARIMA parameters
        model = auto_arima(final_df["Close"], seasonal=False, stepwise=True, trace=True)

        # Split the data into training and testing sets
        n_train = len(final_df) - 90
        train = final_df["Close"][:n_train]
        test = final_df["Close"][n_train:]

        # Make predictions
        predictions = model.predict(n_periods=len(test))
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        st.write(f"Mean Absolute Percentage Error (MAPE) on the test set: {mape:.2f}%")

        # Forecast next 90 days
        forecast = model.predict(n_periods=90)
        forecast_dates = pd.date_range(final_df.index[-1], periods=91)[1:]
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

        # Set index and column names to None for forecast_df
        forecast_df.index.name = None
        forecast_df.columns.name = None

        # Plotting the results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(final_df.index, final_df["Close"], label='Actual')
        ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='green')
        ax.legend()

        st.pyplot(fig)

        st.write("Forecast for next 90 days:")
        st.write(forecast_df[['Date', 'Forecast']].head(10).set_index('Date').round(2))

if __name__ == '__main__':
    main()
