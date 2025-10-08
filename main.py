import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers, period="5y"):
    """Fetch historical prices for tickers for given period."""
    df = yf.download(tickers, period=period)
    df = df['Close']
    df = df.dropna()
    return df

def compute_returns(price_df):
    """Compute daily returns (%) and log returns."""
    returns = price_df.pct_change().dropna()
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return returns, log_returns

def main():
    tickers = ["MAHSCOOTER.NS", "BAJAJHLDNG.NS"]

    # Fetch price data
    price_df = fetch_data(tickers, period="5y")
    print("=== Prices (head) ===")
    print(price_df.head())

    # Correlation on raw prices
    corr_prices = price_df.corr()
    print("\n=== Correlation of Prices ===")
    print(corr_prices)

    # Compute returns
    returns, log_returns = compute_returns(price_df)
    print("\n=== Returns (head) ===")
    print(returns.head())

    # Correlation of daily returns
    corr_returns = returns.corr()
    print("\n=== Correlation of Daily Returns ===")
    print(corr_returns)

    # Correlation of log returns
    corr_log = log_returns.corr()
    print("\n=== Correlation of Log Returns ===")
    print(corr_log)

if __name__ == "__main__":
    main()
