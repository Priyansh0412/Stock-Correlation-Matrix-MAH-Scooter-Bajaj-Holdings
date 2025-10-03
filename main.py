import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data(tickers, period="5y"):
    """Fetch historical prices for tickers for given period."""
    df = yf.download(tickers, period=period)
    # New yfinance returns adjusted Close as Close by default
    df = df['Close']
    df = df.dropna()
    return df

def compute_returns(price_df):
    """Compute daily returns (%) and log returns."""
    returns = price_df.pct_change().dropna()
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return returns, log_returns

def plot_price_series(price_df, tickers):
    plt.figure(figsize=(12, 6))
    for t in tickers:
        plt.plot(price_df.index, price_df[t], label=t)
    plt.title("Price Series (5 Years)")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heatmap(corr_df, title="Correlation Heatmap"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(returns_df):
    pd.plotting.scatter_matrix(returns_df, diagonal='hist',
                               figsize=(8, 8), alpha=0.5)
    plt.suptitle("Scatter Matrix of Returns")
    plt.tight_layout()
    plt.show()

def main():
    tickers = ["MAHSCOOTER.NS", "BAJAJHLDNG.NS"]

    # Fetch price data
    price_df = fetch_data(tickers, period="5y")
    print("Prices (head):")
    print(price_df.head())

    # Correlation on raw prices
    corr_prices = price_df.corr()
    print("\nCorrelation of Prices:")
    print(corr_prices)

    # Plot price series
    plot_price_series(price_df, tickers)

    # Compute returns
    returns, log_returns = compute_returns(price_df)
    print("\nReturns (head):")
    print(returns.head())

    # Correlation of daily returns
    corr_returns = returns.corr()
    print("\nCorrelation of Daily Returns:")
    print(corr_returns)

    # Correlation of log returns
    corr_log = log_returns.corr()
    print("\nCorrelation of Log Returns:")
    print(corr_log)

    # Heatmaps
    plot_heatmap(corr_prices, title="Price Correlation Heatmap")
    plot_heatmap(corr_returns, title="Daily Returns Correlation Heatmap")

    # Scatter matrix (returns)
    plot_scatter_matrix(returns)

    # Rolling correlation (90-day window)
    window = 90
    rolling_corr = returns[tickers[0]].rolling(window).corr(returns[tickers[1]])
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_corr.index, rolling_corr, label=f"Rolling {window}-day Correlation")
    plt.title(f"{tickers[0]} vs {tickers[1]} Rolling Correlation ({window} days)")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
