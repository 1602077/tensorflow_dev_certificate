import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_train_test_datasets(filename, log=False):
    """
    Create train and test datasets using a 80:20 split on BTC price data.

    params:
        filename (str)  : filename of csv containing BTC price data to be time forecasted
        log      (bool) : if true performs basic logging and plotting of train / test sets
    """

    df = pd.read_csv(f"../data/{filename}", parse_dates=["Date"], index_col=["Date"])
    df = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})

    timesteps = df.index.to_numpy()
    prices = df["Price"].to_numpy()

    # create train and test splits
    split_size = int(0.8 * len(prices))  # 80/20 split
    X_train, y_train = timesteps[:split_size], prices[:split_size]
    X_test, y_test = timesteps[split_size:], prices[split_size:]

    if log:
        print(f"Training set length: {len(X_train)}.\nTest set length: {len(X_test)}.")
        plt.figure(figsize=(10,7))
        plot_time_series(X_train, y_train, label="train data")
        plot_time_series(X_test, y_test, label="test data")
        plt.savefig("../logs/train_test_set0.png", bbox_inches="tight", dpi=250)


    return X_train, y_train, X_test, y_test


def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """
    Plots timesteps against values.

    params:
        timestesp: array of timestep values
        values:    array of values across timesteps
        format:    style of plot
        start:     where to start the plot
        end:       where to end the plot
        label:     label to show on plot about values
    """
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend()
    plt.grid(False)
    return


if __name__ == "__main__":
    filename = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
    create_train_test_datasets(filename, log=True)

