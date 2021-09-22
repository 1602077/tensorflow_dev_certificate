import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pprint import pprint


def create_train_test_datasets(filename="BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv", test_split=0.2, log=False, window=False, horizon=1, window_size=7):
    """
    Create train and test datasets using a 80:20 split on BTC price data.

    params:
        filename    (str)  : filename of csv containing BTC price data to be time forecasted
        test_split  (float): size of test set
        log         (bool) : if true performs basic logging and plotting of train / test sets
        window      (bool) : if true window data to transform into a supervised learning problem
        horizon     (int)  : number of days to predict data for
        window_size (int)  : number of days of data used to predict horizon
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

    if window:
        def labelled_windows(x, horizon=horizon):
            """
            Create labels for windows dataset
            """
            return x[:, :-horizon], x[:, -horizon:]


        def make_windows(x, horizon=horizon, window_size=window_size):
            """
            Turns a 1D array into a 2D array of seq labelled windows of window_size with horizon size labels.
            """
            # 1. Create a window of specific window_size + horizon
            window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

            # 2. Create a 2D array of multiple windows steps (minus 1 to account for 0 index)
            window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
            # print(f"Window indexes:\n {window_indexes, window_indexes.shape}")

            # 3. Index on the target array (a time series) with 2D array of multiple window steps
            windowed_array = x[window_indexes]

            # 4. Get labelled windows
            windows, labels = labelled_windows(windowed_array, horizon=horizon)

            return windows, labels

        def make_train_test_split(windows, labels, test_split=test_split):
            """
            Splits matching pairs of windows and labels into train and test sets
            """

            split_size = int(len(windows) * (1-test_split))
            train_windows = windows[:split_size]
            train_labels = labels[:split_size]
            test_windows = windows[split_size:]
            test_labels = labels[split_size:]
            return train_windows, train_labels, test_windows, test_labels

        full_windows, full_labels = make_windows(prices, horizon=horizon, window_size=window_size)
        train_windows, train_labels, test_windows, test_labels = make_train_test_split(full_windows, full_labels, test_split=test_split)

        return train_windows, train_labels, test_windows, test_labels
        

    return X_train, y_train, X_test, y_test


def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """
    Plots timesteps against values.

    params:
        timesteps: array of timestep values
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


def mean_absolute_scaled_error(y_true, y_pred):
    """ MASE IMPLIMENTATION (assuming no seasonality of data)"""

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    # find MAE of naive forecast (no seasonality)
    # our seasonality is 1 day (hence shift of one)
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    """
    Calculates various evaluation metrics and returns results as a dictionary
    """

    # assert float32 datatype for metric calculations
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    results = {
        "MAE": mae.numpy(),
        "MSE": mse.numpy(),
        "RMSE": rmse.numpy(),
        "MAPE": mape.numpy(),
        "MASE": mase.numpy()
    }

    pprint(results)

    return results

def create_model_checkpoint(model_number, model_name):
    """
    Create function to create a model checkpoint of a specific filename
    """
    filepath = f"../models/{model_number}/{model_name}"
    return tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)

def make_preds(model, input_data):
    """
    Uses model to make predictions on input_data.
    """
    return tf.squeeze(model.predict(input_data))


if __name__ == "__main__":
    filename = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
    # create_train_test_datasets(filename, log=True)
    create_train_test_datasets(filename, False, True, 1, 7)

