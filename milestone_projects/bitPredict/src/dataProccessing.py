import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.preprocessing import minmax_scale


def create_train_test_datasets(filename="BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                                test_split=0.2,
                                window=False,
                                horizon=1,
                                window_size=7,
                                block_reward=False):
    """
    Create train and test datasets on BTC price data.

    params:
        filename     (str)   : filename of csv containing BTC price data to be time forecasted
        test_split   (float) : size of test set
        window       (bool)  : if true window data to transform into a supervised learning problem
        horizon      (int)   : number of days to predict data for
        window_size  (int)   : number of days of data used to predict horizon
        block_reward (bool)  : if true add in block_reward values
    """

    df = pd.read_csv(f"../data/{filename}", parse_dates=["Date"], index_col=["Date"])
    df = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})

    timesteps = df.index.to_numpy()
    prices = df["Price"].to_numpy()
    
    if window and not block_reward:
        full_windows, full_labels = make_windows(prices, horizon=horizon, window_size=window_size)
        train_windows, train_labels, test_windows, test_labels = make_train_test_split(full_windows, full_labels, test_split=test_split)
        
        return train_windows, train_labels, test_windows, test_labels

    if block_reward:
        block_reward_1 = 50 # 3rd Jan 2009 i.e. not in our dataset
        block_reward_2 = 25 # 8th Nov 2012
        block_reward_3 = 12.5 # 9th Jul 2016
        block_reward_4 = 6.25 # 18 May 2020

        block_reward_2_datetime = np.datetime64("2012-11-28")
        block_reward_3_datetime = np.datetime64("2016-07-09")
        block_reward_4_datetime = np.datetime64("2020-05-18")

        block_reward_2_days = (block_reward_3_datetime - df.index[0]).days
        block_reward_3_days = (block_reward_4_datetime - df.index[0]).days

        prices_block = df.copy()
        prices_block["block_reward"] = None
        prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
        prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
        prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

        # Plot block_reward and btc price on scaled axes
        scaled_prices_block_df = pd.DataFrame(minmax_scale(prices_block[["Price", "block_reward"]]),
                                                            columns=prices_block.columns,
                                                            index=prices_block.index)
        scaled_prices_block_df.plot(figsize=(10,7), title="BTC Price & Block Reward Size (scaled) [Oct-13 -> May-21]")
        plt.savefig("../logs/dataProcessing_btc_price_block_reward.png", bbox_inches="tight", dpi=250)

        # Make window dataset using pandas (univariate time series helper funcs will not longer work)
        bitcoin_prices_windowed = prices_block.copy()
        for i in range(window_size):  # Add windowed columns
            bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

        # Create X (windows) and y (horizons) features
        X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
        y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

        # Make train and test sets
        split_size = int(0.8 * len(X))
        X_train, y_train = X[:split_size], y[:split_size]
        X_test, y_test = X[split_size:], y[split_size:]

        return X_train, y_train, X_test, y_test

    else:
        # Create basic train and test splits
        split_size = int(0.8 * len(prices))  # 80/20 split
        X_train, y_train = timesteps[:split_size], prices[:split_size]
        X_test, y_test = timesteps[split_size:], prices[split_size:]

        plt.figure(figsize=(10,7))
        plot_time_series(X_train, y_train, label="train data")
        plot_time_series(X_test, y_test, label="test data")
        plt.savefig("../logs/train_test_split_naive.png", bbox_inches="tight", dpi=250)

        return X_train, y_train, X_test, y_test


def nbeats_data_pipeline(filename="BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                        split_size_ratio = 0.8,
                        HORIZON=1,
                        WINDOW_SIZE=7):
    """ Creating a performant data pipeline for n_beats algorithm"""

    # Create N-BEATS input
    df = pd.read_csv(f"../data/{filename}", parse_dates=["Date"], index_col=["Date"])
    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})

    bitcoin_prices_nbeats = bitcoin_prices.copy()
    for i in range(WINDOW_SIZE):
        bitcoin_prices_nbeats[f"Price+{i+1}"] = bitcoin_prices_nbeats["Price"].shift(periods=i+1)

    # Make features and labels
    X = bitcoin_prices_nbeats.dropna().drop("Price", axis=1)
    y = bitcoin_prices_nbeats.dropna()["Price"]

    # Make train and test sets
    split_size = int(len(X) * split_size_ratio)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]

    # Create performant tf datasets
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels = tf.data.Dataset.from_tensor_slices(y_train)

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels = tf.data.Dataset.from_tensor_slices(y_test)
    
    # Combine labels and features zip -> tuple
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels))

    # Batch and prefecth datasets
    BATCH_SIZE = 1024
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, X_test, y_test
        

def labelled_windows(x, horizon):
    """ Create labels for windows dataset """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, horizon, window_size):
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


def make_train_test_split(windows, labels, test_split):
    """ Splits matching pairs of windows and labels into train and test sets """

    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]

    return train_windows, train_labels, test_windows, test_labels


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

    # Account for different sized metrics for longer horizons i.e. reduce metrics
    # to a single value.
    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

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
    """ Creates a model checkpoint of a specific filename """

    filepath = f"../models/{model_number}/{model_name}"
    return tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)

def make_preds(model, input_data):
    """ Use model to make predictions on input_data """
    return tf.squeeze(model.predict(input_data))


def create_future_dataset(filename="BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                                test_split=0.2,
                                window=False,
                                horizon=1,
                                window_size=7,
                                ):

    """ Pipeline to create dataset to predict on the future """

    df = pd.read_csv(f"../data/{filename}", parse_dates=["Date"], index_col=["Date"])
    df = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})

    timesteps = df.index.to_numpy()
    prices = df["Price"].to_numpy()
    

    block_reward_1 = 50 # 3rd Jan 2009 i.e. not in our dataset
    block_reward_2 = 25 # 8th Nov 2012
    block_reward_3 = 12.5 # 9th Jul 2016
    block_reward_4 = 6.25 # 18 May 2020

    block_reward_2_datetime = np.datetime64("2012-11-28")
    block_reward_3_datetime = np.datetime64("2016-07-09")
    block_reward_4_datetime = np.datetime64("2020-05-18")

    block_reward_2_days = (block_reward_3_datetime - df.index[0]).days
    block_reward_3_days = (block_reward_4_datetime - df.index[0]).days

    prices_block = df.copy()
    prices_block["block_reward"] = None
    prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
    prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
    prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

    # Make window dataset using pandas (univariate time series helper funcs will not longer work)
    bitcoin_prices_windowed = prices_block.copy()
    for i in range(window_size):  # Add windowed columns
        bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

    X_all = bitcoin_prices_windowed.dropna().drop(["Price", "block_reward"], axis=1).to_numpy()
    y_all = bitcoin_prices_windowed.dropna()["Price"].to_numpy()

    features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
    labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

    BATCH_SIZE = 1024
    dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))
    dataset_all = dataset_all.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset_all, X_all, y_all, df


if __name__ == "__main__":
    create_future_dataset()
    # filename = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
    # nbeats_data_pipeline()
    # create_train_test_datasets(filename=filename,
    #                             test_split=0.2,
    #                             window=False,
    #                             horizon=1,
    #                             window_size=7,
    #                             block_reward=True)

