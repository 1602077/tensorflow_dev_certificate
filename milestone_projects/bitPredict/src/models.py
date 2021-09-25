import ast
import os
import json
import sys
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pprint import pprint

from dataProccessing import *

MODEL_NAMES = ["0_Baseline_Naive", "1_Dense_W7_H1", "2_Dense_W30_H1", "3_Dense_W30_H7"]

def model_0():
    """
    Baseline (Model 0): Naive forecast
    i.e. predicting the next timestep as the previous timestep
    """

    X_train, y_train, X_test, y_test = create_train_test_datasets()

    naive_forecast = y_test[:-1]

    plt.figure(figsize=(10,7))
    plot_time_series(X_test, y_test, start=350, format="-", label="test data")
    plot_time_series(X_test[1:], naive_forecast, start=350, format="-", label="Naive forecast")
    plt.savefig("../models/0/forecast.png", bbox_inches="tight", dpi=200)

    model_0_results = evaluate_preds(y_true=y_test[1:], y_pred=naive_forecast)

    return model_0_results

def model_1(dir_num, HORIZON=1, WINDOW_SIZE=7):
    """

    """


    train_windows, train_labels, test_windows, test_labels = create_train_test_datasets(window=True, horizon=HORIZON, window_size=WINDOW_SIZE)

    print(f"Length of train and test sets: {len(train_windows), len(test_windows)}")
    # Check to see if train labels are the same (before and after window split)
    X_train, y_train, X_test, y_test = create_train_test_datasets()
    print(f"Train labels are the same before and after window split: {np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:])}")
    
    tf.random.set_seed(42)

    model_1 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON, activation="linear")
    ], name=MODEL_NAMES[1])
    
    model_1.compile(loss="mae", optimizer="adam", metrics=["mae", "mse"])

    model_1.fit(x=train_windows,
                y=train_labels,
                epochs=100,
                verbose=2, 
                batch_size=128, 
                validation_data=(test_windows, test_labels),
                callbacks=[create_model_checkpoint(dir_num, MODEL_NAMES[dir_num])]
    )

    # Load in best performing model
    model_1 = tf.keras.models.load_model(f"../models/{dir_num}/{MODEL_NAMES[dir_num]}")
    model_1_preds = make_preds(model_1, test_windows)

    print(f"\nModel {dir_num} Results:\n")
    model_1_results = evaluate_preds(tf.squeeze(test_labels), model_1_preds)

    # plot model
    offset = 300
    plt.figure(figsize=(10,7))
    # index into test_labels to account for the test window offset
    plot_time_series(timesteps=X_test[-len(test_windows):],  values=test_labels[:, 0], start=offset, label="test data")

    plot_time_series(timesteps=X_test[-len(test_windows):],  values=tf.reduce_mean(model_1_preds, axis=1), start=offset, format="-", label=f"model {dir_num} preds")
    plt.savefig(f"../models/{dir_num}/model_{dir_num}_predictions.png", dpi=250, bbox_inches="tight")
    return


def train_all_models():
    """ Trains all models functionally defined in the format model_{INT} """
    if not os.path.exists(f"../models/0"):
        os.mkdir(f"../models/0")

        sys.stdout = open(f"../models/0/{MODEL_NAMES[0]}.log", "w")
        print(f"Model: {MODEL_NAMES[0]}\n")
        model_0()

    if not os.path.exists(f"../models/1"):
        os.mkdir(f"../models/1")

        sys.stdout = open(f"../models/1/{MODEL_NAMES[1]}.log", "w")
        print(f"Model: {MODEL_NAMES[1]}\n")
        model_1(1, HORIZON=1, WINDOW_SIZE=7)

    if not os.path.exists(f"../models/2"):
        os.mkdir(f"../models/2")

        sys.stdout = open(f"../models/2/{MODEL_NAMES[2]}.log", "w")
        print(f"Model: {MODEL_NAMES[2]}\n")
        model_1(2, HORIZON=1, WINDOW_SIZE=30)

    if not os.path.exists(f"../models/3"):
        os.mkdir(f"../models/3")

        sys.stdout = open(f"../models/3/{MODEL_NAMES[3]}.log", "w")
        print(f"Model: {MODEL_NAMES[3]}\n")
        model_1(3, HORIZON=7, WINDOW_SIZE=30)

    return

def compare_model_performances(model_names=MODEL_NAMES):
    """
    Collate model_performances from log files into a pandas dataframe  
    """
    df = pd.DataFrame()  # Empty df to store model metrics in

    for model_name in model_names:
        model_num = int(model_name.split("_")[0])

        with open(f"../models/{model_num}/{model_name}.log", "r") as f:
            # Read in last 5 lines of log, which contain model metrics 
            # and then convert lines -> str -> dict -> pandas df.
            lines = f.readlines()[-5:]
            json_string = json.dumps(ast.literal_eval(u' '.join(lines)))
            metrics = json.loads(json_string)  # Conert json str to dict
            metrics_and_name = {**{"Model": model_name}, **metrics}

            df = df.append(metrics_and_name, ignore_index=True)

    print(df)
    df.to_csv("../models/model_performance_summaries.csv")

    # Plot models as a horizontal bar chart
    plt.figure(figsize=(10,7))
    df.plot(x="Model", y="MAE", kind="barh")
    plt.xlabel("Mean absolute error")
    plt.savefig("../models/model_performance_MAE_plot.png", bbox_inches="tight", dpi=250)
    


if __name__ == "__main__":
    train_all_models()
    compare_model_performances()

