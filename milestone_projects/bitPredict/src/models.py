import ast
import os
import json
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from pprint import pprint

from dataProccessing import *

MODEL_NAMES = [
    "0_Baseline_Naive",
    "1_Dense_W7_H1",
    "2_Dense_W30_H1",
    "3_Dense_W30_H7",
    "4_Conv1D",
    "5_LSTM",
    "6_Dense_Multivariate",
    "7_NBEATS",
    "8_EnsembleModel"
]

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
    """ Dense model using varying window and horizon sizes """

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


def model_4():
    """ Conv1D Model """

    HORIZON = 1
    WINDOW_SIZE = 7
    train_windows, train_labels, test_windows, test_labels = create_train_test_datasets(window=True, horizon=HORIZON, window_size=WINDOW_SIZE)

    # To use Conv1D, need input shape of (batch_size, timesteps, input_dim)
    x = tf.constant(train_windows[0])
    # add extra dim for input dim i.e. (7,) -> (7,1)
    expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))

    print("Testing lambda layer\n" + 40*"-")
    print(f"Original shape: {x.shape}")
    print(f"Expanded shape: {expand_dims_layer(x).shape}")
    print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")

    tf.random.set_seed(42)

    model_4 = tf.keras.Sequential([
       layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
       layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
       layers.Dense(HORIZON)
    ], name=MODEL_NAMES[4])

    model_4.compile(loss="mae", optimizer="adam", metrics=["mae", "mse"])

    model_4.fit(
        train_windows,
        train_labels,
        batch_size=128,
        epochs=100,
        verbose=2,
        validation_data=(test_windows, test_labels),
        callbacks=[create_model_checkpoint(4, MODEL_NAMES[4])]
    )

    print(model_4.summary())
    model_4 = tf.keras.models.load_model(f"../models/4/{MODEL_NAMES[4]}")
    model_4_preds = make_preds(model_4, test_windows)
    print(f"\nModel 4 Results:\n")
    model_1_results = evaluate_preds(tf.squeeze(test_labels), model_4_preds)
    # plot model
    _, _, X_test, _ = create_train_test_datasets()
    offset = 300
    plt.figure(figsize=(10,7))
    # index into test_labels to account for the test window offset
    plot_time_series(timesteps=X_test[-len(test_windows):],  values=test_labels[:, 0], start=offset, label="test data")
    plot_time_series(timesteps=X_test[-len(test_windows):],  values=model_4_preds, start=offset, format="-", label=f"model 4 preds")
    plt.savefig(f"../models/4/model_4_predictions.png", dpi=250, bbox_inches="tight")

    return


def model_5():
    """ LSTM Model """
    HORIZON = 1
    WINDOW_SIZE = 7
    train_windows, train_labels, test_windows, test_labels = create_train_test_datasets(window=True, horizon=HORIZON, window_size=WINDOW_SIZE)

    tf.random.set_seed(42)

    inputs = layers.Input(shape=(WINDOW_SIZE))
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    # x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(HORIZON)(x)

    model_5 = tf.keras.Model(inputs, outputs, name=MODEL_NAMES[5])

    model_5.compile(loss="mae", optimizer="adam")

    model_5.fit(
        train_windows,
        train_labels,
        batch_size=128,
        epochs=100,
        verbose=2,
        validation_data=(test_windows, test_labels),
        callbacks=[create_model_checkpoint(5, MODEL_NAMES[5])]
    )

    print(model_5.summary())

    model_5 = tf.keras.models.load_model(f"../models/5/{MODEL_NAMES[5]}")
    model_5_preds = make_preds(model_5, test_windows)

    print(f"\nModel 5 Results:\n")
    model_5_results = evaluate_preds(tf.squeeze(test_labels), model_5_preds)
    # plot model
    _, _, X_test, _ = create_train_test_datasets()
    offset = 300
    plt.figure(figsize=(10,7))
    # index into test_labels to account for the test window offset
    plot_time_series(timesteps=X_test[-len(test_windows):],  values=test_labels[:, 0], start=offset, label="test data")
    plot_time_series(timesteps=X_test[-len(test_windows):],  values=model_5_preds, start=offset, format="-", label=f"model 5 preds")
    plt.savefig(f"../models/5/model_5_predictions.png", dpi=250, bbox_inches="tight")
    return


def model_6():
    """ Model 1 architecture with multivariate data """

    HORIZON = 1
    WINDOW_SIZE = 7

    X_train, y_train, X_test, y_test = create_train_test_datasets(test_split=0.2, 
                                                                window=False,
                                                                horizon=HORIZON,
                                                                window_size=WINDOW_SIZE,
                                                                block_reward=True)
    tf.random.set_seed(42)

    model_6 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON)
    ], name=MODEL_NAMES[6])

    model_6.compile(loss="mae", optimizer="adam")

    model_6.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=128,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[create_model_checkpoint(6, MODEL_NAMES[6])]
    )

    print(model_6.summary())

    model_6 = tf.keras.models.load_model(f"../models/6/{MODEL_NAMES[6]}")
    model_6_preds = tf.squeeze(model_6.predict(X_test))

    print(f"\nModel 6 Results:\n")
    model_6_results = evaluate_preds(y_test, model_6_preds)
    return


def model_7():
    """ N-BEATS Basic Block Arch Replication """

    # Create NBESTBLOCK custom layer
    class NBeatsBlock(tf.keras.layers.Layer):
        def __init__(self, 
                    input_size: int, 
                    theta_size: int, 
                    horizon: int, 
                    n_neurons: int, 
                    n_layers: int, 
                    **kwargs):

            super().__init__(**kwargs)
            self.input_size = input_size
            self.theta_size = theta_size
            self.horizon = horizon
            self.n_neurons = n_neurons
            self.n_layers = n_layers
        
            # Block contains stack of 4 FC layers each w/ ReLU activation
            self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range (n_layers)]
            # Output of block theta w/ Lin activation
            self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

        def call(self, inputs):
            x = inputs
            for layer in self.hidden:
                x = layer(x)
            theta = self.theta_layer(x)
            # Ouput the backfast and forecast from theta
            backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
            return backcast, forecast
    
    # Testing N-BEATS Block Class
    HORIZON, WINDOW_SIZE = 1, 7

    dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
                                            theta_size=WINDOW_SIZE + HORIZON,
                                            horizon=HORIZON,
                                            n_neurons=128,
                                            n_layers=4)

    # Create dummy inputs and pass into n-beats block layer
    dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0)
    backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)

    print("\nTesting N-BEATS Block Class")
    print(f"Dummy Input Shape: {dummy_inputs.shape}")
    print(f"Backcast: {tf.squeeze(backcast.numpy())}")
    print(f"Forecast: {tf.squeeze(forecast.numpy())}")

    train_dataset, test_dataset, y_test = nbeats_data_pipeline(HORIZON=HORIZON, WINDOW_SIZE=WINDOW_SIZE)

    # Set up hyperparameters for N-BEATS algorithm
    N_EPOCHS = 5000
    N_NEURONS = 512
    N_LAYERS = 4
    N_STACKS = 30

    INPUT_SIZE = WINDOW_SIZE * HORIZON
    THETA_SIZE = INPUT_SIZE + HORIZON

    tf.random.set_seed(42)

    # 1. Setup an instance of NBeatsBlock
    nbeats_block_layer = NBeatsBlock(
        input_size=INPUT_SIZE,
        theta_size=THETA_SIZE,
        horizon=HORIZON,
        n_neurons=N_NEURONS,
        n_layers=N_LAYERS,
        name="IntialBlock"
    )

    # 2. Create an input layer for N-BEATS stack
    stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")

    # 3. Create intial back- and forecasts
    residuals, forecast = nbeats_block_layer(stack_input)

    # 4 - 6. Create stacks of block layers, stack blocks and double residual stacking
    for i, _ in enumerate(range(N_STACKS-1)):  # first stack already created in 3
        backcast, block_forecast = NBeatsBlock(
            input_size=INPUT_SIZE,
            theta_size=THETA_SIZE,
            horizon=HORIZON,
            n_neurons=N_NEURONS,
            n_layers=N_LAYERS,
            name=f"NBeatsBlock_{i}"
        )(residuals)

        residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}")
        forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

    # 7. Create model
    model_7 = tf.keras.Model(inputs=stack_input, outputs=forecast, name=MODEL_NAMES[7])

    # 8. Compile model
    model_7.compile(loss="mae", optimizer="adam")

    print("\nTraining N-BEATS Model")
    # 9. Fit model
    model_7.fit(
        train_dataset,
        epochs=N_EPOCHS,
        validation_data=test_dataset,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=2)
        ]
    )

    model_7.save(f"../models/7/{MODEL_NAMES[7]}")

    # Make predictions with N-BEATS model
    model_7_preds = make_preds(model_7, test_dataset)
    model_7_results = evaluate_preds(y_test, model_7_preds)

    # Plotting the N-BEATS Architecture
    plot_model(model_7, to_file=f"../models/7/{MODEL_NAMES[7]}.png")

    return


def get_ensemble_models(horizon, train_data, test_data, num_iter=10, num_epochs=1000, loss_fns=["mae", "mse", "mape"]):
    """
    Returns a list of num_iter models each trained on MAE, MSE, MAPE loss.
    e.g. if num_iter=10, a list of 30 models will be returned
    """

    ensemble_models = []

    for i in range(num_iter):
        for loss_function in loss_fns:
            print(f"Optimizing model by reducing {loss_function} for {num_epochs} epochs, model number: {i}")
            # Construct a simple model similiar to model_1
            model = tf.keras.Sequential([
                layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                layers.Dense(horizon)
            ])

            model.compile(
                loss=loss_function,
                optimizer="adam",
                metrics=["mae", "mse"]
            )

            model.fit(
                train_data,
                epochs=num_epochs,
                verbose=0,
                validation_data=test_data,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)
                ]
            )
            ensemble_models.append(model)

    return ensemble_models


def make_ensemble_preds(ensemble_models, data):
    ensemble_preds = []
    for model in ensemble_models:
        preds = model.predict(data)
        ensemble_preds.append(preds)
    return tf.constant(tf.squeeze(ensemble_preds))
        


def model_8():
    """ Training an ensembe model """

    HORIZON, WINDOW_SIZE = 1, 7
    train_dataset, test_dataset, y_test = nbeats_data_pipeline(HORIZON=HORIZON, WINDOW_SIZE=WINDOW_SIZE)

    ensemble_models = get_ensemble_models(
        horizon=HORIZON,
        train_data=train_dataset,
        test_data=test_dataset,
        num_iter=5,
        num_epochs=1000
    )

    ensemble_preds = make_ensemble_preds(ensemble_models, test_dataset)
    ensemble_mean = tf.reduce_mean(ensemble_preds, axis=0)
    ensemble_median = np.median(ensemble_preds, axis=0)
    
    print("Model 8 Results")
    ensemble_reults = evaluate_preds(y_test, ensemble_mean)
            


def train_and_save_model(model_num):
    """ Trains, saves and evaluates a model """

    if not os.path.exists(f"../models/{model_num}"):
        os.mkdir(f"../models/{model_num}")
        sys.stdout = open(f"../models/{model_num}/{MODEL_NAMES[model_num]}.log", "w")
        print(f"Model: {MODEL_NAMES[model_num]}\n")
        eval(f"model_{model_num}()")
    return


def train_all_models():
    """ Trains all models functionally defined in the format model_{INT} """

    train_and_save_model(0) 

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

    train_and_save_model(4)
    train_and_save_model(5)
    train_and_save_model(6)
    train_and_save_model(7)
    train_and_save_model(8)
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
            metrics = json.loads(json_string)  # Convert json str to dict
            metrics_and_name = {**{"Model": model_name}, **metrics}

            df = df.append(metrics_and_name, ignore_index=True)

    print(df)
    df.to_csv("../models/model_performance_summaries.csv")

    # Plot models as a horizontal bar chart
    plt.figure(figsize=(10,7))
    df.plot(x="Model", y="MAE", kind="barh")
    plt.xlabel("Mean absolute error")
    plt.savefig("../models/model_performance_MAE_plot.png", bbox_inches="tight", dpi=250)
    return


if __name__ == "__main__":
    # model_8()
    train_all_models()
    compare_model_performances()

