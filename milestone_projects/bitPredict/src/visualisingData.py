import sys
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime


sys.stdout = open("../logs/visualisingData.log", "w")

def visualise_data_pandas(filename):
    """ Visualise BTC price using pandas module """

    df = pd.read_csv(f"../data/{filename}",
                    parse_dates=["Date"],
                    index_col=["Date"])

    print(f"Filename: {filename}\n")
    print("\nVisualising using pandas library")
    print(df.head(), "\n")
    print(df.info())
    print(f"\nNumber of samples: {len(df)}\n")  # roughly 8 years of daily data

    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    print(bitcoin_prices.head())
    
    # Plot data
    bitcoin_prices.plot(figsize=(10, 7),
                        title = "Price of Bitcoin (Oct-13 to May-21)",
                        ylabel="BTC Price",
                        legend=False)

    plt.savefig("../logs/visualisingData_bitcoinPrices_pd.png", dpi=200, bbox_inches="tight")
    return


def visualise_data_csv(filename):
    """ Visualise BTC price using python's csv module """

    print("\nVisualing using csv module")
    timesteps, btc_price = [], []
    with open(f"../data/{filename}", "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)  # skip headers
        for line in csv_reader:
            timesteps.append(datetime.strptime(line[1], "%Y-%m-%d"))
            btc_price.append(float(line[2]))
    print(f"First 10 timesteps:\n{timesteps[:10]}\n{btc_price[:10]}\n")
    
    # plot data
    plt.figure(figsize=(10,7))
    plt.plot(timesteps, btc_price)
    plt.title("Price of Bitcoin (Oct-13 to May-21)")
    plt.xlabel("Date")
    plt.ylabel("BTC Price")
    plt.savefig("../logs/visualisingData_bitcoinPrices_csv.png", dpi=200, bbox_inches="tight")
    return

if __name__ == "__main__":
    filename = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
    visualise_data_pandas(filename=filename)
    visualise_data_csv(filename=filename)

