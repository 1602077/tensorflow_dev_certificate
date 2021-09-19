import os
import sys


def setup():
    """
    Helper function to download data and setup dir structure for project
    """
    # Dowloadnig data
    if not os.path.exists("../data"):
        os.mkdir("../data")
        os.chdir("../data")
        url = "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
        os.system("wget " + str(url))

    os.chdir("/Users/jcmunday/Desktop/tensorflow/milestone_projects/bitPredict")

    new_dirs = ["logs", "models"]
    for dir in new_dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    return


if __name__ == "__main__":
    setup()

