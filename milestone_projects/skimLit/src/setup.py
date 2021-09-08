import os


def setup():
    """
    Helper function to download mrdbourke's helper_functions.py and download the pubmed_rct dataset from github repo
    """
    os.chdir("/Users/jcmunday/Desktop/tensorflow/milestone_projects/skimLit/src")
    # Download helper functions
    if not os.path.exists("helper_functions.py"):
        os.system("wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py")

    # Download RTC dataset
    os.chdir("..")
    if not os.path.exists("data"):
        os.system("git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git")
        os.rename("pubmed-rct", "data")

    # make new directories to store logs and model files
    new_dirs = ["logs", "models"]
    for dir in new_dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    return


if __name__ == "__main__":
    setup()
