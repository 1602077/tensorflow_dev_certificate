import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf


def get_lines(filename):
    """
    Read filename and returns the lines of text as a list
    Args:
        filename: a string containing the target filepath
    Returns:
        A list of strings with one string per line from the target filename.
    """
    with open(filename, "r") as f:
        return f.readlines()


def preprocess_text_with_line_numbers(filename):
    """
    Returns a list of dictionaries of abstract line data.

    Takes in filename, reads its contents and sorts through each line,
    extracting things like the target label, the text of the sentence,
    how many sentences are in the current abstract and what sentence number
    the target line is.

    Args:
      filename: a string of the target text file to read and extract line data
      from.

    Returns:
      A list of dictionaries each containing a line from an abstract,
      the lines label, the lines position in the abstract and the total number
      of lines in the abstract where the line is from. For example:

      [{"target": 'CONCLUSION',
        "text": The study couldn't have gone better, turns out people are kinder than you think",
        "line_number": 8,
        "total_lines": 8}]
    """
    input_lines = get_lines(filename)  # get all lines from filename
    abstract_lines = ""  # create an empty abstract
    abstract_samples = []  # create an empty list of abstracts

    # Loop through each line in target file
    for line in input_lines:
        if line.startswith("###"):  # check to see if line is an ID line
            abstract_id = line
            abstract_lines = ""  # reset abstract string
        elif line.isspace():  # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines()  # split abstract into separate lines
        # Iterate through each line in abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}  # create empty dict to store data from line
                target_text_split = abstract_line.split("\t")  # split target label from text
                line_data["target"] = target_text_split[0]  # get target label
                line_data["text"] = target_text_split[1].lower()  # get target text and lower it
                line_data["line_number"] = abstract_line_number  # what number line does the line appear in the abstract
                # how many total lines are in the abstract? (start from 0)
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)  # add line data to abstract samples list

        else:  # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line

    return abstract_samples


def create_dataset(train_filename, val_filename, test_filename, one_hot=True):
    """
    Creates a train, val, test dataset with labels either one hot or seq
    encoded from a set of input directories.
    """
    train_df = pd.DataFrame(preprocess_text_with_line_numbers(train_filename))
    val_df = pd.DataFrame(preprocess_text_with_line_numbers(val_filename))
    test_df = pd.DataFrame(preprocess_text_with_line_numbers(test_filename))

    if one_hot:
        one_hot_encoder = OneHotEncoder(sparse=False) 
        train_labels = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
        val_labels = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
        test_labels = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

        num_classes = len(one_hot_encoder.categories_)
        class_names = one_hot_encoder.categories_
    else:
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_df["target"].to_numpy())
        val_labels = label_encoder.transform(val_df["target"].to_numpy())
        test_labels = label_encoder.transform(test_df["target"].to_numpy())

        num_classes = len(label_encoder.classes_)
        class_names = label_encoder.classes_

    train_df.drop(["target"], axis=1, inplace=True)
    val_df.drop(["target"], axis=1, inplace=True)
    test_df.drop(["target"], axis=1, inplace=True)
    
    return train_df, train_labels, val_df, val_labels, test_df, test_labels, num_classes, class_names


def create_tf_datasets(data_dir):

    datasets = create_dataset(
        data_dir + "train.txt",
        data_dir + "dev.txt",
        data_dir + "test.txt",
        one_hot=True
    )

    train_df, train_labels, val_df, val_labels, test_df, test_labels, num_classes, class_names = datasets
    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()
    test_sentences = test_df["text"].tolist()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels))

    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return train_dataset, valid_dataset, test_dataset
