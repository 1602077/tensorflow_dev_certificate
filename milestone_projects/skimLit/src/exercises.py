import os
import string
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.utils import plot_model

from helper_functions import calculate_results
from preprocess_data import preprocess_text_with_line_numbers

pd.set_option('display.max_columns', None, 'display.max_rows', None)

# 0. Load Data
print(f"{50*'-'}\n  Loading Data\n{50*'-'}")
data_dir = "../data/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
train_filename, val_filename, test_filename = data_dir + "train.txt", data_dir + "dev.txt", data_dir + "test.txt"

train_df = pd.DataFrame(preprocess_text_with_line_numbers(train_filename))
val_df = pd.DataFrame(preprocess_text_with_line_numbers(val_filename))
test_df = pd.DataFrame(preprocess_text_with_line_numbers(test_filename))
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()
print(f"Number of train, val & test sentence: {len(train_sentences)}, {len(val_sentences)}, {len(test_sentences)}.")

sent_lens = [len(sentence.split()) for sentence in train_sentences]
# what length for padding would cover 95% of examples
output_seq_len = int(np.percentile(sent_lens, 95))

# CREATE TEXT VECTORISER LAYER
print(f"{50*'-'}\n  Creating Vectoriser Layer\n{50*'-'}")
max_tokens = 68000
vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_seq_len)
text_ds = tf.data.Dataset.from_tensor_slices(train_sentences).batch(32).prefetch(tf.data.AUTOTUNE)
vectorizer.adapt(text_ds)

rct_20k_text_vocab = vectorizer.get_vocabulary()

print(f"First 5 words in vocab: {rct_20k_text_vocab[:5]}")

# Create a dictionary to map words back
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))
test = ["the", "cat", "sat", "on", "the", "mat"]
print(f"{test}    ->    {[word_index[w] for w in test]}")


def split_chars(text):
    """ Split sentences into characters"""
    return " ".join(list(text))


print(f"\n{50*'-'}\n  Creating character embeddings\n{50*'-'}")
# split sequence level data splits into char-level data split
train_chars = [split_chars(sent) for sent in train_sentences]
val_chars = [split_chars(sent) for sent in val_sentences]
test_chars = [split_chars(sent) for sent in test_sentences]

char_lens = [len(sent) for sent in train_sentences]
mean_char_len = np.mean(char_lens)
print(f"Average character length: {mean_char_len}")
# output_seq_char_len = 100
output_seq_char_len = int(np.percentile(char_lens, 95))
print(f"Sequence length which covers 95% of sentences: {output_seq_char_len}")

# Get all keyboard characters
alphabet = string.ascii_lowercase + string.digits + string.punctuation
print(f"Possible characters: {alphabet}")
NUM_CHAR_TOKENS = len(alphabet) + 2  # for space and OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_len,
                                    name="char_vectoriser")
char_vectorizer.adapt(train_chars)
char_vocab = char_vectorizer.get_vocabulary()
print(f"Number of chars in vocab: {len(char_vocab)}")
print(f"Top 5 chars: {char_vocab[:5]}")
print(f"Bottom 5 chars: {char_vocab[-5:]}")

# Create character level embeddings
char_embed = layers.Embedding(input_dim=len(char_vocab),
                              output_dim=25,
                              mask_zero=False,
                              name="char_embed")

one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())


# Create positional embeddings
print(f"\n{50*'-'}\n  Creating positional embeddings\n{50*'-'}")
print("Number of different line numbers")
print(train_df["line_number"].value_counts())

# Use tf to create one-hot-encoded tensors of our "line_number" column
train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)
print(f"Train line number one-hot [:10]:\n{train_line_numbers_one_hot[:10]}")
print(f"Shape: {train_line_numbers_one_hot.shape}")

# one_hot our total lines feature
print(f"Length coverage 95% of total_lines: {np.percentile(train_df.total_lines, 95)}")
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)
print(f"Train total line numbers one-hot [:10]:\n{train_total_lines_one_hot[:10]}")
print(f"Shape: {train_total_lines_one_hot.shape}")


# Create tri-embedding datasets into a tf.data.Dataset
train_char_token_pos_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,
                                                                train_total_lines_one_hot,
                                                                train_sentences,
                                                                train_chars))
train_token_pos_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_char_token_pos_dataset = tf.data.Dataset.zip((train_char_token_pos_data, train_token_pos_labels))
train_char_token_pos_dataset = train_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_char_token_pos_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                              val_total_lines_one_hot,
                                                              val_sentences,
                                                              val_chars))
val_token_pos_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_pos_dataset = tf.data.Dataset.zip((val_char_token_pos_data, val_token_pos_labels))
val_char_token_pos_dataset = val_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

test_char_token_pos_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
                                                               test_total_lines_one_hot,
                                                               test_sentences,
                                                               test_chars))
test_token_pos_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_char_token_pos_dataset = tf.data.Dataset.zip((test_char_token_pos_data, test_token_pos_labels))
test_char_token_pos_dataset = test_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


# Creating GloVe Embeddings
glove_dir = "../models/GLoVe"
if not os.path.exists(glove_dir):
    os.mkdir(glove_dir)
    os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
    os.system(f"unzip -q glove.6B.zip -d {glove_dir}")
    os.system("rm glove.6B.zip")

embeddings_index = {}
with open(glove_dir + "/glove.6B.300d.txt") as f:
    for line in f:
        word, coeffs = line.split(maxsplit=1)
        coeffs = np.fromstring(coeffs, "f", sep=" ")
        embeddings_index[word] = coeffs

print("\nFound %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 300
hits, misses = 0, 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True,
    name="GLoVe_Embedding"
)

sys.stdout = open("../logs/exercises.log", "w")
# Model 7: Tri-Embedding Model using GLoVE Embeddings
M7_NAME = "Model7_Tri_Embed_GLoVe"
# Building the tribrid embedding model
# 1. Token inputs
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_inputs")
token_vec = vectorizer(token_inputs)
x = embedding_layer(token_vec)
x = layers.GlobalAveragePooling1D()(x)
token_outputs = layers.Dense(128, activation="relu")(x)
token_model = tf.keras.Model(token_inputs, token_outputs, name="M7_token_model")
# token_model.summary()

# 2. Char inputs
char_inputs = layers.Input(shape=(1,), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
char_model = tf.keras.Model(char_inputs, char_bi_lstm, name="M7_bi_lstm")
# char_model.summary()

# 3. Line number inputs
line_num_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_num_inputs")
line_num_dense = layers.Dense(32, activation="relu")(line_num_inputs)
line_num_model = tf.keras.Model(line_num_inputs, line_num_dense, name="M7_line_num")
# line_num_model.summary()

# 4. Total line number inputs
total_num_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="total_num_inputs")
x = layers.Dense(32, activation="relu")(total_num_inputs)
total_num_model = tf.keras.Model(total_num_inputs, x, name="M7_total_num")
# total_num_model.summary()

# 7. Combine token & char embed into hybrid embedding
combined_embedding = layers.Concatenate(name="M7_char_token_hybrid_embedding")([token_model.output, char_model.output])
combined_embedding = layers.Dense(256, activation="relu")(combined_embedding)
combined_embedding = layers.Dropout(0.5)(combined_embedding)

# 6. Combine positional embedding with combined token and char embeddings
tribrid_embeddings = layers.Concatenate(name="M7_char_token_pos_embed")([line_num_model.output,
                                                                         total_num_model.output,
                                                                         combined_embedding])
# 7. Create output layer
output_layer = layers.Dense(5, activation="softmax", name="output_layer")(tribrid_embeddings)

# 7. Piece together model
model_7 = tf.keras.Model(inputs=[line_num_model.input, total_num_model.input, token_model.input, char_model.input],
                         outputs=output_layer,
                         name=M7_NAME)

model_7.summary()
plot_model(model_7, to_file="../logs/model7_summary.png", show_shapes=True)

# Compile token, char and positional embedding model
model_7.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


history_model_7 = model_7.fit(
    train_char_token_pos_dataset,
    steps_per_epoch=int(0.1 * len(train_char_token_pos_dataset)),
    epochs=5,
    validation_data=val_char_token_pos_dataset,
    validation_steps=int(0.1 * len(val_char_token_pos_dataset)),
    verbose=2
)

model_7_probabilities = model_7.predict(val_char_token_pos_dataset)
model_7_predictions = tf.argmax(model_7_probabilities, axis=1)

model_7_results = calculate_results(val_labels_encoded, model_7_predictions)
print("\nModel 7 Results:")
pprint(model_7_results)

models = [7]
model_summaries = pd.DataFrame({
    eval("M" + str(num) + "_NAME"): eval("model_" + str(num) + "_results")
    for num in models})
model_summaries = model_summaries.transpose()
model_summaries["accuracy"] = model_summaries["accuracy"] / 100.

print(model_summaries)
model_summaries.to_csv("../models/model_exercises_summaries.csv", index=False)
