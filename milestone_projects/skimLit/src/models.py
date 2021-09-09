import pandas as pd
from preprocess_data import create_dataset, create_tf_datasets, preprocess_text_with_line_numbers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_hub as hub
from tensorflow.keras.utils import plot_model
import random
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from helper_functions import calculate_results
from pprint import pprint
import matplotlib.pyplot as plt
import string

pd.set_option('display.max_columns', None, 'display.max_rows', None)
sys.stdout = open("../logs/models.log", "w")

# LOAD DATA
data_dir = "../data/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
datasets = create_dataset(
    data_dir + "train.txt",
    data_dir + "dev.txt",
    data_dir + "test.txt",
    one_hot=False
)
train_df, train_labels, val_df, val_labels, test_df, test_labels, num_classes, class_names = datasets
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

# BASELINE MODEL: TF-IDF Multinomial Naive Bayes Classifier
models = []
M0_NAME = "Model0_TF-IDF-Multinomial-Naive-Bayes"
models.append(M0_NAME)
print(f"{50*'-'}\n  {M0_NAME}\n{50*'-'}")
model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()),
    ("clf", MultinomialNB()),
])

print(model_0)

model_0.fit(X=train_sentences, y=train_labels)

baseline_preds = model_0.predict(val_sentences)
model_0_results = calculate_results(y_true=val_labels, y_pred=baseline_preds)

print(f"\nBaseline Model Results")
pprint(model_0_results, indent=2)

print(f"\n{50*'-'}\n  Preparing data for deep learning\n{50*'-'}")
sent_lens = [len(sentence.split()) for sentence in train_sentences]
avg_sent_len = np.mean(sent_lens)
print(f"Average length of each sentence: {avg_sent_len:.2f}")
plt.hist(sent_lens, bins=20)
plt.savefig('../logs/3_SentenceDistribution.png', bbox_inches='tight', dpi=200)
print("Most sentences are around ~25 words long.")
# what length for padding would cover 95% of examples
output_seq_len = int(np.percentile(sent_lens, 95))
print(f"95% percentile sentence length: {output_seq_len}")
print(f"Maximum sentence length in training set: {max(sent_lens)}")

# CREATE TEXT VECTORISER LAYER
max_tokens = 68000

text_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_seq_len)
text_vectorizer.adapt(train_sentences)

rct_20k_text_vocab = text_vectorizer.get_vocabulary()

# Create custom text embedding
token_embed = layers.Embedding(input_dim=len(rct_20k_text_vocab), output_dim=128, mask_zero=True,
                               name="token_embedding")

# Creating fast loading datasets
train_data, val_data, test_data = create_tf_datasets(data_dir)

# Model 1: Conv-1D

M1_NAME = "Model1_Conv_1D"
models.append(M1_NAME)

print(f"\n{50*'-'}\n  {M1_NAME}\n{50*'-'}")

inputs = layers.Input(shape=(1,), dtype=tf.string)
text_vectors = text_vectorizer(inputs)
token_embeddings = token_embed(text_vectors)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model_1 = tf.keras.Model(inputs, outputs, name=M1_NAME)
model_1.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model_1.summary()
history_model_1 = model_1.fit(
    train_data,
    steps_per_epoch=int(0.1*len(train_data)),
    epochs=3,
    validation_data=val_data,
    validation_steps=int(0.1 * len(val_data)),
    verbose=2
)
model_1_pred_probs = model_1.predict(val_data)
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)
model_1_results = calculate_results(val_labels, model_1_preds)
print("\nModel 1 Results:")
pprint(model_1_results)


# Building USE feature extractor
print(f"\n{50*'-'}\n  Creating embeddings using USE feature extractor from tf_hub\n{50*'-'}")
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")

random_train_sentence = random.choice(train_sentences)
print(f"Random sentence: {random_train_sentence}")
use_embedded_sentence = tf_hub_embedding_layer([random_train_sentence])
print(f"Sentence after embedding (first 30 values only):\n {use_embedded_sentence[0][:30]}\n")
print(f"Length of sentence embedding: {len(use_embedded_sentence[0])}")


# Model 2: Feature Extractor with Embedding layers
M2_NAME = "Model2_USE_FeatExtr"
models.append(M2_NAME)
print(f"\n{50*'-'}\n  {M2_NAME}\n{50*'-'}")
inputs = layers.Input(shape=[], dtype=tf.string)
pretrained_emebdding = tf_hub_embedding_layer(inputs)
x = layers.Dense(128, activation="relu")(pretrained_emebdding)
outputs = layers.Dense(5, activation="softmax")(x)

model_2 = tf.keras.Model(inputs, outputs, name=M2_NAME)
model_2.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model_2.summary()
history_model_2 = model_2.fit(
    train_data,
    steps_per_epoch=int(0.1*len(train_data)),
    epochs=3,
    validation_data=val_data,
    validation_steps=int(0.1 * len(val_data)),
    verbose=2
)
model_2_pred_probs = model_2.predict(val_data)
model_2_preds = tf.argmax(model_2_pred_probs, axis=1)
model_2_results = calculate_results(val_labels, model_2_preds)
print("\nModel 2 Results:")
pprint(model_2_results)

print(f"\n{50*'-'}\n  Creating character embeddings\n{50*'-'}")


def split_chars(text):
    """ Split sentences into characters"""
    return " ".join(list(text))


# split sequence level data splits into char-level data split
train_chars = [split_chars(sent) for sent in train_sentences]
val_chars = [split_chars(sent) for sent in val_sentences]
test_chars = [split_chars(sent) for sent in test_sentences]

char_lens = [len(sent) for sent in train_sentences]
mean_char_len = np.mean(char_lens)
print(f"Average character length: {mean_char_len}")
plt.hist(char_lens, bins=10)
plt.savefig('../logs/4_CharDistribution.png', bbox_inches='tight', dpi=200)
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
train_filename, val_filename, test_filename = data_dir + "train.txt", data_dir + "dev.txt", data_dir + "test.txt"
train_df = pd.DataFrame(preprocess_text_with_line_numbers(train_filename))
val_df = pd.DataFrame(preprocess_text_with_line_numbers(val_filename))
test_df = pd.DataFrame(preprocess_text_with_line_numbers(test_filename))

one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())


# optimise datasets for performance
train_char_data = tf.data.Dataset.from_tensor_slices((train_chars, train_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
val_char_data = tf.data.Dataset.from_tensor_slices((val_chars, val_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
test_char_data = tf.data.Dataset.from_tensor_slices((test_chars, test_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)

# Model 3: Conv1D w/ Character Embedding
M3_NAME = "Model3_Conv1D_CharEmbed"
models.append(M3_NAME)

print(f"\n{50*'-'}\n  {M3_NAME}\n{50*'-'}")

inputs = layers.Input(shape=(1,), dtype="string")
char_vec = char_vectorizer(inputs)
char_embeddings = char_embed(char_vec)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(char_embeddings)
x = layers.GlobalMaxPool1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model_3 = tf.keras.Model(inputs, outputs, name=M3_NAME)

model_3.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model_3.summary()

history_model_3 = model_3.fit(
    train_char_data,
    steps_per_epoch=int(0.1*len(train_char_data)),
    epochs=3,
    validation_data=val_char_data,
    validation_steps=int(0.1 * len(val_char_data)),
    verbose=2
)
model_3_pred_probs = model_3.predict(val_char_data)
model_3_preds = tf.argmax(model_3_pred_probs, axis=1)

model_3_results = calculate_results(val_labels_encoded, model_3_preds)
print("\nModel 3 Results:")
pprint(model_3_results)

# Model 4: Combined pretrained token embeddings and character embeddings (multi-modal model)
M4_NAME = "Model4_TokenAndCharEmbed"
models.append(M4_NAME)
print(f"\n{50*'-'}\n  {M4_NAME}\n{50*'-'}")

# 1. Setup token inputs/model
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs, outputs=token_outputs, name="M4_TokenModel")

# 2. Create Character level model
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm, name="M4_CharModel")

# 3. Concat models together (i.e. create a hybird token embedding)
token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_model.output, char_model.output])

# 4. Build series of output layers on top of 3
combined_dropout = layers.Dropout(0.5)(token_char_concat)
combined_dense = layers.Dense(128, activation="relu")(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output = layers.Dense(num_classes, activation="softmax")(final_dropout)

# 5. Construct model with char and token inputs
model_4 = tf.keras.Model(inputs=[token_model.input, char_model.input], outputs=output, name=M4_NAME)
model_4.summary()
plot_model(model_4, to_file="../logs/model4_summary.png", show_shapes=True)

model_4.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

# Combining token and character data into a tf.data dataset
train_char_token_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars))
train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels))
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_char_token_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars))
val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

test_char_token_data = tf.data.Dataset.from_tensor_slices((test_sentences, test_chars))
test_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_char_token_dataset = tf.data.Dataset.zip((test_char_token_data, test_char_token_labels))
test_char_token_dataset = test_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print("Check prefetch train and token datasets")
print(train_char_token_dataset)
print(val_char_token_dataset)

history_model_4 = model_4.fit(
    train_char_token_dataset,
    steps_per_epoch=int(0.1*len(train_char_token_dataset)),
    epochs=3,
    validation_data=val_char_token_dataset,
    validation_steps=int(0.1 * len(val_char_token_dataset)),
    verbose=2
)

model_4_pred_probs = model_4.predict(val_char_token_dataset)
model_4_preds = tf.argmax(model_4_pred_probs, axis=1)

model_4_results = calculate_results(val_labels_encoded, model_4_preds)
print("\nModel 4 Results:")
pprint(model_4_results)

# Model 5: Tri_Embedding Model - Sentence, Vector & Positional Embeddings
M5_NAME = "Model5_Tri_Embed_Model"
models.append(M5_NAME)
print(f"\n{50*'-'}\n  {M5_NAME}\n{50*'-'}")

# Create positional embeddings
print("Number of different line numbers")
train_df["line_number"].value_counts()
plt.figure()
train_df.line_number.plot.hist(xlabel="Number of lines")
plt.savefig("../logs/5_line_number_distribution.png", bbox_inches="tight", dpi=250)

# Use tf to create one-hot-encoded tensors of our "line_number" column
train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)
print(f"Train line numbers one-hot [:10]:\n{train_line_numbers_one_hot[:10]}\nShape: {train_line_numbers_one_hot.shape}")

# one_hot our total lines feature
plt.figure()
train_df["line_number"].plot.hist(xlabel="Total line number")
plt.savefig("../logs/6_total_line_num_dist.png", bbox_inches="tight", dpi=250)
print(f"Length coverage 95% of total_lienes: {np.percentile(train_df.total_lines, 95)}")
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)
print(f"Train total line numbers one-hot [:10]:\n{train_total_lines_one_hot[:10]}\nShape: {train_total_lines_one_hot.shape}")


# Building the tribrid embedding model
# 1. Token inputs
token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(token_inputs, token_outputs, name="M5_token_model")

# 2. Char inputs
char_inputs = layers.Input(shape=(1,), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
char_model = tf.keras.Model(char_inputs, char_bi_lstm, name="M5_bi_lstm")

# 3. Line number inputs
line_num_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_num_inputs")
line_num_dense = layers.Dense(32, activation="relu")(line_num_inputs)
line_num_model = tf.keras.Model(line_num_inputs, line_num_dense, name="M5_line_num")

# 4. Total line number inputs
total_num_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="total_num_inputs")
x = layers.Dense(32, activation="relu")(total_num_inputs)
total_num_model = tf.keras.Model(total_num_inputs, x, name="M5_total_num")

# 5. Combine token & char embed into hybrid embedding
combined_embedding = layers.Concatenate(name="M5_char_token_hybrid_embedding")([token_model.output, char_model.output])
combined_embedding = layers.Dense(256, activation="relu")(combined_embedding)
combined_embedding = layers.Dropout(0.5)(combined_embedding)

# 6. Combine positional embedding with combined token and char embeddings
tribrid_embeddings = layers.Concatenate(name="M5_char_token_pos_embed")([line_num_model.output,
                                                                         total_num_model.output,
                                                                         combined_embedding])
# 7. Create output layer
output_layer = layers.Dense(5, activation="softmax", name="output_layer")(tribrid_embeddings)

# 8. Piece together model
model_5 = tf.keras.Model(inputs=[line_num_model.input, total_num_model.input, token_model.input, char_model.input],
                         outputs=output_layer,
                         name=M5_NAME)

model_5.summary()
plot_model(model_5, to_file="../logs/model5_summary.png", show_shapes=True)

# Compile token, char and positional mebedding model
model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Create tribrid embedding datasets into a tf.data.Dataset
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


history_model_5 = model_5.fit(
    train_char_token_pos_dataset,
    steps_per_epoch=int(0.1*len(train_char_token_pos_dataset)),
    epochs=3,
    validation_data=val_char_token_pos_dataset,
    validation_steps=int(0.1 * len(val_char_token_pos_dataset)),
    verbose=2
)

model_5_pred_probs = model_5.predict(val_char_token_pos_dataset)
model_5_preds = tf.argmax(model_5_pred_probs, axis=1)

model_5_results = calculate_results(val_labels_encoded, model_5_preds)
print("\nModel 5 Results:")
pprint(model_5_results)

# Model 6: Tribrid Model trained on full dataset
M6_NAME = "Model6_TribridModel_FullDataset"
models.append(M6_NAME)
print(f"\n{50*'-'}\n  {M6_NAME}\n{50*'-'}")

model_6 = tf.keras.models.clone_model(model_5)
model_6._name = M6_NAME

model_6.summary()

model_6.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_model_6 = model_6.fit(
    train_char_token_pos_dataset,
    steps_per_epoch=len(train_char_token_pos_dataset),
    epochs=3,
    validation_data=val_char_token_pos_dataset,
    validation_steps=len(val_char_token_pos_dataset),
    verbose=2
)

model_6_pred_probs = model_6.predict(val_char_token_pos_dataset)
model_6_preds = tf.argmax(model_6_pred_probs, axis=1)

model_6_results = calculate_results(val_labels_encoded, model_6_preds)
print("\nModel 6 Results:")
pprint(model_6_results)

# Comparing Model Performances
print(f"\n{50*'-'}\n  MODEL PERFORMANCE SUMMARIES\n{50*'-'}\n")
model_summaries = pd.DataFrame({
    eval("M" + str(num) + "_NAME"): eval("model_" + str(num) + "_results")
    for num in range(len(models))})
model_summaries = model_summaries.transpose()
model_summaries["accuracy"] = model_summaries["accuracy"] / 100.

print(model_summaries)
model_summaries.to_csv("../models/model_summaries.csv", index=False)

# plot all performance metrics
model_summaries.plot(kind="barh", figsize=(10, 7), ylabel="metrics").legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig("../models/model_metrics.png", bbox_inches="tight", dpi=200)

# Save all models
for j in range(len(models)):
    model = eval("model_" + str(j))
    model_name = eval("M" + str(j) + "_NAME")
    model.save("../models/model_name")
