import io
import os
import sys
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout = open('logs/wordEmbeddingDemo.log', 'w')

# DOWNLOAD DATASET AND UPACK
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# CREATE TRAIN & VAL SETSo
batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

print("\n Example reviews from train dataset:")
for text_batch, label_batch in train_ds.take(1):
  for i in range(5):
    print(label_batch[i].numpy(), text_batch.numpy()[i], '\n')

# CONFIGURE DATASET FOR PERFORMANCE
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# CREATE EMBEDDING LAYER
# 1,00 word vocab into 5 dims
embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1,2,3]))
print(f"Example embedding for vector [1,2,3]:\n{result.numpy()}\n")

def custom_standardization(input_data):
    """
    Standardisation function to strip HTML break tags '<br />'.
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
sequence_length = 100

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build vocab
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# CREATE CLASSIFICATION MODEL
embedding_dim=16

NAME = "Model_WordEmbeddingDemo"

model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1)
], name=NAME)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/"+NAME)

model_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback],
    verbose=2
)

print()
model.summary()

def plot_history(history):
    """
    Plots accuracy and loss curves for a tf history object
    """
    data = pd.DataFrame(history.history)

    plt.figure(figsize=(10,7))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Loss Curves")
    data[["loss", "val_loss"]].plot(xlabel="epochs", ylabel="loss", ax=axes[0])
    data[["accuracy", "val_accuracy"]].plot(xlabel="epochs", ylabel="accuracy", ax=axes[1])
    plt.tight_layout()

    return plt.savefig('lossCurves.png', bbox_inches='tight', dpi=250)

plot_history(model_history)

