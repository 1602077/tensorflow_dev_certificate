import io
import re
import string
import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout = open("word2vec.log", "w")

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Vectorise an example sentence
sentence = "The wide road shimmered in the hot sun" 
tokens = list(sentence.lower().split())
print(f"Example sentence: {sentence}")
print(f"No. of tokens: {len(tokens)}\n")

# create a vocab to save mappings
vocab, index = {}, 1
vocab['<pad>'] = 0
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1
    vocab_size = len(vocab)
print(f"Vocab: {vocab}")

# inverse vocab to save mappings from int indx to tokens
inverse_vocab = {index: token for token, index in vocab.items()}
print(f"Inverse_vocab: {inverse_vocab}\n")

example_sequence = [vocab[word] for word in tokens]
print(f"Vectorised Sentence: {example_sequence}")

# Generate skipgrams from one sentence
window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sequence,
    vocabulary_size=vocab_size,
    window_size=window_size,
    negative_samples=0)

print(f"\nNumber of skipgrams: {len(positive_skip_grams)}")
print("Some example skipgrams:")
for target, context in positive_skip_grams[:5]:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

# Negative sample for one skip-gram
# skipgrams func returns all positive skip gram pairs by sliding over a given
# window span, get additional samples by randomly sampling worsd from vocab.
# negative sampling means that for each training sample only a small % of weights
# are updated as otherwise there would bea tremendous amount of weights

# Get target and context words for one positive skip-gram.
target_word, context_word = positive_skip_grams[0]

# Set the number of negative samples per positive context.
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,  # class that should be sampled as 'positive'
    num_true=1,  # each positive skip-gram has 1 positive context class
    num_sampled=num_ns,  # number of negative context words to sample
    unique=True,  # all the negative samples should be unique
    range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
    seed=SEED,  # seed for reproducibility
    name="negative_sampling"  # name of this operation
)
#print(negative_sampling_candidates)
print(f"\nNegative sampling candidates: {[inverse_vocab[index.numpy()] for index in negative_sampling_candidates]}")

# Add a dimension so you can use concatenation (on the next step).
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

# Concat positive context word with negative sampled words.
context = tf.concat([context_class, negative_sampling_candidates], 0)

# Label first context word as 1 (positive) followed by num_ns 0s (negative).
label = tf.constant([1] + [0]*num_ns, dtype="int64")

# Reshape target to shape (1,) and context and label to (num_ns+1,).
target = tf.squeeze(target_word)
context = tf.squeeze(context)
label = tf.squeeze(label)

print(f"target_index    : {target}")
print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}\n")

print("target  :", target)
print("context :", context)
print("label   :", label)

# Compile all stesp into one func
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
print(f"\nSampling table:\n{sampling_table}")
# sampling_table[i] denotes the probabilty of sampling the i'th most common
# word in a dataset.

# GENERATE TRAINING DATA

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):
    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=SEED,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


# PREPARE TRAINING DATA FOR WORD2VEC
shakespeare_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
path_to_file = tf.keras.utils.get_file('shakespeare.txt', shakespeare_url)

print("\nShakespeare dataset (fist 20 lines):")
with open(path_to_file) as f:
    lines = f.read().splitlines()
for line in lines[:20]:
    print(line)

# use non empty lines to constructt a TextLineDataset object
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# vectorise as in the wordEmbedding demo
# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

vocab_size = 4096
sequence_length = 10

vectorize_layer = layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(text_ds.batch(1024))

inverse_vocab = vectorize_layer.get_vocabulary()
print(f"\nInverse vocab[:20]: {inverse_vocab[:20]}")

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
print(f"\nNumber of sequences: {len(sequences)}")

for seq in sequences[:5]:
  print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

# Use sequences to generate our training data
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

# Configure datset for performance
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

# CREATE MODEL

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    word_emb = self.target_embedding(target)
    context_emb = self.context_embedding(context)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    return dots

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

history = word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback], verbose=2)

def plot_history(history):
    """
    Plots accuracy and loss curves for a tf history object
    """
    data = pd.DataFrame(history.history)

    plt.figure(figsize=(10,7))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Loss Curves")
    data[["loss"]].plot(xlabel="epochs", ylabel="loss", ax=axes[0])
    data[["accuracy"]].plot(xlabel="epochs", ylabel="accuracy", ax=axes[1])
    plt.tight_layout()

    return plt.savefig('lossCurves.png', bbox_inches='tight', dpi=250)

plot_history(history)

