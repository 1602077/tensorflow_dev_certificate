import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pprint import pprint
from preprocess_data import get_lines, preprocess_text_with_line_numbers, create_dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
import random


sys.stdout = open("../logs/visualisingData.log", "w")

data_dir = "../data/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

filenames = [data_dir + filename for filename in os.listdir(data_dir)]

print(f"Current Working Directory: {data_dir}")
print(f"Files:")
for file in filenames:
    print(file)

train_lines = get_lines(data_dir + "train.txt")
print(f"\nFirst 20 lines in 'train.txt':\n{train_lines[:20]}")
print(f"\nNumber of lines in train_lines: {len(train_lines)}")

train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt")
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")

print(f"\nLength of train, validation, and test datasets:")
print(f"{len(train_samples)}, {len(val_samples)}, {len(test_samples)}")
print(f"\nFirst 10 lines in train_samples:")
pprint(train_samples[:10])

train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)

print("\nTrain dataset pandas dataframe head:")
print(train_df.head(14))

# Distribution of labels in training data
print(f"\nDistribution of labels in training data:\n{train_df.target.value_counts()}")
train_df.target.value_counts().plot(kind="barh")
plt.xlabel("Label Count")
plt.tight_layout()
plt.savefig("../logs/1_CountLabels.png", dpi=200)

plt.figure()
train_df.total_lines.plot.hist(bins=25)
plt.xlabel("Sentences per Abstract")
plt.savefig("../logs/2_LineLengthDistribution.png", dpi=200, bbox_inches="tight")

# Get lists of sentences
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

# Make numeric labels
one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

# Extract labels and encode them into ints
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

print(f"\nEncoded labels example (5): {train_labels_encoded[:5]}")

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
print(f"Number of classes: {num_classes}; Classes: {class_names}")

data = create_dataset(
    data_dir + "train.txt",
    data_dir + "dev.txt",
    data_dir + "test.txt"
)
train_df, train_labels, val_df, val_labels, test_df, test_labels, num_classes, class_names = data
print(train_df.head())

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
print(f"\n{50*'-'}\n  Create text vectorisation layer\n{50*'-'}")
max_tokens = 68000

text_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_seq_len)
text_vectorizer.adapt(train_sentences)

target_sentence = random.choice(train_sentences)
print(f"""
Random sentence:\n{target_sentence}
Length of text: {len(target_sentence)}
Vectorised text:\n {text_vectorizer([target_sentence])}
""")

rct_20k_text_vocab = text_vectorizer.get_vocabulary()
print(f"""
Number of words in vocab: {len(rct_20k_text_vocab)}
Most common words in vocab: {rct_20k_text_vocab[:5]}
Least common words in vocab: {rct_20k_text_vocab[-5:]}
""")

print(f"Text vectoriser config:\n")
pprint(text_vectorizer.get_config())

# Create custom text embedding
token_embed = layers.Embedding(input_dim=len(rct_20k_text_vocab), output_dim=128, mask_zero=True,
                               name="token_embedding")
vectorized_sentence = text_vectorizer([target_sentence])
embedded_sentence = token_embed(vectorized_sentence)
print(f"""
Sentence before vectorisation: {target_sentence}\n
Sentence after vectorisation (before embedding):
{vectorized_sentence}\n
Embedding sentence:
{embedded_sentence}\n
Embedded sentence shape: {embedded_sentence.shape}
""")
