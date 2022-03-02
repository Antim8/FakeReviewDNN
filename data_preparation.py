from cProfile import label
from random import shuffle
import numpy as np
import tensorflow as tf
import pandas as pd

# label for fake or real review
label = pd.read_csv("fake_reviews_dataset.csv", usecols=[2]).values

for [i], j in zip(label, range(len(label))):
    if i == "CG":
        label[j] = 1
    else:
        label[j] = 0
label = np.squeeze(label, axis=1)

# The review 
text = pd.read_csv("fake_reviews_dataset.csv", usecols=[3]).values
text = np.squeeze(text, axis=1)

df = pd.DataFrame(columns=['text','label'])

df['text'] = text
df['label'] = label

# Um die daten zu mischen (geht das?)
df = df.sample(frac=1)

#Only train and test dataset so far, (maybe validation?)
#Evtl. muss vorher auch geshuiffelt werden wegen den categrories 
test_df = df[int(len(text)/2):]
train_df = df[:int(len(text)/2)]


# Tokenizing 
num_words = 8000
oov_token = '<UNK>'
maxlen = max(len(x) for x in train_df['text'])
padding = 'post'

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=8000, oov_token=oov_token)
tokenizer.fit_on_texts(train_df['text'])

train_seq = tokenizer.texts_to_sequences(train_df['text'])
test_seq = tokenizer.texts_to_sequences(test_df['text'])

train_seq = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=maxlen, padding=padding)
test_seq = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=maxlen, padding=padding)


