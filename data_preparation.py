from cProfile import label
from random import shuffle
from re import I
import numpy as np
import tensorflow as tf
import pandas as pd


def get_dataset():

    # label for fake or real review
    # CG -> Computer-generated fake reviews; OR = Original reviews
    label = pd.read_csv("fake_reviews_dataset.csv", usecols=[2]).values

    # CG -> 1; OR -> 0
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
    # Shuffle the data, since it is sorted by categories
    df = df.sample(frac=1)

    text_data = tokenizer(df['text'])

    ds = tf.data.Dataset.from_tensor_slices((text_data, df['label'].values.astype(np.int32)))

    return ds

def tokenizer(text_data, num_words=8000, oov_token='<UNK>', maxlen=None, padding='post'):

    if maxlen == None:
        maxlen = max(len(x) for x in text_data)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=8000, oov_token=oov_token)
    tokenizer.fit_on_texts(text_data)

    text_seq = tokenizer.texts_to_sequences(text_data)
    text_seq = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=maxlen, padding=padding)
    
    return text_seq

def data_pipeline(ds, shuffle=1000, batch=32, prefetch=20):
    ds = ds.shuffle(shuffle)
    ds = ds.batch(batch)
    ds = ds.prefetch(prefetch)

    return ds
    

