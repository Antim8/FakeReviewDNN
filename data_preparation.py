from cgi import test
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter


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
    

    # The review 
    text = pd.read_csv("fake_reviews_dataset.csv", usecols=[3]).values
    text = np.squeeze(text, axis=1)

    df = pd.DataFrame(columns=['text','label'])
    df['text'] = text

    #label = np.expand_dims(label, axis=1)

    #print(label)

    df['label'] = label


    # Shuffle the data, since it is sorted by categories
    df = df.sample(frac=1)

    

    training_set_size = int(df.shape[0]*0.5)

    train_df = df[:training_set_size]
    test_df = df[training_set_size:]

    validation_set_size = int(test_df.shape[0]*0.5)

    vali_df = test_df[:validation_set_size]
    test_df = test_df[validation_set_size:]
    

    train_text = train_df.text.to_numpy()
    train_label = train_df.label.to_numpy()

    test_text = test_df.text.to_numpy()
    test_label = test_df.label.to_numpy()

    vali_text = vali_df.text.to_numpy()
    vali_label = vali_df.label.to_numpy()

    train_label = np.expand_dims(train_label, axis=1)
    test_label = np.expand_dims(test_label, axis=1)
    vali_label = np.expand_dims(vali_label, axis=1)

    return train_text, train_label, test_text, test_label, vali_text, vali_label

def get_dataframe():

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

    return df

def tokenizer(train_text_data, test_text_data, oov_token='<UNK>', maxlen=None, padding='post'):

    if maxlen == None:
        maxlen = max(len(x) for x in train_text_data)

    num_words = len(get_counts(train_text_data))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words, oov_token=oov_token)
    tokenizer.fit_on_sequences(train_text_data)

    train_text_seq = tokenizer.texts_to_sequences(train_text_data)
    train_text_seq = tf.keras.preprocessing.sequence.pad_sequences(train_text_seq, maxlen=maxlen, padding=padding)
    
    test_text_seq = tokenizer.texts_to_sequences(test_text_data)
    test_text_seq = tf.keras.preprocessing.sequence.pad_sequences(test_text_seq, maxlen=maxlen, padding=padding)
    
    return train_text_seq, test_text_seq, maxlen, num_words

def data_pipeline(ds, shuffle=1000, batch=64, prefetch=20):
    ds = ds.shuffle(shuffle)
    ds = ds.batch(batch)
    ds = ds.prefetch(prefetch)

    return ds

def get_counts(text_data):

    count = Counter()
    for text in text_data:
        for word in text.split():
            count[word] += 1
        return count


    
#train_text, train_label, test_text, test_label, vali_text, vali_label = get_dataset()

#print(train_label.shape)