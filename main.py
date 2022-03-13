import tensorflow as tf
import data_preparation
import numpy as np
import tensorflow_hub as hub
import model_import
import pandas

seq_length = 70 

train_text, train_label, test_text, test_label = data_preparation.get_dataset()
#pre_model = model_import.get_pretrained_model(seq_length=seq_length)

#train_txt, test_txt, maxlen, num_words = data_preparation.tokenizer(train_text, test_text)

import sentencepiece as spm


#a = spm.SentencePieceProcessor('tf2_ulmfit/enwiki100-toks-sp35k-cased.vocab')
#print(a)

#TODO tokensize with the given vokabulary 
#https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=imfPyYlVZmxz

sp = spm.SentencePieceProcessor()
sp.load('tf2_ulmfit/enwiki100-toks-sp35k-cased.model')


train_text_tokens = sp.encode_as_ids(train_text)
print(train_text_tokens)


'''train_txt, train_label, test_txt, test_label = data_preparation.get_dataset()

train_tokens, test_tokens, max_len, num_unique_words = data_preparation.tokenizer(train_txt, test_txt)

# Create LSTM model
from tensorflow.keras import layers

hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2",
                        input_shape=[], dtype=tf.string)


# Embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
# Turns positive integers (indexes) into dense vectors of fixed size. (other approach could be one-hot-encoding)

# Word embeddings give us a way to use an efficient, dense representation in which similar words have 
# a similar encoding. Importantly, you do not have to specify this encoding by hand. An embedding is a 
# dense vector of floating point values (the length of the vector is a parameter you specify).

model = tf.keras.models.Sequential()
#model.add(layers.Embedding(num_unique_words,output_dim=32 ,input_length=max_len))
model.add(hub_layer)
# The layer will take as input an integer matrix of size (batch, input_length),
# and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
# Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.
#model.add(layers.Reshape(12,))
model.add(layers.Dense(16, activation="relu"))
#model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1,activation="sigmoid"))

model.summary()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

model.fit(train_tokens, train_label.astype(np.int64), epochs=20, validation_data=(test_tokens, test_label.astype(np.int64)), verbose=2)


predictions = model.predict(train_tokens)
predictions = [1 if p > 0.5 else 0 for p in predictions]'''