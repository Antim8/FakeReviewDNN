from cgi import test
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
import pandas as pd
import data_preparation



df = data_preparation.get_dataframe()
df['label'] = df['label'].astype(np.int64)

ds = tf.data.Dataset.from_tensor_slices(
    (tf.cast(df["text"].values, tf.string),
    tf.cast(df["label"].values, tf.int32))
)

train_size = int(0.7 * len(ds))


train_data = ds.take(train_size)
test_data = ds.skip(train_size)
#validation_data = test_data.skip(val_size)
#test_data = test_data.take(test_data)


embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
                           
model = Sequential()
model.add(hub_layer)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=5,
                    validation_data=test_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))