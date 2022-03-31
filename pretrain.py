import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as tf_text

#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

#lm = hub.load("https://tfhub.dev/google/wiki40b-lm-en/1")

embedding = "https://tfhub.dev/google/universal-sentence-encoder/4"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

print(hub_layer)

model = tf.keras.Sequential()
model.add(hub_layer)
print(model.summary())