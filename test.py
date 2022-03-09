import tensorflow as tf
import tensorflow_hub as hub
'''
model = hub.KerasLayer("https://tfhub.dev/google/wiki40b-lm-en/1")

embeddings = model(["The rain in Spain.", "falls",
                    "mainly", "In the plain!"])

print(embeddings.shape)  '''

#model = tf.keras.models.load_model("fwd_wt103.h5")
#print(model.summary)




hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2",
                           input_shape=[], dtype=tf.string)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

