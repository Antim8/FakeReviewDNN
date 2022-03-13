# Das erstmal ignorieren (see model_import.py)

import tensorflow as tf
#import tensorflow_text as text
import tensorflow_hub as hub
import data_preparation as dp
import tensorflow_datasets as tfds

print(tf.__version__)

tfds.disable_progress_bar()

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

hub_layer = hub.KerasLayer("https://tfhub.dev/google/elmo/3",
                        input_shape=[], dtype=tf.string)

model = tf.keras.Sequential()

#model.add(tf.keras.layers.Embedding(
#    input_dim = len(encoder.get_vocabulary()),
###    output_dim = 64,
#    mask_zero=True
# ````))
model.add(hub_layer)
#model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
#model.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(tf.keras.layers.Dense(1))

print(model.summary())


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)




