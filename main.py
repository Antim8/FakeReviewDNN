import tensorflow as tf
import data_preparation
import model_import


train_text, train_label, test_text, test_label, vali_text, vali_label = data_preparation.get_dataset()

lm_num, encoder_num, mask_num, spm_encoder_model= model_import.get_pretrained_model(256)

train_text = spm_encoder_model(tf.constant(train_text, dtype=tf.string))
test_text = spm_encoder_model(tf.constant(test_text, dtype=tf.string))
vali_text = spm_encoder_model(tf.constant(vali_text, dtype=tf.string))



model = tf.keras.Sequential([
     encoder_num,
     tf.keras.layers.Dense(16, activation='relu'),
     tf.keras.layers.Flatten(), # check 
     tf.keras.layers.Dense(1)
])

encoder_num.trainable = False



train_label = train_label.astype('int32')
test_label = test_label.astype('int32')
vali_label = vali_label.astype('int32')


train_data = tf.data.Dataset.from_tensor_slices((train_text, train_label))
test_data = tf.data.Dataset.from_tensor_slices((test_text, test_label))
vali_data = tf.data.Dataset.from_tensor_slices((vali_text, vali_label))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=5,
                    validation_data=vali_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))