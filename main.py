import tensorflow as tf
import data_preparation
import model_import
import model
import numpy as np
from tqdm import tqdm
import math
import datetime
from discriminative_fine_tuning import get_optimizers
import tensorflow_addons as tfa






train_text, train_label, test_text, test_label, vali_text, vali_label = data_preparation.get_dataset()

lm_num, encoder_num, mask_num, spm_encoder_model= model_import.get_pretrained_model(256)

train_text = spm_encoder_model(tf.constant(train_text, dtype=tf.string))
test_text = spm_encoder_model(tf.constant(test_text, dtype=tf.string))
vali_text = spm_encoder_model(tf.constant(vali_text, dtype=tf.string))

pre = model_import.get_list_of_layers(encoder_num)

pre.append(tf.keras.layers.Dense(16, activation='relu'))
pre.append(tf.keras.layers.Flatten())
pre.append(tf.keras.layers.Dense(1))


model = tf.keras.Sequential(pre)

#encoder_num.trainable = False

optimizers_and_layers = get_optimizers(layers=pre, num_epochs=5, num_updates_per_epoch=316)

#print(optimizers_and_layers)

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)



train_label = train_label.astype('int32')
test_label = test_label.astype('int32')
vali_label = vali_label.astype('int32')


train_data = tf.data.Dataset.from_tensor_slices((train_text, train_label))
test_data = tf.data.Dataset.from_tensor_slices((test_text, test_label))
vali_data = tf.data.Dataset.from_tensor_slices((vali_text, vali_label))

model.summary()

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=5,
                    validation_data=vali_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):

    print("%s: %.3f" % (name, value))

'''train_text, train_label, test_text, test_label, vali_text, vali_label = data_preparation.get_dataset()



lm_num, encoder_num, mask_num, spm_encoder_model= model_import.get_pretrained_model(256)

train_text = spm_encoder_model(tf.constant(train_text, dtype=tf.string))
test_text = spm_encoder_model(tf.constant(test_text, dtype=tf.string))
vali_text = spm_encoder_model(tf.constant(vali_text, dtype=tf.string))

lm_num, encoder_num, mask_num, spm_encoder_model= model_import.get_pretrained_model(256)

train_text = spm_encoder_model(tf.constant(train_text, dtype=tf.string))
test_text = spm_encoder_model(tf.constant(test_text, dtype=tf.string))
vali_text = spm_encoder_model(tf.constant(vali_text, dtype=tf.string))

train_label = train_label.astype('int32')
test_label = test_label.astype('int32')
vali_label = vali_label.astype('int32')


train_ds = tf.data.Dataset.from_tensor_slices((train_text, train_label))
test_ds = tf.data.Dataset.from_tensor_slices((test_text, test_label))
val_ds = tf.data.Dataset.from_tensor_slices((vali_text, vali_label))



# Define where to save the log
hyperparameter_string= "Your_Settings_Here"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{hyperparameter_string}/{current_time}/train"
val_log_path = f"logs/{hyperparameter_string}/{current_time}/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)

# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)

for epoch in range(5):
    
    print(f"Epoch {epoch}:")
    
    # Training:
    
    for data in tqdm.notebook.tqdm(train_ds,position=0, leave=True):
        metrics = model.train_step(data)
    
    # print the metrics
    print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
    
#predictions = model.predict(train_tokens)
#predictions = [1 if p > 0.5 else 0 for p in predictions]
'''

