import tensorflow as tf
import data_preparation
import model_import
import model
import numpy as np


'''train_text, train_label, test_text, test_label, vali_text, vali_label = data_preparation.get_dataset()

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
    print("%s: %.3f" % (name, value))'''






##############################################################################
############################ SUBCLASSING API #################################


train_text, train_label, test_text, test_label, vali_text, vali_label = data_preparation.get_dataset()

lm_num, encoder_num, mask_num, spm_encoder_model= model_import.get_pretrained_model(256)

train_text = spm_encoder_model(tf.constant(train_text, dtype=tf.string))
test_text = spm_encoder_model(tf.constant(test_text, dtype=tf.string))
vali_text = spm_encoder_model(tf.constant(vali_text, dtype=tf.string))

train_label = train_label.astype('int32')
test_label = test_label.astype('int32')
vali_label = vali_label.astype('int32')

train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_text, test_label))
vali_dataset = tf.data.Dataset.from_tensor_slices((vali_text, vali_label))

train_dataset = data_preparation.data_pipeline(train_dataset)
test_dataset = data_preparation.data_pipeline(test_dataset)
vali_dataset = data_preparation.data_pipeline(vali_dataset)




for i in train_dataset.take(3):
     print(i)


model = model.Classification(encoder_num)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_function = tf.keras.losses.BinaryCrossentropy()

loss_metric = tf.keras.metrics.Mean()



def train_step(model, input, target, loss_function, optimizer):
  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model(input)
    loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  # test over complete test data

  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
   
    
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  np.argmax(target, axis=0) == np.argmax(prediction, axis=0) # vorher war axis=1
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy

### Hyperparameters
num_epochs = 10
learning_rate = 0.001


# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
# Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

#testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)