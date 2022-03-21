import tensorflow as tf
import datetime
import tensorflow_datasets as tfds
import math
from tqdm import tqdm

def random_data():
    
    file_path = "logs/test"
    summary_writer = tf.summary.create_file_writer(file_path)

    loss_function = tf.keras.losses.MeanSquaredError()

    for i in range(100):
        
        # compute loss (here targets and predictions would come from the data and the model)
        targets = tf.constant([0.3,0.3,-0.8])
        predictions = targets + tf.random.normal(shape=targets.shape, stddev=100/(i+1)) # decreasing noise
        
        loss = loss_function(targets,predictions)
        
        # image batch (these would be obtained from the model)
        
        image_batch = tf.random.uniform(shape=(32,28,28,1),dtype=tf.float32)
        
        
        # audio batch (would be obtained from the model but here it's just a hard coded sine wave of 110hz)
        
        x = 2* math.pi*tf.cast(tf.linspace(0,32000*5, 32000*5), tf.float32)*110/32000
        x = tf.expand_dims(x, axis=0) # add batch dimension
        x = tf.expand_dims(x, axis=-1) # add last dimension
        x = tf.repeat(x, 32, axis=0) # repeat to have a batch of 32
        audio_batch = tf.math.sin(x) # obtain sine wave
        
        
        # text (this would be the output of a language model after one training epoch)
        
        text = tf.constant("This is the sampled output of a language model")
        
        
        # histogram (e.g. of activations of a dense layer during training)
        
        activations_batch = tf.random.normal(shape=(32,20,1))
        min_activations = tf.reduce_min(activations_batch, axis=None)
        max_activations = tf.reduce_max(activations_batch, axis=None)
        histogram = tf.histogram_fixed_width_bins(activations_batch, 
                                                value_range=[min_activations, max_activations])
        
        
        # now we want to write all the data to a log-file.
        with summary_writer.as_default():
            
            # save the loss scalar for the "epoch"
            tf.summary.scalar(name="loss", data=loss, step=i)
            
            # save a batch of images for this epoch (have to be between 0 and 1)
            tf.summary.image(name="generated_images",data = image_batch, step=i, max_outputs=32)
            
            # save the batch of audio for this epoch
            tf.summary.audio(name="generated_audio", data = audio_batch, 
                            sample_rate = 32000, step=i, max_outputs=32)
            
            # save the generated text for that epoch
            tf.summary.text(name="generated_text", data = text, step=i)
            
            # save a histogram (e.g. of activations in a layer)
            tf.summary.histogram(name="layer_N_activations", data = histogram, step=i)
    
    
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
    
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                       ]
        
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)   
        
        L2_lambda = 0.01
        dropout_amount = 0.5
        
        self.all_layers = [
            
            tf.keras.layers.Conv2D(filters=32, 
                                   kernel_size=5, 
                                   strides=1, 
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform,
                                   activation=None,
                                   kernel_regularizer=tf.keras.regularizers.L2(L2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            
            tf.keras.layers.MaxPool2D(pool_size=2,strides=1),
            
            tf.keras.layers.Dropout(dropout_amount),
            
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",activation=None,
                                  kernel_initializer=tf.keras.initializers.glorot_uniform,
                                   kernel_regularizer=tf.keras.regularizers.L2(L2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(L2_lambda)),
            tf.keras.layers.Activation(tf.nn.relu),
            
            tf.keras.layers.Dropout(dropout_amount),
            
            tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.L2(L2_lambda)),
        ]
    
    def call(self, x, training=False):

        for layer in self.all_layers:
            try:
                x = layer(x,training)
            except:
                x = layer(x)
       
        return x
    
    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
            
    @tf.function
    def train_step(self, data):
        
        x, targets = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(targets,predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        x, targets = data
        
        predictions = self(x, training=False)
        
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        self.metrics[0].update_state(loss)
        
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}
    

def augment(x):
    return data_augmentation_model(x)

if __name__ == "__main__":
    ds = tfds.load("fashion_mnist", as_supervised=True)

    train_ds = ds["train"]
    val_ds = ds["test"]

    data_augmentation_model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.Resizing(32,32),
        tf.keras.layers.RandomCrop(28,28)
    ])


    train_ds = train_ds.map(lambda x,y: (augment(x)/255, tf.one_hot(y, 10, dtype=tf.float32)),\
                            num_parallel_calls=tf.data.AUTOTUNE).shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda x,y: (x/255, tf.one_hot(y, 10, dtype=tf.float32)),\
                    num_parallel_calls=tf.data.AUTOTUNE).shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)
    # instantiate the model
    model = CNN()

    # run model on input once so the layers are built
    model(tf.keras.Input((28,28,1)));
    # Define where to save the log

    hyperparameter_string= "test"
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
        
        for data in tqdm(train_ds,position=0, leave=True):
            metrics = model.train_step(data)
        
        # print the metrics
        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        
        # logging the validation metrics to the log file which is used by tensorboard
        with train_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()
        
        
        # Validation:
        
        for data in val_ds:
            metrics = model.test_step(data)
        
        print([f"val_{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        
        # logging the validation metrics to the log file which is used by tensorboard
        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
        # reset all metrics
        model.reset_metrics()
        
        print("\n")