import tensorflow as tf
import tensorflow_datasets as tfds
import math
import model_import as mi
from tqdm import tqdm
import datetime
import data_preparation
import slanted_triangular_lr
import tensorflow_addons as tfa
from discriminative_fine_tuning import get_optimizers
from tf2_ulmfit.ulmfit_tf2 import apply_awd_eagerly
from tf2_ulmfit.ulmfit_tf2 import ConcatPooler



class Fake_detection(tf.keras.Model):
    def __init__(self, classifier=False):
        super(Fake_detection, self).__init__()

        self.num_epoch = 1
        self.num_updates_per_epoch = 316

        self.classifier = classifier

        seq_length = 256

        self.lm_num, self.encoder_num, _, self.spm_encoder_model = mi.get_pretrained_model(seq_length)

        

        if classifier:
            self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.BinaryAccuracy(name="acc"),
                        #tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                       ]
            self.loss_function = tf.keras.losses.BinaryCrossentropy()  
            pretrained_layers = mi.get_list_of_layers(self.encoder_num)

            pretrained_layers.append(ConcatPooler())
            pretrained_layers.append(tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False))
            pretrained_layers.append(tf.keras.layers.Dropout(0.1))

            additional_layers = tf.keras.Sequential([
                tf.keras.layers.Dense(50, activation='linear', use_bias=False, input_shape=(1200,)),
                tf.keras.layers.ReLU(),
                tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, use_bias=False, activation='sigmoid')
            ])

            for layer in additional_layers.layers:
                pretrained_layers.append(layer)

            num_epochs_list = []

            for num in range(len(pretrained_layers)):
                epoch = self.num_epoch - num
                num_epochs_list.append(epoch)

        else:
            self.metrics_list = [
                            tf.keras.metrics.Mean(name="loss"),
                            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
                            #tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                        ]
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()  
            pretrained_layers, self.spm_encoder_model = mi.prepare_pretrained_model(self.encoder_num, 'new_amazon.model', seq_length)
            num_epochs_list = self.num_epoch

        for layer in pretrained_layers:
            layer.trainable = True

        self.training_list = []
        for _ in range(len(pretrained_layers)):
            self.training_list.append(False)
        self.no_training = self.training_list
        self.training_list_index = len(pretrained_layers) - 1

        self.all_layers = pretrained_layers
        #for layer in self.all_layers:
        #    layer.trainable = True

        # DFF
        self.optimizers_and_layers = get_optimizers(layers=self.all_layers, num_epochs=num_epochs_list, num_updates_per_epoch=self.num_updates_per_epoch)

        self.optimizer = tfa.optimizers.MultiOptimizer(self.optimizers_and_layers)
        #self.optimizer = tf.keras.optimizers.Adam()

        

    def encode(self, text):
        #print(text)

        
        #print(tf.constant(text, dtype=tf.string))
        
        return self.spm_encoder_model(tf.constant(text, dtype=tf.string))
       


    def gradual_unfreezing(self):

        #TODO more layers than epochs error

        self.training_list[self.training_list_index] = True

        temp = self.training_list_index > 0

        self.training_list_index = (self.training_list_index - 1) * temp

    def temp_call_classifier(self, x, training=False):
    
        training = self.training_list * training + self.no_training * (int(training)+1)
        
        

        for i, layer in enumerate(self.all_layers[:10]):
            try:
                layer.trainable = training[i]
                x = layer(x)
            except:
                x = layer(x)
               
        #print(self.encoder_num.output)
        x = self.all_layers[10](x)


        for i, layer in enumerate(self.all_layers[11:]):
            i += 11
            try:
                layer.trainable = training[i]
                x = layer(x)
                #print(training[i],i)
            except:
                x = layer(x)
                #print('no')

        return x
    
    def temp_call(self, x, training=False):
         
        for layer in self.all_layers[:10]:
            x = layer(x)

        for layer in self.all_layers[10:]:
            try:
                x = layer(x, training=training)
                
            except:
                x = layer(x)
       
        return x

    def call(self, x, training=False):

        x = tf.cond(tf.constant(self.classifier,dtype=tf.bool), lambda: self.temp_call_classifier(x, training=training), lambda: self.temp_call(x, training=training))

        return x
    
   
    
    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
            
    #@tf.function
    def train_step(self, data):
        
        x, targets = data
        apply_awd_eagerly(fmodel, 0.5) # wird applied? #slanted triangular lr auch in der library lieber beides selber oder von library nutzen
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)

            #fmodel.summary()


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

    #@tf.function
    def test_step(self, data):

        x, targets = data
        
        predictions = self(x, training=False)
        
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        self.metrics[0].update_state(loss)
        
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}


    
    
if __name__ == "__main__":

    classifier = False
    
    fmodel = Fake_detection(classifier=classifier)

    if classifier:
        train_text, train_label, test_text, test_label,_,_ = data_preparation.get_dataset()

        train_label = train_label.astype('int32')
        test_label = test_label.astype('int32')

        

    else:
        train_text, train_label, test_text, test_label = data_preparation.get_amazon_dataset()


    




    train_text = fmodel.encode(train_text)
    test_text = fmodel.encode(test_text)
    
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_text, test_label))
    train_dataset = data_preparation.data_pipeline(train_dataset)
    test_dataset = data_preparation.data_pipeline(test_dataset)

    
    
    
    
    # Define where to save the log
    hyperparameter_string = "First_run"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"logs/{hyperparameter_string}/{current_time}/train"
    val_log_path = f"logs/{hyperparameter_string}/{current_time}/val"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)
    
    

    for epoch in range(fmodel.num_epoch):

        if fmodel.classifier:
            fmodel.gradual_unfreezing()
        
        
        print(f"Epoch {epoch}:")
        
        # Training:
        
        for data in tqdm(train_dataset,position=0, leave=True):
            metrics = fmodel.train_step(data)
            #print(fmodel.optimizer.get_config())
            #print(fmodel.optimizers_and_layers)
            
        # print the metrics
        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        
        # logging the validation metrics to the log file which is used by tensorboard
        with train_summary_writer.as_default():
            for metric in fmodel.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
        # reset all metrics (requires a reset_metrics method in the model)
        fmodel.reset_metrics()
        
        
        # Validation:
        
        for data in test_dataset:
            metrics = fmodel.test_step(data)

            
        
        print([f"val_{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        
        # logging the validation metrics to the log file which is used by tensorboard
        with val_summary_writer.as_default():
            for metric in fmodel.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
        # reset all metrics
        fmodel.reset_metrics()
        
        print("\n")

    print(fmodel.summary())
    print(len(fmodel.layers))
