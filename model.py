import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
import datetime
import util
import tensorflow_addons as tfa
from model_util import get_optimizers
from tf2_ulmfit.ulmfit_tf2 import apply_awd_eagerly
from tf2_ulmfit.ulmfit_tf2 import ConcatPooler

import test



class Fake_detection(tf.keras.Model):
    """Subclass model with features inspired by ULMFiT."""
    
    def __init__(self, classifier=False):
        """Class init

        Args:
            classifier (bool, optional): Use classifier or fine-tune model. Defaults to False.
        """
        
        super(Fake_detection, self).__init__()

        self.num_epoch = 1
        self.num_updates_per_epoch = 650

        self.classifier = classifier

        seq_length = 256

        self.lm_num, self.encoder_num, self.spm_encoder_model = util.get_pretrained_model(seq_length)

        if classifier:

            self.spm_encoder_model = util.get_spm_encoder_model(seq_length=seq_length, model_path='new_amazon.model')

            self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.BinaryAccuracy(name="acc"),
                       ]
            self.loss_function = tf.keras.losses.BinaryCrossentropy()  
            temp_pretrained = util.get_list_of_layers(self.encoder_num)

            pretrained_layers = [temp_pretrained[0], temp_pretrained[1]]

            temp_pretrained2 = util.get_fine_tuned_layers()
            for layer in temp_pretrained2[2:]:
                pretrained_layers.append(layer)

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
                        ]
            self.loss_function = tf.keras.losses.CategoricalCrossentropy()  
            pretrained_layers, self.spm_encoder_model = util.prepare_pretrained_model(self.encoder_num, 'new_amazon.model', seq_length)
            num_epochs_list = self.num_epoch

        for layer in pretrained_layers:
            layer.trainable = True

        self.training_list = []
        for _ in range(len(pretrained_layers)):
            self.training_list.append(False)
        self.no_training = self.training_list
        self.training_list_index = len(pretrained_layers) - 1

        self.all_layers = pretrained_layers

        # Discriminiative fine tuning
        self.optimizers_and_layers = get_optimizers(layers=self.all_layers, num_epochs=num_epochs_list, num_updates_per_epoch=self.num_updates_per_epoch)

        self.optimizer = tfa.optimizers.MultiOptimizer(self.optimizers_and_layers)

        

    def encode(self, text : tf.string) -> tf.Tensor:
        """Encodes text to tf.Tensor using sentencepiece.

        Args:
            text (tf.string): Text

        Returns:
            tf.Tensor: Text represented as tf.Tensor
        """

        return self.spm_encoder_model(tf.constant(text, dtype=tf.string))
       


    def gradual_unfreezing(self):
        """Keeps track of layers to be updated for gradual unfreezing."""

        #TODO more layers than epochs error

        self.training_list[self.training_list_index] = True

        temp = self.training_list_index > 0

        self.training_list_index = (self.training_list_index - 1) * temp

    def temp_call_classifier(self, x : tf.Tensor, training : bool = False) -> tf.Tensor:
        """Call function if classifier is true.

        Args:
            x (tf.Tensor):              Input
            training (bool, optional):  Weather weights should be trained or not. Defaults to False.

        Returns:
            tf.Tensor: Output of the layers.
        """
    
        training = self.training_list * training + self.no_training * (int(training)+1)
        
        

        for i, layer in enumerate(self.all_layers[:10]):
            try:
                layer.trainable = training[i]
                x = layer(x)
            except:
                x = layer(x)
               
        x = self.all_layers[10](x)


        for i, layer in enumerate(self.all_layers[11:]):
            i += 11
            try:
                layer.trainable = training[i]
                x = layer(x)
            except:
                x = layer(x)

        return x
    
    def temp_call(self, x : tf.Tensor, training : bool = False) -> tf.Tensor:
        """Call function if one fine-tunes the model.

        Args:
            x (tf.Tensor):              Input
            training (bool, optional):  Weather weights should be trained or not. Defaults to False.

        Returns:
            tf.Tensor: Output of the layers.
        """
         
        for layer in self.all_layers[:10]:
            x = layer(x)

        for layer in self.all_layers[10:]:
            try:
                x = layer(x, training=training)
            except:
                x = layer(x)
       
        return x
    #TODO if else
    def call(self, x : tf.Tensor, training : bool = False) -> tf.Tensor:
        """Calls either temp_call or temp_call_classifier.

        Args:
            x (tf.Tensor):              Input
            training (bool, optional):  Weather weights should be trained or not. Defaults to False.

        Returns:
            tf.Tensor: Output of the layers.
        """

        x = tf.cond(tf.constant(self.classifier,dtype=tf.bool), lambda: self.temp_call_classifier(x, training=training), lambda: self.temp_call(x, training=training))

        return x
    
   
    
    def reset_metrics(self):
        """Reset metric states."""
        
        for metric in self.metrics:
            metric.reset_states()
            
    #TODO tf function possible?
    def train_step(self, data : tf.data.Dataset) -> dict:
        """The train step of the model.

        Args:
            data (tf.data.Dataset): The given dataset composed of train_text and train_label.

        Returns:
            dict: Results of the train step.
        """
        
        x, targets = data
        apply_awd_eagerly(fmodel, 0.5) 
        
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

    #TODO tf function possible?
    def test_step(self, data : tf.data.Dataset) -> dict:
        """The test step of the model.

        Args:
            data (tf.data.Dataset): The given dataset composed of test_text and test_label.

        Returns:
            dict: Results of the test step.
        """

        x, targets = data
        
        predictions = self(x, training=False)
        
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        self.metrics[0].update_state(loss)
        
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}


    
    
if __name__ == "__main__":

    classifier = True
    
    fmodel = Fake_detection(classifier=classifier)

    if classifier:
        
        train_text, train_label, test_text, test_label = util.get_dataset()
        train_label = train_label.astype('int32')
        test_label = test_label.astype('int32')

    else:
        train_text, train_label, test_text, test_label = util.get_amazon_dataset()

    train_text = fmodel.encode(train_text)
    test_text = fmodel.encode(test_text)
        
    train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_text, test_label))
    train_dataset = util.data_pipeline(train_dataset)
    test_dataset = util.data_pipeline(test_dataset)


    
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
        # print the metrics
        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        
        # logging the validation metrics to the log file which is used by tensorboard
        with train_summary_writer.as_default():
            for metric in fmodel.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
        # reset all metrics (requires a reset_metrics method in the model)
        fmodel.reset_metrics()
        
        
        # Testing:
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

    fmodel.summary()

    if classifier:
        

        fmodel.save_weights('./weights/classifier_model')
    else:
        fmodel.summary()
        fmodel.save('saved_model/fine_tuned_model')
        fmodel.save_weights('./weights/lm_finetuning_model')



