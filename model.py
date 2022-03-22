import tensorflow as tf

class Classification(tf.keras.Model):

    def __init__(self, ulmfit):
        super(Classification, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.metrics_list = [
                    tf.keras.metrics.Mean(name="loss"),
                    tf.keras.metrics.CategoricalAccuracy(name="acc"),
                    tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                    ]

        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.layers = [
            ulmfit(trainable = False),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)]

        

    def call(self, x, training=False):

        
        x = self.layers[0](x)
        for layer in self.layers[1:]:
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




