import tensorflow as tf

class Classification(tf.keras.Model):

    def __init__(self, ulmfit):
        super(Classification, self).__init__()
        self.ulmfit = ulmfit
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, inputs):

        x = self.ulmfit(inputs)
        x = self.dense1(x)
        x = self.flatten(x)
        x = self.dense2(x)

        return x 


