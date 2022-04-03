import tensorflow as tf
import math


class STLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, num_epochs : int, updates_perepoch : int, cut_frac : float = 0.1, ratio : int = 32, lr_max : float =0.01):

        self.cut_frac = cut_frac
        self.ratio = ratio
        self.lr_max = lr_max
        
        # the number of epochs times the number of updates per epoch
        self.trainings_iterations = tf.cast(num_epochs * updates_perepoch, tf.float32)

        self.cut = math.floor(self.trainings_iterations * self.cut_frac)

        self.lr = 0

    def __call__(self, step : int) -> float:

        help1 = step < self.cut
        help1 = tf.cast(help1, tf.float32)
        help2 = 1. - help1

        p = help1 * (step/self.cut) + help2 * (1-((step-self.cut)/(self.cut*((1/self.cut_frac)-1))))

        self.lr = self.lr_max * ((1+p*(self.ratio-1))/self.ratio)

        return self.lr