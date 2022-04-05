import tensorflow as tf
import math


class STLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Subclass of tf.keras.optimizers.schedules.LearningRateSchedule to implement slanted triangular learning rates."""

    def __init__(self, num_epochs : int, updates_perepoch : int, cut_frac : float = 0.1, ratio : int = 32, lr_max : float =0.01):
        """Init 

        Args:
            num_epochs (int):           Number of epochs.
            updates_perepoch (int):     Updates per epoch.
            cut_frac (float, optional): Percentage when to decrease. Defaults to 0.1.
            ratio (int, optional):      Specifies the difference of size from the lowest lr to the maxium lr. Defaults to 32.
            lr_max (float, optional):   Maxiumum Learning Rate. Defaults to 0.01.
        """
        

      

        self.cut_frac = cut_frac
        self.ratio = ratio
        self.lr_max = lr_max
        
        # the number of epochs times the number of updates per epoch
        self.trainings_iterations = tf.cast(num_epochs * updates_perepoch, tf.float32)

        self.cut = math.floor(self.trainings_iterations * self.cut_frac)

        self.lr = 0

    def __call__(self, step : int) -> float:
        """Call function.

        Args:
            step (int): The current step of the learning process. 

        Returns:
            float: The calculated learning rate for the current step. 
        """

        help1 = step < self.cut
        help1 = tf.cast(help1, tf.float32)
        help2 = 1. - help1

        p = help1 * (step/self.cut) + help2 * (1-((step-self.cut)/(self.cut*((1/self.cut_frac)-1))))

        self.lr = self.lr_max * ((1+p*(self.ratio-1))/self.ratio)

        return self.lr

def get_optimizers(layers : list, num_epochs : int, num_updates_per_epoch : int, factor : float=2.6) -> list:
    """Returns a list of adam optimizers and their corresponding layers. 

    Args:
        layers (list):                  List of tf.keras.layers.
        num_epochs (int):               Number of epochs.
        num_updates_per_epoch (int):    Number of updates per epoch.
        factor (float, optional):       Learning rates decrease per layer. Defaults to 2.6.

    Returns:
        list: Adam optimizers and their corresponding layers.
    """

    optimizers = []
    num_layers = len(layers)

    for num in range(num_layers):
        try:
            lr_max = 0.01/(num * factor)
        except:
            lr_max = 0.01
        if type(num_epochs) == list:
            optimizers.append(tf.keras.optimizers.Adam(learning_rate=STLR(num_epochs[num], num_updates_per_epoch, lr_max=lr_max)))
        else:
            optimizers.append(tf.keras.optimizers.Adam(learning_rate=STLR(num_epochs, num_updates_per_epoch, lr_max=lr_max)))

    optimizers = optimizers[::-1]
    
    optimizers_and_layers = []

    for  optimizer, layer in zip(optimizers, layers):

        optimizers_and_layers.append((optimizer, layer))

    return optimizers_and_layers

