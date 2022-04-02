import tensorflow as tf
import slanted_triangular_lr

def get_optimizers(layers : tf.keras.layers, num_epochs : int, num_updates_per_epoch : int, factor : float=2.6) -> list:

    optimizers = []
    num_layers = len(layers)

    for num in range(num_layers):
        try:
            lr_max = 0.01/(num * factor)
        except:
            lr_max = 0.01
        if type(num_epochs) == list:
            optimizers.append(tf.keras.optimizers.Adam(learning_rate=slanted_triangular_lr.STLR(num_epochs[num], num_updates_per_epoch, lr_max=lr_max)))
        else:
            optimizers.append(tf.keras.optimizers.Adam(learning_rate=slanted_triangular_lr.STLR(num_epochs, num_updates_per_epoch, lr_max=lr_max)))

    optimizers = optimizers[::-1]
    
    optimizers_and_layers = []

    for  optimizer, layer in zip(optimizers, layers):

        optimizers_and_layers.append((optimizer, layer))

    return optimizers_and_layers

