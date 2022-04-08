import tensorflow as tf

def get_fine_tuned_layers():
    """Return the layers of the saved model which is finetuned on amazon reviews.

    Returns:
        list: Layers of the fine tuned model.
    """
    model = tf.keras.models.load_model('saved_model/fine_tuned_model')

    temp_layers = []

    for layer in model.layers:
        temp_layers.append(layer)

    temp_layers = temp_layers[3:-2]

    layers = []

    for layer in temp_layers:
        layers.append(layer)

    model = tf.keras.Sequential()
   
    for layer in layers:
        model.add(layer)

    layers = []

    for layer in model.layers:
        layers.append(layer)


    return layers

get_fine_tuned_layers()
