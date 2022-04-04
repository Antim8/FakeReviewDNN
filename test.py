from spacy import Vocab
import tensorflow as tf
import model_import as mi

def testoo():
    model = tf.keras.models.load_model('saved_model/fine_tuned_model')

    temp_layers = []

    for layer in model.layers:
        temp_layers.append(layer)

    temp_layers = temp_layers[3:-2]

    layers = []

    for layer in temp_layers:
        layers.append(layer)

    


    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.InputLayer(input_shape=(256), batch_size=64))

    #for layer in layers:
    #   model.add(layer)

    #model.summary()

    return layers

print(testoo())






'''model = tf.keras.models.load_model('saved_model/fine_tuned_model')

layers = []

for layer in model.layers:
    layers.append(layer)

layers = layers[3:-2]

model = tf.keras.Sequential()

for layer in layers:
    model.add(layer)

print(model.layers[0].get_weights()[0])
print(model.layers[0])'''


'''model = tf.keras.models.load_model('saved_model/fine_tuned_model')

layers = [tf.keras.layers.InputLayer(256)]

for layer in model.layers:
    layers.append(layer)

layers = layers[4:-2]

model = tf.keras.Sequential()

for layer in layers:
    model.add(layer)

    

 

print(testoo())'''

'''seq_length = 256

lm_num, encoder_num, _, spm_encoder_model = mi.get_pretrained_model(seq_length)
pretrained_layers, spm_encoder_model = mi.prepare_pretrained_model(encoder_num, 'new_amazon.model', seq_length)

model = tf.keras.Sequential()

for layer in pretrained_layers:
    model.add(layer)

model.summary()

model.load_weights('weights/lm_finetuning_model').expect_partial()'''

