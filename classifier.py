from model_import import get_pretrained_model
from tf2_ulmfit.ulmfit_tf2 import ConcatPooler
import tensorflow as tf

#TODO selber implementieren

def get_classifier(model):

    
    # Concat Pooling
    drop_pooler = ConcatPooler(name="ConcatPooler")(model.output)

    bnorm_pooler = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False)(drop_pooler)
    bnorm_drop = tf.keras.layers.Dropout(0.1)(bnorm_pooler)
    fc1 = tf.keras.layers.Dense(50, activation='linear', use_bias=False)(bnorm_drop)
    relu1 = tf.keras.layers.ReLU()(fc1)
    bnorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False)(relu1)
    drop2 = tf.keras.layers.Dropout(0.1)(bnorm1)
    fc_final = tf.keras.layers.Dense(1, use_bias=False, activation='sigmoid')(drop2)




    classifier = tf.keras.models.Model(inputs=model.inputs, outputs=fc_final)

    classifier.summary()

    return classifier





    