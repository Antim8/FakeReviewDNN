import model_import
import tensorflow as tf

prompt = ["I am a big"]

_, enc_num, _, spm_encoder_model = model_import.get_pretrained_model(seq_length=256)

prompt = spm_encoder_model(tf.constant(prompt, dtype=tf.string))
print(prompt)
dennis = tf.keras.layers.Dense(units=35_000, activation=tf.nn.softmax)

x = enc_num(prompt)
#x = dennis(x)
a, b = tf.math.top_k(x, 5)

#print(a)
print(b)
