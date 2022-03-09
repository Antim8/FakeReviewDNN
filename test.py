import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text as tf_text

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.WARN)

language = "en"
hub_module = "https://tfhub.dev/google/wiki40b-lm-{}/1".format(language)
max_gen_len = 20

print("Using the {} model to generate sequences of max length {}.".format(hub_module, max_gen_len))

g = tf.Graph()
n_layer = 12
model_dim = 768

with g.as_default():
  text = tf.placeholder(dtype=tf.string, shape=(1,))

  # Load the pretrained model from TF-Hub
  module = hub.Module(hub_module)

  # Get the word embeddings, activations at each layer, negative log likelihood
  # of the text, and calculate the perplexity.
  embeddings = module(dict(text=text), signature="word_embeddings", as_dict=True)["word_embeddings"]
  activations = module(dict(text=text), signature="activations", as_dict=True)["activations"]
  neg_log_likelihood = module(dict(text=text), signature="neg_log_likelihood", as_dict=True)["neg_log_likelihood"]
  ppl = tf.exp(tf.reduce_mean(neg_log_likelihood, axis=1))

def feedforward_step(module, inputs, mems):
  """Generate one step."""
  # Set up the input dict for one step of generation
  inputs = tf.dtypes.cast(inputs, tf.int64)
  generation_input_dict = dict(input_tokens=inputs)
  mems_dict = {"mem_{}".format(i): mems[i] for i in range(n_layer)}
  generation_input_dict.update(mems_dict)

  # Generate the tokens from the language model
  generation_outputs = module(generation_input_dict, signature="prediction", as_dict=True)

  # Get the probablities and the inputs for the next steps
  probs = generation_outputs["probs"]
  new_mems = [generation_outputs["new_mem_{}".format(i)] for i in range(n_layer)]

  return probs, new_mems

with g.as_default():
  # Tokenization with the sentencepiece model.
  token_ids = module(dict(text=text), signature="tokenization", as_dict=True)["token_ids"]
  inputs_np = token_ids
  # Generate text by statically unrolling the computational graph
  mems_np = [np.zeros([1, 0, model_dim], dtype=np.float32) for _ in range(n_layer)]

  # Generate up to `max_gen_len` tokens
  sampled_ids = []
  for step in range(max_gen_len):
    probs, mems_np = feedforward_step(module, inputs_np, mems_np)
    sampled_id = tf.random.categorical(tf.math.log(probs[0]), num_samples=1, dtype=tf.int32)
    sampled_id = tf.squeeze(sampled_id)
    sampled_ids.append(sampled_id)
    inputs_np = tf.reshape(sampled_id, [1, 1])

  # Transform the ids into text
  sampled_ids = tf.expand_dims(sampled_ids, axis=0)
  generated_text = module(dict(token_ids=sampled_ids), signature="detokenization", as_dict=True)["text"]

  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
  
seed = "\n_START_ARTICLE_\n1882 Prince Edward Island general election\n_START_PARAGRAPH_\nThe 1882 Prince Edward Island election was held on May 8, 1882 to elect members of the House of Assembly of the province of Prince Edward Island, Canada."


with tf.Session(graph=g).as_default() as session:
  session.run(init_op)
  
with session.as_default():
    results = session.run([embeddings, neg_log_likelihood, ppl, activations, token_ids, generated_text], feed_dict={text: [seed]})
    embeddings_result, neg_log_likelihood_result, ppl_result, activations_result, token_ids_result, generated_text_result = results
    generated_text_output = generated_text_result[0].decode('utf-8')

print(generated_text_output)