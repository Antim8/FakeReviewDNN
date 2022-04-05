import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow_text import SentencepieceTokenizer

from tf2_ulmfit.ulmfit_tf2 import tf2_ulmfit_encoder


def get_pretrained_model(seq_length : int, model_path : str ='tf2_ulmfit/enwiki100-toks-sp35k-cased.model') -> tuple:
    """Returns models with trained weights (Wikipedia 35k) and the SentencePiece encoder model.

    Args:
        seq_length (int): Sequence length.
        model_path (str, optional): Path of a SentencePiece model. Defaults to 'tf2_ulmfit/enwiki100-toks-sp35k-cased.model'.

    Returns:
        tuple: 
            lm_num (tf.keras.engine.functional.Functional):            Language Model with head (softmax).
            encoder_num (tf.keras.engine.functional.Functional):       Language Model without head.
            spm_encoder_model (tf.keras.engine.functional.Functional): Model to encode text.
            
    """
    
    # load the model
    spm_args = {'spm_model_file': model_path,
                'add_bos': True,
                'add_eos': True,
                'fixed_seq_len': seq_length}
    lm_num, encoder_num, _, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args,
                                                                        fixed_seq_len=seq_length
                                                                        )

    # load the weights
    encoder_num.load_weights('tf2_ulmfit/keras_weights/enwiki100_20epochs_toks_35k_cased').expect_partial() 

    return lm_num, encoder_num, spm_encoder_model

def prepare_pretrained_model(pretrained_model : tf.keras.engine.functional.Functional, new_spm : str, seq_length : int) -> tuple:
    """Returns layers of the model for fine tuning the general language model and the SentencePiece encoder model.

    Args:
        pretrained_model (tf.keras.engine.functional.Functional): General domain language model.
        new_spm (str): Path to the adjusted SentencePiece model.
        seq_length (int): Sequence Length.

    Returns:
        tuple: 
            keep (list):                                                Layers of the model for fine tuning the general language model.
            spm_encoder_model (tf.keras.engine.functional.Functional):  Model to encode text.
    """

    layers = get_list_of_layers(pretrained_model)

    layers = layers[2:]

    model = gfile.GFile(new_spm, 'rb').read()

    tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 

    vocab_size = tokenizer.vocab_size()

    spm_args = {'spm_model_file': new_spm,
                'add_bos': True,
                'add_eos': True,
                'fixed_seq_len': seq_length}
    _, encoder_num, _, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args,
                                                                        fixed_seq_len=seq_length,
                                                                        vocab_size=vocab_size
                                                                      )
    new_layers = get_list_of_layers(encoder_num)

    keep = []

    keep.append(new_layers[0])
    keep.append(new_layers[1])
     
    for layer in layers:
        keep.append(layer) 

    keep.append(tf.keras.layers.GlobalAveragePooling1D())
    keep.append(tf.keras.layers.Dense(vocab_size, activation='softmax'))
 
    return keep, spm_encoder_model

def get_list_of_layers(model : tf.keras.Model) -> list:
    """Returns the layers of a given model.

    Args:
        model (tf.keras.Model): TensorFlow Model.
    Returns:
        list: Layers of the model.
    """

    l = []
    for layer in model.layers:
        
        l.append(layer)
       
    return l

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