# Ich denke, dass beste ist mit dem ULMfit model von tensorflow_hub zu arbeiten 
# documentation: https://bitbucket.org/edroneteam/tf2_ulmfit/src/master/
# vorher halt noch sich die dateien ziehen und die requirements installieren -> habs mal noch nich commited wegen upload

from tf2_ulmfit.ulmfit_tf2 import tf2_ulmfit_encoder
import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer
from tensorflow.python.platform import gfile



def get_pretrained_model(seq_length, model_path='tf2_ulmfit/enwiki100-toks-sp35k-cased.model'):
    
    # load the model
    spm_args = {'spm_model_file': model_path,
                'add_bos': True,
                'add_eos': True,
                'fixed_seq_len': seq_length}
    lm_num, encoder_num, mask_num, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args,
                                                                        fixed_seq_len=seq_length,
                                                                        )

    # load the weights
    encoder_num.load_weights('tf2_ulmfit/keras_weights/enwiki100_20epochs_toks_35k_cased').expect_partial() 

    return lm_num, encoder_num, mask_num, spm_encoder_model

def prepare_pretrained_model(pretrained_model, new_spm, seq_length):

    layers = get_list_of_layers(pretrained_model)

    layers = layers[2:]

    spm_args = {'spm_model_file': new_spm,
                'add_bos': True,
                'add_eos': True,
                'fixed_seq_len': seq_length}
    lm_num, encoder_num, mask_num, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args,
                                                                        fixed_seq_len=seq_length,
                                                                      )

    model = gfile.GFile(new_spm, 'rb').read()

    tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 

    vocab_size = tokenizer.vocab_size()
    
    

    new_layers = get_list_of_layers(encoder_num)

    keep = []

    keep.append(new_layers[0])
    keep.append(new_layers[1])
     
    for layer in layers:
        keep.append(layer) 

    keep.append(tf.keras.layers.GlobalAveragePooling1D())
    keep.append(tf.keras.layers.Dense(vocab_size, activation='softmax'))
 

    return keep


def get_list_of_layers(model):

    l = []
    for layer in model.layers:
        
        l.append(layer)
       
    return l


'''from tensorflow_text import SentencepieceTokenizer


from tensorflow.python.platform import gfile

model = gfile.GFile('shortenSPM.model', 'rb').read()

tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 

vocab_size = tokenizer.vocab_size()

lm_num, encoder_num, mask_num, spm_encoder_model = get_pretrained_model(70)

layers = prepare_pretrained_model(encoder_num, 'shortenSPM.model', 70, vocab_size)

model = tf.keras.Sequential()

for layer in layers:
    model.add(layer)

model.summary()'''


