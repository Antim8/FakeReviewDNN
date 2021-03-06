import random

import numpy as np
import pandas as pd
import sentencepiece
import sentencepiece as spm
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.platform import gfile
from tensorflow_text import SentencepieceTokenizer

from tf2_ulmfit.ulmfit_tf2 import tf2_ulmfit_encoder
from sentencepiece import sentencepiece_model_pb2 as model





def get_pretrained_model(seq_length : int, model_path : str ='tf2_ulmfit/enwiki100-toks-sp35k-cased.model') -> tuple:
    """Returns models with trained weights (Wikipedia 35k) and the SentencePiece encoder model.

    Args:
        seq_length (int): Sequence length.
        model_path (str, optional): Path of a SentencePiece model. Defaults to 'tf2_ulmfit/enwiki100-toks-sp35k-cased.model'.

    Returns:
        tuple: 
            lm_num (tf.keras.Model):            Language Model with head (softmax).
            encoder_num (tf.keras.Model):       Language Model without head.
            spm_encoder_model (tf.keras.Model): Model to encode text.
            
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

def get_spm_encoder_model(seq_length : int, model_path : str) -> tf.keras.Model:
    """Get the sentencepiece encoder model to encode text on the givin sentenpiece model.

    Args:
        seq_length (int): _description_
        model_path (str, optional): _description_.

    Returns:
        tf.keras.Model: _description_
    """

    spm_args = {'spm_model_file': model_path,
                'add_bos': True,
                'add_eos': True,
                'fixed_seq_len': seq_length}
    _, _, _, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args,
                                                                        fixed_seq_len=seq_length
                                                                        )

    return spm_encoder_model



def prepare_pretrained_model(pretrained_model : tf.keras.Model, new_spm : str, seq_length : int) -> tuple:
    """Returns layers of the model for fine tuning the general language model and the SentencePiece encoder model.

    Args:
        pretrained_model (tf.keras.Model): General domain language model.
        new_spm (str): Path to the adjusted SentencePiece model.
        seq_length (int): Sequence Length.

    Returns:
        tuple: 
            keep (list):                                                Layers of the model for fine tuning the general language model.
            spm_encoder_model (tf.keras.Model):                    Model to encode text.
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

def get_dataset() -> tuple: 
    """Returns the train and test data of the Fake Review dataset for the classification to detect fake reviews.

    Returns:
        tuple: 
            train_text (list):  Review text.
            train_label (list): 1 -> Computer-generated review, 0 -> Original review.
            test_text (list):   Review text.
            test_label (list):  1 -> Computer-generated review, 0 -> Original review.
    """

    # CG = Computer-generated fake reviews; OR = Original reviews
    label = pd.read_csv("fake_reviews_dataset.csv", usecols=[2]).values

    # CG -> 1; OR -> 0
    for [i], j in zip(label, range(len(label))):
        if i == "CG":
            label[j] = 1
        else:
            label[j] = 0
    

    # The review 
    text = pd.read_csv("fake_reviews_dataset.csv", usecols=[3]).values
    text = np.squeeze(text, axis=1)

    df = pd.DataFrame(columns=['text','label'])
    df['text'] = text

    df['label'] = label

    # Shuffle the data, since it is sorted by categories
    df = df.sample(frac=1)

    training_set_size = int(df.shape[0]*0.5)
    train_df = df[:training_set_size]
    test_df = df[training_set_size:]

    train_text = train_df.text.to_numpy()
    train_label = train_df.label.to_numpy()
    test_text = test_df.text.to_numpy()
    test_label = test_df.label.to_numpy()

    train_label = np.expand_dims(train_label, axis=1)
    test_label = np.expand_dims(test_label, axis=1)

    return train_text, train_label, test_text, test_label


def get_amazon_dataset() -> tuple:
    """Returns the train and test data of the amazon dataset for the fine-tuning of the general domain model.

    Returns:
        tuple: 
            train_text (list):  Review text.
            train_label (list): One-hot encoding of the review texts last token. 
            test_text (list):   Review text.
            test_label (list):  One-hot encoding of the review texts last token.
        
    """

    df = pd.read_parquet("rev_clean_data.parquet")
    df.columns = ['text', 'label']

    df = df.dropna(how='any', axis=0)

    labels = []

    for l in df.label:
        labels.append(l.tolist())
    df.label = labels

    training_size = int(df.shape[0]*0.7)

    train_df = df[:training_size]
    test_df = df[training_size:]

    train_text = train_df.text.to_numpy()
    train_label = train_df.label.tolist()


    test_text = test_df.text.to_numpy()
    test_label = test_df.label.tolist()

    return train_text, train_label, test_text, test_label



def data_pipeline(ds: tf.data.Dataset, shuffle:int=1000, batch:int=64) -> tf.data.Dataset:
    """Preprocess data (shuffle, batch, prefetch).

    Args:
        ds (tf.data.Dataset):       TensorFlow dataset.
        shuffle (int, optional):    Shuffle size. Defaults to 1000.
        batch (int, optional):      Batch size. Defaults to 64.
        prefetch (int, optional):   Prefetch size. Defaults to 20.

    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    
    ds = ds.shuffle(shuffle)
    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def get_amazon_data() -> list:
    """Load the amazon data into a list.

    Returns:
        list: Reviews in the amazon dataset ("rev_data.txt")
    """

    path = "rev_data.txt"

    file = open(path, 'r')
    reviews = file.readlines()

    for i, review in enumerate(reviews):
        reviews[i] = review.rstrip('\n')
        

    return reviews

def get_tokens_to_keep(text : list, tokenizer : SentencepieceTokenizer) -> list:
    """Evaluate which tokens appear in a different text corpus given an existing corpus.

    Args:
        text (list):                        List of strings 
        tokenizer (SentencepieceTokenizer): Sentencepiece Tokenizer trained on an existing corpus.

    Returns:
        list: Index of tokens which appear in the new corpus as well as in the given corpus.
    """

    tokens = []

    for line in text:
        for word in line.split():

            tokens.append(tokenizer.string_to_id(word).numpy())

    return list(set(tokens))

def shorten_SPM(model_path='tf2_ulmfit/enwiki100-toks-sp35k-cased.model'):
    """Deletes tokens of the Sentencepiece model, which do not appear in the amazon data.

    Args:
        model_path (str, optional): Path to the .model file. Defaults to 'tf2_ulmfit/enwiki100-toks-sp35k-cased.model'.
    """
    model = gfile.GFile(model_path, 'rb').read()
    tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 
    SPM = model.ModelProto()
    SPM.ParseFromString(open(model_path, 'rb').read()) 


    tokens_to_keep = get_tokens_to_keep(get_amazon_data(), tokenizer)

    index = len(SPM.pieces) - 1

    keep = []

    while len(SPM.pieces):

        piece = SPM.pieces.pop()
        if index < 1_000 or index in tokens_to_keep:
            keep.append(piece)

        index -= 1

    keep = list(reversed(keep))

    for piece in keep:
        SPM.pieces.append(piece)
    
    # create new file with shortend tokens
    with open("shortenSPM.model", 'wb') as f:
        f.write(SPM.SerializeToString())

def merge_SPM(first_model_path : str = 'shortenSPM.model', second_model_path: str = 'amazon.model'):
    """Compines the tokens of two sentencepiece models and creates a new model file.

    Args:
        first_model_path (str, optional):   Path to the first .model file. Defaults to 'shortenSPM.model'.
        second_model_path (str, optional):  Path to the second .model file. Defaults to 'amazon.model'.
    """

    first_SPM = model.ModelProto()
    first_SPM.ParseFromString(open(first_model_path, 'rb').read()) 
    second_SPM = model.ModelProto()
    second_SPM.ParseFromString(open(second_model_path, 'rb').read()) 

    new_pieces = []
    temp_pieces = []

    for piece in second_SPM.pieces:

        temp_pieces.append(piece.piece)

    for piece in first_SPM.pieces:

        if piece.piece not in temp_pieces:
            new_pieces.append(piece)

    for piece in new_pieces:
        second_SPM.pieces.append(piece)

    with open("new_amazon.model", 'wb') as f:
        f.write(second_SPM.SerializeToString())
    
def prepare_for_generation(text_data:str, model_path:str):
    """Create and save a dataframe of a given text corpus and sentencepiece model to a parquet file. Simple preprocessing steps applied.

    Args:
        text_data (str):    Path of text data.
        model_path (str):   Path to a sentencepiece model.
    """
    
    model = gfile.GFile(model_path, 'rb').read()
    spm = SentencepieceTokenizer(model, add_bos=True, add_eos=True)
    revst = spm.tokenize(text_data)
    
    revst = [list(x.numpy()) for x in revst]
    inp = []
    label = []
    for r in revst:
        
        if len(r) < 5:
            if len(r) <= 3 or 0 in r:
                del(r)
                continue
        if len(r) >= 7 and r[-7:].count(0) >= 2:
            del(r)
            continue
        if r[-2] == 0:
            del(r)
            continue
        tmp_inp = tf.convert_to_tensor(r[:-3])
        tmp_label = tf.convert_to_tensor(r[-3])
        tmp_inp = spm.detokenize(tmp_inp)
        tmp_label = spm.id_to_string(tmp_label)
        inp.append(tmp_inp)
        label.append(tmp_label)

    labels = []
        
    new_inp, new_label = [] , []
    for i, l in zip(inp, label):
        i = i.numpy().decode('utf-8')
        l = l.numpy().decode('utf-8')
        new_inp.append(i)
        new_label.append(l)

    

    sp = sentencepiece.SentencePieceProcessor()
    sp.load('new_amazon.model')

    for label in new_label:
        temp = []
        for id in range(sp.vocab_size()):
            if sp.id_to_piece(id) == label:
                temp.append(1)
            else: 
                temp.append(0)
        labels.append(temp)

    

    new_label = labels

    data = pd.DataFrame(columns=['input','label'], data=zip(new_inp, new_label))
    data.to_parquet('rev_clean_data.parquet')



def create_amazon_dataset():
    """Create a text file with amazon reviews from specific categories."""

    _datasets = [
        "Apparel_v1_00",
        "Beauty_v1_00",
        "Books_v1_00",
        "Home_v1_00",
        "Video_v1_00",
        "Wireless_v1_00"
    ]

    final_dataset = []

    for rev_dataset in _datasets:
        ds = None
        ds = tfds.load('amazon_us_reviews/{}'.format(rev_dataset))
        ds = ds["train"]
        ds = ds.shuffle(buffer_size=10000)
        
        for d in ds.take(10000):
            
            # We convert the data from being a tf tensor to a python string and strip newlines
            final_dataset.append(d["data"]["review_body"].numpy().decode("utf-8").replace("\n", " "))
            
        print("{} done".format(rev_dataset))

    # Shuffle the dataset to not have it ordered by categories
    random.shuffle(final_dataset)   
    
    with open("./rev_data.txt", "w") as f:
        for review in final_dataset:
            
            # We use try here to avoid an error concerning unknown tokens or emojis
            try:
                f.write(review)
                f.write("\n")
            except:
                pass

def train_sentencepiece_model(input : str = "rev_data.txt", name : str = "amazon", vocab_size : str = "2000"):
    """Trains a sentencepiece model on the amazon dataset.

    Args:
        input (str, optional):          Path to the text file. Defaults to "rev_data.txt".
        name (str, optional):           Name of the model file. Defaults to "amazon".
        vocab_size (str, optional):     Vocabulary size. Defaults to "2000".
    """
    
    spm.SentencePieceTrainer.train("--input=" + input + "--model_prefix=" + name + " --vocab_size=" + vocab_size)


'''from tensorflow_text import SentencepieceTokenizer
from tensorflow.python.platform import gfile
model = gfile.GFile('new_amazon.model', 'rb').read()
tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 
vocab_size = tokenizer.vocab_size()
print(vocab_size)'''

