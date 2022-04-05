from turtle import st
import numpy as np
import tensorflow as tf
import pandas as pd
import sentencepiece
from collections import Counter
from tensorflow_text import SentencepieceTokenizer
from tensorflow.python.platform import gfile
import sentencepiece as spm


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


def data_pipeline(ds: tf.data.Dataset, shuffle:int=1000, batch:int=64, prefetch:int=20) -> tf.data.Dataset:
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
    #TODO advanced shit maybe
    ds = ds.prefetch(prefetch)

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
#TODO typing

#TODO what is SPM 
def shorten_SPM(SPM : spm.sentencepiece_model_pb2.ModelProto, tokenizer:SentencepieceTokenizer):
    """Deletes tokens of the Sentencepiece model, which do not appear in the amazon data.

    Args:
        SPM (spm.sentencepiece_model_pb2.ModelProto):   Sentencepiece model to be shorten.
        tokenizer (SentencepieceTokenizer):         Sentencepiece Tokenizer trained on an existing corpus.
    """

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
#TODO typing
def merge_SPM(first_SPM : spm.sentencepiece_model_pb2.ModelProto, second_SPM : spm.sentencepiece_model_pb2.ModelProto):
    """Compines the tokens of two sentencepiece models and creates a new model file.

    Args:
        first_SPM (spm.sentencepiece_model_pb2.ModelProto):     Sentencepiece model to be combined.
        second_SPM (spm.sentencepiece_model_pb2.ModelProto):    Sentencepiece model to be combined.
    """

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
        
    new_inp, new_label = [] , []
    for i, l in zip(inp, label):
        i = i.numpy().decode('utf-8')
        l = l.numpy().decode('utf-8')
        new_inp.append(i)
        new_label.append(l)

    labels = []
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

