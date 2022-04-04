from turtle import st
import numpy as np
import tensorflow as tf
import pandas as pd
import sentencepiece
from collections import Counter
from tensorflow_text import SentencepieceTokenizer
from tensorflow.python.platform import gfile


def get_dataset() -> tuple: 

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


def data_pipeline(ds, shuffle:int=1000, batch:int=64, prefetch:int=20) -> tf.data.Dataset:
    
    ds = ds.shuffle(shuffle)
    ds = ds.batch(batch)
    #TODO advanced shit maybe
    ds = ds.prefetch(prefetch)

    return ds


def get_amazon_data() -> list:

    path = "rev_data.txt"

    file = open(path, 'r')
    reviews = file.readlines()

    for i, review in enumerate(reviews):
        reviews[i] = review.rstrip('\n')
        

    return reviews

def get_tokens_to_keep(text:str, tokenizer:SentencepieceTokenizer) -> list:

    tokens = []

    for line in text:
        for word in line.split():

            tokens.append(tokenizer.string_to_id(word).numpy())

    return list(set(tokens))
#TODO typing
def shorten_SPM(SPM, tokenizer:SentencepieceTokenizer):

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
    

    with open("shortenSPM.model", 'wb') as f:
        f.write(SPM.SerializeToString())
#TODO typing
def merge_SPM(new_SPM, old_SPM):

    new_pieces = []
    temp_pieces = []

    for piece in old_SPM.pieces:

        temp_pieces.append(piece.piece)

    for piece in new_SPM.pieces:

        if piece.piece not in temp_pieces:
            new_pieces.append(piece)

    for piece in new_pieces:
        old_SPM.pieces.append(piece)

    with open("new_amazon.model", 'wb') as f:
        f.write(old_SPM.SerializeToString())
    
def prepare_for_generation(text_data:str, model_path:str):
    
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

        
if __name__ == "__main__":
    
    with open("./rev_data.txt", "r") as f:
        text_data = f.readlines()
    prepare_for_generation(text_data, "./new_amazon.model")

    




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


    

'''from tensorflow_text import SentencepieceTokenizer


from tensorflow.python.platform import gfile

model = gfile.GFile('new_amazon.model', 'rb').read()

tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string)  
print(tokenizer.vocab_size())'''
        

        
if __name__ == "__main__":
    
    with open("./rev_data.txt", "r") as f:
        text_data = f.readlines()
    prepare_for_generation(text_data, "./new_amazon.model")

   

'''from tensorflow_text import SentencepieceTokenizer


from tensorflow.python.platform import gfile

model = gfile.GFile('tf2_ulmfit/enwiki100-toks-sp35k-cased.model', 'rb').read()

tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 

print(tokenizer.string_to_id("this"))'''

'''from sentencepiece import sentencepiece_model_pb2 as model

m = model.ModelProto()
m.ParseFromString(open("tf2_ulmfit/enwiki100-toks-sp35k-cased.model", 'rb').read())


a = shorten_SPM(m ,tokenizer)
print(a)


model = gfile.GFile('shortenSPM.model', 'rb').read()
tokenizer = SentencepieceTokenizer(model=model, out_type=tf.string) 
print(tokenizer.vocab_size())'''

'''import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='shortenSPM.model')
vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
print(len(vocabs))'''
