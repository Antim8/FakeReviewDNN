from cgi import test
from tkinter.tix import DirTree
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
from itertools import chain
from model_import import get_pretrained_model
from tf2_ulmfit.ulmfit_tf2 import tf2_ulmfit_encoder
from tensorflow_text import SentencepieceTokenizer
from tensorflow.python.platform import gfile
import sentencepiece
from ast import literal_eval

def get_dataset():

    # label for fake or real review
    # CG -> Computer-generated fake reviews; OR = Original reviews
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

    #label = np.expand_dims(label, axis=1)

    #print(label)

    df['label'] = label


    # Shuffle the data, since it is sorted by categories
    df = df.sample(frac=1)


    training_set_size = int(df.shape[0]*0.5)

    train_df = df[:training_set_size]
    test_df = df[training_set_size:]

    validation_set_size = int(test_df.shape[0]*0.5)

    vali_df = test_df[:validation_set_size]
    test_df = test_df[validation_set_size:]
    

    train_text = train_df.text.to_numpy()
    train_label = train_df.label.to_numpy()

    test_text = test_df.text.to_numpy()
    test_label = test_df.label.to_numpy()

    vali_text = vali_df.text.to_numpy()
    vali_label = vali_df.label.to_numpy()

    train_label = np.expand_dims(train_label, axis=1)
    test_label = np.expand_dims(test_label, axis=1)
    vali_label = np.expand_dims(vali_label, axis=1)

    return train_text, train_label, test_text, test_label, vali_text, vali_label

def get_amazon_dataset():

    df = pd.read_csv("rev_clean_data.csv")
    df.columns = ['text', 'label']

    df = df.dropna(how='any', axis=0)

    print(len(df['label'][0]))
    labels = []

    for l in df['label'][:]:
        labels.append(literal_eval(l))

    df.label = labels

    
    print(df.head())
    

    #for element in df.label.to_numpy():
    #    print(len(element))

    training_size = int(df.shape[0]*0.7)

    train_df = df[:training_size]
    test_df = df[training_size:]

    train_text = train_df.text.to_numpy()
    train_label = train_df.label #.to_numpy()
    #print(train_label.shape)

    test_text = test_df.text.to_numpy()
    test_label = test_df.label.to_numpy()

    return train_text, train_label, test_text, test_label

#a,b,c,d = get_amazon_dataset()

#print(b, b.shape)


def get_dataframe():

    # label for fake or real review
    # CG -> Computer-generated fake reviews; OR = Original reviews
    label = pd.read_csv("fake_reviews_dataset.csv", usecols=[2]).values

    # CG -> 1; OR -> 0
    for [i], j in zip(label, range(len(label))):
        if i == "CG":
            label[j] = 1
        else:
            label[j] = 0
    label = np.squeeze(label, axis=1)

    # The review 
    text = pd.read_csv("fake_reviews_dataset.csv", usecols=[3]).values
    text = np.squeeze(text, axis=1)

    df = pd.DataFrame(columns=['text','label'])
    df['text'] = text
    df['label'] = label

    return df

def tokenizer(train_text_data, test_text_data, oov_token='<UNK>', maxlen=None, padding='post'):

    if maxlen == None:
        maxlen = max(len(x) for x in train_text_data)

    num_words = len(get_counts(train_text_data))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words, oov_token=oov_token)
    tokenizer.fit_on_sequences(train_text_data)

    train_text_seq = tokenizer.texts_to_sequences(train_text_data)
    train_text_seq = tf.keras.preprocessing.sequence.pad_sequences(train_text_seq, maxlen=maxlen, padding=padding)
    
    test_text_seq = tokenizer.texts_to_sequences(test_text_data)
    test_text_seq = tf.keras.preprocessing.sequence.pad_sequences(test_text_seq, maxlen=maxlen, padding=padding)
    
    return train_text_seq, test_text_seq, maxlen, num_words

def data_pipeline(ds, shuffle=1000, batch=64, prefetch=20):
    ds = ds.shuffle(shuffle)
    ds = ds.batch(batch)
    ds = ds.prefetch(prefetch)

    return ds

def get_counts(text_data):

    count = Counter()
    for text in text_data:
        for word in text.split():
            count[word] += 1
        return count


def get_amazon_data():

    path = "rev_data.txt"

    file = open(path, 'r')
    reviews = file.readlines()

    for i, review in enumerate(reviews):
        reviews[i] = review.rstrip('\n')
        

    return reviews

def get_tokens_to_keep(text, tokenizer):

    tokens = []

    for line in text:
        for word in line.split():

            tokens.append(tokenizer.string_to_id(word).numpy())

    return list(set(tokens))

def shorten_SPM(SPM, tokenizer):

    tokens_to_keep = get_tokens_to_keep(get_amazon_data(), tokenizer)

    index = len(SPM.pieces) - 1

    keep = []

    while len(SPM.pieces):

        piece = SPM.pieces.pop()
        if index < 1_000 or index in tokens_to_keep:
            keep.append(piece)
            #print(index, "<--->", piece)

        index -= 1

    keep = list(reversed(keep))

    for piece in keep:
        SPM.pieces.append(piece)
    

    with open("shortenSPM.model", 'wb') as f:
        f.write(SPM.SerializeToString())

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
    
def prepare_for_generation(text_data, model_path):
    
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


'''from sentencepiece import sentencepiece_model_pb2 as model

old = model.ModelProto()
old.ParseFromString(open("shortenSPM.model", 'rb').read())

new = model.ModelProto()
new.ParseFromString(open("amazon.model", 'rb').read())






extent(new, old)'''



#a,b,c,d = get_amazon_dataset()

#print(b, b.shape)
















'''from tensorflow.python.platform import gfile
model = gfile.GFile('amazon.model', 'rb').read()
sp = spm.SentencePieceProcessor(model_file=model)
vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
print(len(vocabs))'''

        
if __name__ == "__main__":
    #text_data = ["I love this movie so much <#br#>", "I hate this movie", "I love you"]
    with open("./rev_data.txt", "r") as f:
        text_data = f.readlines()
    prepare_for_generation(text_data, "./shortenSPM.model")

    








    

    
        

        
 
    

   

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