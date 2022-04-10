""" This is the main script for user interaction with the program
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import tensorflow_text
from tensorflow.python.platform import gfile
from util import get_dataset
from random import randrange

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing 
def enablePrint():
    sys.stdout = sys.__stdout__

def main():

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    print("Hello there fellow human! Welcome to the Fake Review Detector!")
    print("Do you want to insert a review [0] or would you like to get a review [1]?")

    val = None
    while val not in ['0','1']:
        val = input("Press 0 for own review. Press 1 to get review: ")

    if val == '0':

        print("Please put your reviews into the \"ToAnalyze\" folder according to the Readme.md file in it")
        print("Once you are done, please hit a key to continue")
        input()
        print("Analyzing...")

        revs = []
    
        with open ("./ToAnalyze/revs.txt", "r") as f:
            for l in f:
                revs.append(str(l))

    else:

        print("Load model...")
        _, _, review, label = get_dataset()

        index = randrange(len(review))

        rev, r_label = review[index], label[index]

        revs = [rev]

    sols = []

    blockPrint()
    model = tf.keras.models.load_model('saved_model\classifier_model')
    spmodel = gfile.GFile('new_amazon.model', 'rb').read()
    tokenizer = tensorflow_text.SentencepieceTokenizer(spmodel,add_bos=True, add_eos=True)
    enablePrint()
    
    for rev in revs:
        sentence = tokenizer.tokenize(rev)
        sentence = tf.pad(sentence, paddings=[[0,256-tf.shape(sentence)[0]]], constant_values=1)
        sentence = tf.expand_dims(sentence, axis=0)

        sols.append(tf.round(model.predict(sentence)))
    
    print("Analyzing done")
    if val == '1':

        print("Review: " + review[index])
        print("\n")

        if r_label == 0:
                label_str = "Original Review"
        else:
            label_str = "Computer generated Review"

        if sols[0] == 0:

            print("Prediction: Original Review, Label: " + label_str)
            if sols[0] == r_label:
                print("Correct classification.")
            else:
                print("Wrong classification.")
        else:
            print("Prediction: Computer generated Review, Label: " + label_str)
            if sols[0] == r_label:
                print("Correct classification.")
            else:
                print("Wrong classification.")
    else:
        labels_str = []
        for e in sols:
            if 1 in e:
                labels_str.append("Computer generated Review")
            else:
                labels_str.append("Original Review")


        with open ("./ToAnalyze/results.txt", "w") as f:
            for i in range(len(revs)):
                f.write(f"{revs[i]}" + labels_str[i] + "\n")

        print("The results are in the \"ToAnalyze/result.txt\" file")
            
    

if __name__ == "__main__":
    main()