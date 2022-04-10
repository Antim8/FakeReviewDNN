""" This is the main script for user interaction with the program
"""
import tensorflow as tf
import tensorflow_text
from tensorflow.python.platform import gfile

def main():
    
    print("Hello there fellow human! Welcome to the Fake Review Detector!")
    print("Please put your reviews into the \"ToAnalyze\" folder according to the Readme.md file in it")
    print("Once you are done, please hit a key to continue")
    input()
    print("Analyzing...")
    
    revs = []
    
    with open ("./ToAnalyze/revs.txt", "r") as f:
        for l in f:
            revs.append(str(l))
            
    print("Reviews read")
    
    
    sols = []

    model = tf.keras.models.load_model('saved_model\classifier_model')
    spmodel = gfile.GFile('new_amazon.model', 'rb').read()
    tokenizer = tensorflow_text.SentencepieceTokenizer(spmodel,add_bos=True, add_eos=True)
    
    for rev in revs:
        sentence = tokenizer.tokenize(rev)
        sentence = tf.pad(sentence, paddings=[[0,256-tf.shape(sentence)[0]]], constant_values=1)
        sentence = tf.expand_dims(sentence, axis=0)

        sols.append(tf.round(model.predict(sentence)))
    
    print("Analyzing done")
    
    with open ("./ToAnalyze/results.txt", "W") as f:
        for i in range(len(revs)):
            f.write(f"{revs[i]}" + " isFake: " + str(sols[i]) + "\n")
            
    print("The results are in the \"ToAnalyze/result.txt\" file")


if __name__ == "__main__":
    main()