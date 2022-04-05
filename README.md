# FakeReviewDNN

## Description:
This is an ULMFIT-based NLP approach on the detection of online Fake-reviews as an example to show the implementation of ULMFIT in Tensorflow and the usage of the pretrained model by Hubert Karbowy and the endrone team. [https://bitbucket.org/edroneteam/tf2_ulmfit/src/master/]

The implementation and documentation was done by: Tim Kapferer @TimKapf, Sofia Worsfold @fiabox and Tim Petersen @Antim8

To get a better understanding of the hows and whys take a look at the paper-like file in the [Scientific background folder]

----

## Prerequisites
- Have a python environment with tensorflow installed [link]
- Install the requirements.txt with pip
  - be sure that your os and python version are supported for tensorflow_text, otherwise errors will occur
- clone the repository by endrone [link] into a folder of name "tf2_ulmit" otherwise the imports won't work
- if you want to graphically inspect the generated logs install tensorboard and start it with specific logdir [link]

---
## Structure:

| File/Folder | usage |
| --- |--- |
| logs | Here are example accuracies etc for some runs |
| amazon.model | lm trained on amazon review data |
| new_amazon.model | bettered amazon model |
| main.py | The main file for the user to interact |
| utils.py | helper functions |
| model_util.py | functions that help to train ULMFIT |
| fake_review_dataset.csv |  Our main dataset to train classifier |
| model.py | Our model wrapping the ULMFIT by edrone |
| rev_(clean)_data | fetched and cleaned Amazon review data of the official Tensorflow dataset |
| shortenSPM.model | shortened original vm from 35000 to around 4-5k sentencepieces |

---
## How it was trained
Check out the folder [Scientific background]

---
## How to use it

