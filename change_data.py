import json
from black import out
import pandas as pd
import ijson
import gzip
import random

#data = pd.read_json('D:/Dev/Datasets/All_Amazon_Review.json')
"""
data = []

with open("D:/Dev/Datasets/All_Amazon_Review.json", "r") as f:
    for product in ijson.items(f, "item"):
        review = product["reviewText"]
        data.append(review)
        print(review)
"""

### load the meta data

data = []
with gzip.open('C:/Users/peter/Downloads/AMAZON_FASHION.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# total length of list, this number equals total number of products


# first row of the list
#print(data[1]["reviewText"])

output = []
for i in range(0, len(data)):
    try: 
        output.append(data[i]["reviewText"].replace("\n", " "))
    except:
        pass
    
output = random.sample(output, 2000)
    
#print(repr(output[2]))
#print(repr(max(output, key=len)))

    
# write to a file
with open("./AMAZON_FASHION_REVIEWS.txt", "w") as f:
    for i in range(0, len(output)):
        f.write(output[i])
        f.write("\n")

