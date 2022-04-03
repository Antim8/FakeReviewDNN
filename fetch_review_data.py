import tensorflow_datasets as tfds
import random

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