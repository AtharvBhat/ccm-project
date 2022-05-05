#imports
import PIL
from PIL import Image
import sys
import os
import numpy as np

from get_item_attributes import get_item_attributes

#change dir to this dir
os.chdir(os.getcwd())


#This is stupid. but its easy and it works.
paths = ["canary/red",
        "canary/yellow",
        "daisy/red",
        "daisy/yellow",
        "oak/red",
        "oak/green",
        "pine/green",
        "robin/yellow",
        "robin/red",
        "rose/red",
        "rose/yellow",
        "salmon/gray",
        "salmon/red",
        "sunfish/"]

names_items, names_relations, names_attributes, items, relations, attributes = get_item_attributes()

train = []
validation = []


#get each image, convert it to a numpy array and store it in a dictionary
for path in paths:
    item, color = path.split('/')
    if os.name == 'nt':
        files = os.listdir("data/" + path + "/")
    else:
        files = os.listdir(path + "/")
    data_list = []

    #open all files
    for file in files:
        try:
            if os.name == 'nt':
                image = Image.open("data/"+path+"/"+file)
            else:
                image = Image.open(path+"/"+file)
        except Exception as e:
            print(f"couldnt open file {file} because {e}. skipping !")
        image = np.array(image)

        name_idx = np.where(names_items == item.capitalize())
        item_idx = np.where(items[:,name_idx] == 1)[0]

        #get 4 different patterns for each image
        for i, r, a in zip(items[item_idx], relations[item_idx], attributes[item_idx]):
            if color in ("red", "yellow", "green"):
                a[np.array([18, 19, 20])] = 0
                color_idx = np.where(names_attributes == color.capitalize())
                a[color_idx] = 1
            data = {"image" : image, "item": item, "color":color, "relation": r, "attribute":a}
            data_list.append(data)
    np.random.shuffle(data_list)
    validation += data_list[:5]
    train += data_list[5:]

np.random.shuffle(train)
np.random.shuffle(validation)

print(f"Number of train samples :{len(train)}")
print(f"Number of validation samples :{len(validation)}")

import pickle as pkl

with open('data/train.pkl', 'wb') as f:
    pkl.dump(train, f)

with open('data/validation.pkl', 'wb') as f:
    pkl.dump(validation, f)