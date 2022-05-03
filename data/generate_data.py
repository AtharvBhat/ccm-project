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

#get each image, convert it to a numpy array and store it in a dictionary
for path in paths:
    item, color = path.split('/')
    files = os.listdir("data/" + path + "/")
    
    #open all files
    for file in files:
        image = Image.open("data/"+path+"/"+file)
        image = np.array(image)
        
        
        
        