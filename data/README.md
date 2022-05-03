Instructions to get data working :- 

* Run `generate_data.py` The script should generate training and validation splits
* Each split is a list of dictionaries where each dictionary contains the image in numpy format and its relation and attributes.
* This can now be converted into a `Dataset` class and a pytorch `Dataloader`.