import torch
import torchvision.transforms as T
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset

class CcmDataset(Dataset):

    def __init__(self, path):
        self.data = None

        #transforms
        self.transform = T.Compose([T.ToTensor(),
                                    T.Resize((224,224))])

        #load data
        with open(path,'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]["image"]
        relation = self.data[i]["relation"]
        attribute = self.data[i]["attribute"]

        return (self.transform(image), torch.tensor(relation, dtype=torch.float32)), torch.tensor(attribute, dtype=torch.float32)
            