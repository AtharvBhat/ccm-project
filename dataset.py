from statistics import variance
import torch
import torchvision.transforms as T
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
import math

class CcmDataset(Dataset):

    def __init__(self, path, mean=0, std=0, k=None, b_std=None):
        self.data = None

        #transforms
        self.transform = T.Compose([T.ToTensor(),
                                    T.Resize((224,224)),
                                    AddGaussianNoise(mean, std),
                                    T.GaussianBlur(k, b_std)]) if k is not None and b_std is not None else T.Compose([T.ToTensor(),
                                                                                                                    T.Resize((224,224)),
                                                                                                                    AddGaussianNoise(mean, std)])

        #load data
        with open(path,'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]["image"]
        relation = self.data[i]["relation"]
        attribute = self.data[i]["attribute"]

        return (self.transform(image) , torch.tensor(relation, dtype=torch.float32), self.data[i]["item"], self.data[i]["color"]), torch.tensor(attribute, dtype=torch.float32)
            
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.mean == None:
            self.mean = tensor.mean()
        return torch.clip(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0, 1)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)