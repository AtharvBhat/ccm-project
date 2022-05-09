import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from dataset import CcmDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

def train_one_epoch(model : nn.Module, criterion : nn.MSELoss, optimizer : torch.optim, device : torch.device, dataloader : torch.utils.data.DataLoader):
    model = model.to(device)
    model.train()
    total_error = 0
    for i, (x, y) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        image = x[0]
        relation = x[1]
        image, relation, y = image.to(device), relation.to(device), y.to(device)
        output, hidden, rep = model(image, relation)
        loss = criterion(output, y)
        total_error+= loss.item()
        loss.backward()
        optimizer.step()
    return total_error/len(dataloader)


def validation(model : nn.Module, criterion : nn.MSELoss, device : torch.device, dataloader : torch.utils.data.DataLoader):
    model = model.to(device)
    model.eval()
    total_error = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            image = x[0]
            relation = x[1]
            image, relation, y = image.to(device), relation.to(device), y.to(device)
            output, hidden, rep = model(image, relation)
            loss = criterion(output, y)
            total_error+= loss.item()
    return total_error/len(dataloader)


def get_rep(model : nn.Module, mean=0, std=0, k=None, b_std=None):
    # Extract the hidden activations on the Representation Layer for each item
    # 
    # Input
    #  model : Net class object
    #
    # Output
    #  rep : dict{item: avg representation vector for validaiton set}, where each row is an item
    representations = {}
    validaiton_data = CcmDataset("data/validation.pkl", mean, std, k, b_std)
    model.eval()
    for item in validaiton_data:
        x, y = item
        image, relation, item, color = x[0], x[1], x[2], x[3]
        output, hidden, rep = model(image.unsqueeze(0), relation.unsqueeze(0))

        key = f"{item}+{color}"
        if key in representations:
            representations[key][0] += rep.detach()[0].numpy()
            representations[key][1] += 1
        else:
            representations[key] = [rep.detach()[0].numpy(), 1]
        
    for item in representations:
        representations[item] = representations[item][0]/representations[item][1]
   
    return representations

def cumulate_reps(reps):
    names = list(reps[0].keys())
    names_out = [f"{x.split('+')[1]} {x.split('+')[0]}".strip() for x in names]
    out_reps = []
    for rep in reps:
        np_rep = []
        for name in names:
            np_rep.append(rep[name])
        out_reps.append(np.array(np_rep))
    return out_reps, names_out



def plot_rep(rep1, rep2, rep3, rep4, rep5, names):
    #  Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    #  using bar graphs
    # 
    #  Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    #  names : [nitem list] of item names
    #
    nepochs_list = list(range(3,20,4))
    nrows = len(names)
    R = np.dstack((rep1,rep2,rep3, rep4, rep5))    
    mx = R.max()
    mn = R.min()
    depth = R.shape[2]
    count = 1
    plt.figure(1,figsize=(25,25))
    for i in range(nrows):
        for d in range(R.shape[2]):
            plt.subplot(nrows, depth, count)
            rep = R[i,:,d]
            plt.bar(range(rep.size),rep)
            plt.ylim([mn,mx])
            plt.xticks([])
            plt.yticks([])        
            if d==0:
                plt.ylabel(names[i])
            if i==0:
                plt.title("epoch " + str(nepochs_list[d]))
            count += 1
    plt.show()

def plot_dendo(rep1,rep2,rep3,names):
    #  Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    #  using hierarchical clustering
    # 
    #  Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    #  names : [nitem list] of item names
    #
    nepochs_list = list(range(9,20,5))
    linked1 = linkage(rep1,'single')
    linked2 = linkage(rep2,'single')
    linked3 = linkage(rep3,'single')
    mx = np.dstack((linked1[:,2],linked2[:,2],linked3[:,2])).max()+0.1    
    plt.figure(2,figsize=(25,25))
    plt.subplot(3,1,1)    
    dendrogram(linked1, labels=names, color_threshold=0)
    plt.ylim([0,mx])
    plt.title('Hierarchical clustering; ' + "epoch " + str(nepochs_list[0]))
    plt.ylabel('Euclidean distance')
    plt.subplot(3,1,2)
    plt.title("epoch " + str(nepochs_list[1]))
    dendrogram(linked2, labels=names, color_threshold=0)
    plt.ylim([0,mx])
    plt.subplot(3,1,3)
    plt.title("epoch " + str(nepochs_list[2]))
    dendrogram(linked3, labels=names, color_threshold=0)
    plt.ylim([0,mx])
    plt.show()