import torch
import torch.nn as nn
from tqdm.notebook import tqdm

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
        for i, (x, y) in tqdm(enumerate(dataloader)):
            image = x[0]
            relation = x[1]
            image, relation, y = image.to(device), relation.to(device), y.to(device)
            output, hidden, rep = model(image, relation)
            loss = criterion(output, y)
            total_error+= loss.item()
    return total_error/len(dataloader)