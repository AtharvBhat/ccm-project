import torch
import torch.nn as nn

def train_one_epoch(model : nn.Module, criterion : nn.MSELoss, optimizer : torch.optim, device : torch.device, dataloader : torch.utils.data.DataLoader, print_freq=50):
    model = model.to(device)
    model.train()
    total_error = 0
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        image = x[0]
        relation = x[1]
        image, relation = image.to(device), relation.to(device)
        output, hidden, rep = model(image, relation)
        loss = criterion(output, y)
        total_error+= loss.item()
        if i%print_freq == 0:
            print(f"loss : {loss}")
        loss.backward()
        optimizer.step()
    return total_error/len(dataloader)

