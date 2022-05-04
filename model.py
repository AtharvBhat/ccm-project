import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    
    def __init__(self, num_rep, num_hidden) -> None:
        super().__init__()
        #cnn extractor instead of onehot encoded inputs
        self.cnn_extractor = torchvision.models.resnet18(pretrained=True)
        self.cnn_extractor.fc = nn.Identity()
        #representation layer
        self.representation_layer = nn.Linear(512, num_rep)
        #hidden layer
        self.hidden_layer = nn.Linear(num_rep + 4, num_hidden)
        #classifier
        self.classifier = nn.Linear(num_hidden, 36)


    def forward(self, image, relation):
        cnn_features = F.relu(self.cnn_extractor(image))
        rep = F.relu(self.representation_layer(cnn_features))
        hidden = F.relu(self.hidden_layer(torch.cat((rep, relation),1)))
        output = torch.sigmoid(self.classifier(hidden))

        return output, hidden, rep



