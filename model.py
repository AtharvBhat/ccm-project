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
        self.bottleneck = nn.Linear(512, num_rep)
        #representation layer
        self.representation_layer = nn.Linear(num_rep, num_rep)
        #hidden layer
        self.hidden_layer = nn.Linear(num_rep + 4, num_hidden)
        #classifier
        self.classifier = nn.Linear(num_hidden, 36)
        self.softmax = nn.Softmax(dim=-1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(num_rep)
        self.bn3 = nn.BatchNorm1d(num_rep)
        self.bn4 = nn.BatchNorm1d(num_hidden)

    def forward(self, image, relation):
        cnn_features = F.relu(self.bn1(self.cnn_extractor(image)))
        bottleneck = self.softmax(self.bn2(self.bottleneck(cnn_features)))
        rep = F.relu(self.bn3(self.representation_layer(bottleneck)))
        hidden = F.relu(self.bn4(self.hidden_layer(torch.cat((rep, relation),1))))
        output = torch.sigmoid(self.classifier(hidden))

        return output, hidden, rep



