import torch
from torch import nn
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim
class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        
        self.fc = nn.Linear(4608,212)
       
        
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    


