from src import detect_faces, show_bboxes,cal_landmark
from PIL import Image
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import numpy.random as npr
import sys
import utils as utils
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim
from data_load import my_dataset
from net import Net
size = 96

#data
labelpath = '/home/lc/106_resnet/96_landmark.txt'		#the ground truth
dataset = my_dataset(labelpath)
dataset.load()
dataLoader = DataLoader(dataset=dataset,batch_size=50,shuffle=True)
#net
resnet = models.resnet18(pretrained=True)
net = Net(resnet)

#train
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(2):
    train_loss = 0
    train_acc = 0
    net.float().cuda()
    net.train()
    for i, (im, label) in enumerate(dataLoader):
        im =im.type('torch.FloatTensor')
        im = Variable(im).cuda()
        label = Variable(label).cuda()


        out = net(im)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    eval_loss = 0

#     for im, label in testdataLoader:
#         im = Variable(im)
#         label = Variable(label)
#         out = net(im)
#         loss = criterion(out, label)
#         eval_loss += loss
    print('epoch: {}, Train Loss: {:.6f}'.format(e, train_loss / len(dataLoader)))
torch.save(net, 'my_net.pkl')
