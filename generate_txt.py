import os
from src import detect_faces, show_bboxes, cal_landmark
from PIL import Image
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from change_order import fchange
import numpy as np
import cv2
import numpy.random as npr
import sys
import utils as utils
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim
from cal_loss import mean_error
size = 96
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.fc = nn.Linear(4608, 212)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


pic_path = '/media/lc/My Passport/20181129face_data/all/'
txt_path = '/media/lc/My Passport/20181129face_data/txt/'
#new_path = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_dev/'
#old_path  = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/dev/'
for dirs,sub,pics in os.walk(pic_path):
    for pic in pics:
        img_path = pic_path + pic
        img = Image.open(img_path)
        bounding_boxes, landmarks, flag = cal_landmark(img_path)
        print img_path, flag
        if flag == 0:
            continue
        f = open(txt_path+pic.split('.')[0]+'.txt','a')
        s = landmarks[0]
        for p in s:
            f.write(str(p))
            f.write(' ')