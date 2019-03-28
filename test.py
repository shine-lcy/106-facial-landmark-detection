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


if len(sys.argv) < 2:
    print ("Call this program like this:\n"
        "python test.py /home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_test/0130.jpg")
    exit()


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
img_path=sys.argv[1]
#img_path = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_test/0130.jpg'

img = Image.open(img_path)

bounding_boxes, landmarks, flag = cal_landmark(img_path)

new_landmarks = []
#change order
for p in landmarks:
   new_p = fchange(p)
   new_landmarks.append(new_p)

landmark = new_landmarks[0]

a = show_bboxes(img, bounding_boxes, new_landmarks)
a.show()
