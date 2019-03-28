from src import detect_faces, show_bboxes
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

img_path = '/home/lc/106_resnet/show/2.jpg'
img = Image.open(img_path)
bounding_boxes, _, x = detect_faces(img)
print bounding_boxes
bb = bounding_boxes[0]
print bb
