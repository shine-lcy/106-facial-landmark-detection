from torch.utils.data import Dataset
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
from net import Net


class my_dataset(Dataset):
    def __init__(self, labelpath):
        self.img_list = [] 
        self.label_list = []
        self.labelpath = labelpath
        
    def load(self):
        with open(self.labelpath, 'r') as f2:
            annotations = f2.readlines()
        img_list = []
        label_list = []
        for annotation in annotations:
           
            img_path = annotation.split(' ')[0]
            img = cv2.imread(img_path)
            h,w,c = img.shape
            assert c==3, "munsy"
            img_list.append(img) 
            label =  annotation.strip().split(' ')[1:]
            assert len(label)==212, annotation
            label = np.array(label, dtype=np.float32)
            label_list.append(label)
        self.img_list = img_list
        self.label_list = label_list 
        return img_list,label_list

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = img.transpose((2,0,1))
        label = self.label_list[idx]
        return img, label

    def __len__(self):
        return len(self.label_list)


