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
import cmath

size = 96


def mean_error(landmark, txt):
	error = 0
	eye_d = cmath.sqrt((int(txt[210])-int(txt[208]))**2+(int(txt[211])-int(txt[209]))**2)
	#print eye_d
	for i in range(106):
		tmp = cmath.sqrt((landmark[2*i]-int(txt[2*i]))**2+(landmark[2*i+1]-int(txt[2*i+1]))**2)
		#print i
		tmp = tmp/eye_d
		#print tmp
		error = error + tmp
	mean_error = error/106
	return mean_error
	
	
