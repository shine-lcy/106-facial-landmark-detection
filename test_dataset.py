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


if len(sys.argv) < 5:
    print ("Call this program like this:\n"
        "python test_dataset.py /home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_test/ /home/lc/cy/hourglass-facekeypoints-detection/datasets/test/test_out/new_test_result/ /home/lc/cy/hourglass-facekeypoints-detection/datasets/test/test_out/new_landmark/ /home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_500_txt/")
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

imgpath=sys.argv[1]
new_imgpath=sys.argv[2]
landmark_out=sys.argv[3]
txt_path=sys.argv[4]

#imgpath = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_test/'				
#new_imgpath = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/test_out/new_test_result/'
#landmark_out = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/test_out/new_landmark/'
#txt_path = '/home/lc/cy/hourglass-facekeypoints-detection/datasets/test/new_500_txt/'    
i = 0
loss = 0
for root, dirs, imgs in os.walk(imgpath):
    num = len(imgs)
    print num
    for img in imgs:
        i = i+1
        print i
        # if i<369:
        #     continue
        img_path = imgpath + img
        print img_path
        #print img_path
        landmark_out_name = os.path.join(landmark_out, img.replace("jpg", "txt"))
        txt_f = os.path.join(txt_path, img.replace("jpg","txt"))
        f_txt = open(txt_f, 'r')
        txt = f_txt.readline().strip().split()
        im = Image.open(img_path)
        print 'beginning'
        bounding_boxes, landmarks, flag = cal_landmark(img_path)
        #print flag
        if flag == 0:
            continue
        #print landmarks[0]

        #a.show()

        new_landmark = []
        for p in landmarks:
            p = fchange(p)
            new_landmark.append(p)
        landmark_list = new_landmark[0]
        a = show_bboxes(im, bounding_boxes, new_landmark)
        a.save(new_imgpath+img)
        f = file(landmark_out_name, "w")
        for k in range(106):
            x = landmark_list[2*k]
            y = landmark_list[2*k+1]
            f.write("{0}".format(int(x)))
            f.write(' ')
            f.write("{0}".format(int(y)))
            f.write(' ')
        f.write('\n')
        loss_i = mean_error(landmark_list, txt)
        loss = loss + loss_i

        if i > 501:
           break
    mean_loss = loss/i
    print loss, mean_loss

    break

