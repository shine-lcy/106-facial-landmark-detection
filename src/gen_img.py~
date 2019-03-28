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
size = 96

def gen_img(txt_path,annotation,l_idx):
    annotation = annotation.strip().split(' ')
    assert len(annotation)==212,"each line should have 212 element"
    im_path = os.path.join(imgpath,txt_path.replace("txt", "jpg"))
    img = cv2.imread(im_path)
    img = cv2.flip(img,1)
    assert (img is not None)
    height, width, channel = img.shape
   
    landmark = list(map(float, annotation))
    for i in range(106):
        landmark[2*i] = width - landmark[2*i]
    landmark = np.array(landmark, dtype=np.float)
    X = []
    Y = []
    count = 0
    for i in landmark:
        if count % 2 == 0:
            X.append(i)
        else:
            Y.append(i)
        count += 1
    x1 = min(X)
    y1 = min(Y)
    x2 =  max(X)
    y2 = max(Y)
    gt_box = [x1,y1,x2,y2]
    gt_box = np.array(gt_box, dtype=np.int32)
    
    crop_face = img[max(0,int(gt_box[1])-20):min(width,int(gt_box[3])+21), max(0,int(gt_box[0])-20):min(height,int(gt_box[2])+21)]
    crop_face = cv2.resize(crop_face,(size,size))
    # gt's width
    w = x2 - x1
    # gt's height
    h = y2 - y1
    # random shift
    for i in range(20):
        bbox_size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
        delta_x = npr.randint(-w * 0.2, w * 0.2)
        delta_y = npr.randint(-h * 0.2, h * 0.2)
        nx1 = int(max(x1 + w / 2 - bbox_size / 2 + delta_x, 0))
        ny1 = int(max(y1 + h / 2 - bbox_size / 2 + delta_y, 0))

        nx2 = int(nx1 + bbox_size)
        ny2 =int(ny1 + bbox_size)
        if nx2 > width or ny2 > height:
            continue
        crop_box = np.array([nx1, ny1, nx2, ny2])
        #print(crop_box)
        cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
        resized_im = cv2.resize(cropped_im, (96, 96),interpolation=cv2.INTER_LINEAR)

        offset_x1 = (x1 - nx1) / float(bbox_size)
        offset_y1 = (y1 - ny1) / float(bbox_size)
        offset_x2 = (x2 - nx2) / float(bbox_size)
        offset_y2 = (y2 - ny2) / float(bbox_size)

        # cal iou
        iou = utils.IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
        if iou > 0.65:
            save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx)
            cv2.imwrite(save_file, resized_im)
            f.write(save_file)
            for i in range(106):
                xi = (landmark[i*2] - nx1) / float(bbox_size)
                yi = (landmark[i*2+1] - ny1) / float(bbox_size)
                f.write(" {0} {1}" .format(xi, yi))
            f.write('\n')

            l_idx += 1
    return l_idx

imgpath = '/home/lc/cy/106_demo/img/'
landmark = '/home/lc/cy/106_demo/landmark/'
save_landmark_anno = '/home/lc/cy/106_demo/1.txt'
landmark_imgs_save_dir = '/home/lc/cy/106_demo/new_img'
f = open(save_landmark_anno, 'w')
l_idx = 0
for dirs,sub,txts in os.walk(landmark):
    num = len(txts)
    print("%d total images" % num) 
    for txt in txts:
        with open(landmark+txt, 'r') as f2:
             annotation=f2.read()
             l_idx = gen_img(txt,annotation,l_idx) 
        f2.close() 
        idx +=1
        if idx % 100 == 0:
            print("%d images done, landmark images: %d"%(idx,l_idx))
f.close()
