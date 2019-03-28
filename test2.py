#!/usr/bin/python
import os
import sys
sys.path.append("/home/lc/dlib-19.6")
sys.path.append("/home/lc/dlib-19.6/python_examples")
import glob

from train_shape_predictor import get_detection
import cv2

def get_detect_faces(img_path):
    x_list, y_list = get_detection(img_path)
    point_list = []
    point_list.append(min(x_list))
    point_list.append(min(y_list))
    point_list.append(max(x_list))
    point_list.append(max(y_list))
    return point_list

a = get_detect_faces("/home/lc/106_resnet/show/2.jpg")
print a 

