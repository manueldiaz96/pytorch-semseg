#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:06:47 2019

@author: manuel
"""

import os
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

def image_preproc(img, img_size):

    img = cv2.resize(img, (img_size[1], img_size[0]))
    
    img = img.astype(np.float64)
    img = img / 255.0 #Normalize
    
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    
    return img

def get_sem_mask(model_file_name):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #img_path = input('Image path: ')
    img_path = 'results/munich_000009_000019_leftImg8bit.png'
    
    if len(img_path):
        if img_path[-3:]=='png' or img_path[-3:]=='jpg':   
            print("Read Input Image from : %s"%(img_path))
        else:
            raise Exception('Non PNG or JPG image!')
        
    else:
        img_path = 'results/munich_000009_000019_leftImg8bit.png'
        
    img = cv2.imread(img_path)
    
    img_orig = img
    
    model_name = model_file_name[: model_file_name.find("_")]
    
    data_loader = get_loader('cityscapes')
    loader = data_loader(root=None, is_transform=True, test_mode=True)
    n_classes = loader.n_classes
    
    img = image_preproc(img, loader.img_size)
    
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version='cityscapes')
    
    try:
        state = convert_state_dict(torch.load(model_file_name)["model_state"])
    except: 
        state = convert_state_dict(torch.load(model_file_name, map_location='cpu')["model_state"])
        
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    
    images = img.to(device)
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    
    return pred, img_orig

def get_color_mask(pred, img):
    
    img = cv2.resize(img, (pred.shape[1], pred.shape[0]))

    classes = np.unique(pred)
    
    road = cv2.inRange(pred, 0, 0)
    
    dilatation_size = 2
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    road = cv2.dilate(road, element)
    
    road = cv2.cvtColor(road, cv2.COLOR_GRAY2RGB)
    
    maskedImg = cv2.bitwise_and(img, road)
    
    return maskedImg

model_file_name = 'segnet_cityscapes_best_model.pkl'

pred, img = get_sem_mask(model_file_name)

maskedImg = get_color_mask(pred, img)

plt.subplots(1,1, figsize=(20,20))
plt.imshow(maskedImg)

#plt.savefig('results/munich_000009_000019_leftImg8bit_road_.png')


'''
plt.subplots(4,4, figsize=(20,20))

for i in range(len(classes)):
    print(type(i))
    plt.subplot(4,len(classes)//4 + 1,i+1)
    img_pl = cv2.inRange(pred, i, i)
    plt.imshow(img_pl, cmap='gray')

    
plt.savefig('heatmaps.png')
'''