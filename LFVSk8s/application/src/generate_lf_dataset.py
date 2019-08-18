#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:21:10 2019

@author: rohit
"""

import numpy as np
import math
import cv2
import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

lf_dataset = []
lf = []



def get_data_config(config):
    lf_dim, lf_dataset_new = get_data(config)
    H = W = lf_dim * config.patch_size
    lf_dataset_new = np.reshape(lf_dataset_new, (-1, H, W,config.channels))
    print ('lf_dataset_total : {}'.format(lf_dataset_new.shape))

    mean = np.mean(lf_dataset_new)
    std = np.std(lf_dataset_new,dtype=np.float64)
    lf_dataset_new = (lf_dataset_new - mean)/(std)
    return lf_dim, lf_dataset_new
    

def get_data(config):
    base_dir = config.base_dir
    patch_size = config.patch_size
    strides = config.strides
    all_folders = get_all_data_folders(base_dir)
    lf_dataset_new = []

    num_files = 0

    for data_folder in all_folders:
        
        num_files = num_files + 1
        print ('num_files : {}'.format(num_files))
        
        print (data_folder)
        views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])
#        print ('views : {}'.format(np.shape(views)))
        lf = lf_from_Ychannel_only_new(data_folder, views, config)
        print ('lf : {}'.format(np.shape(lf)))
        lf_dataset = lf_from_patches_new(lf, patch_size, strides)
        print ('lf_dataset : {}'.format(np.shape(lf_dataset)))
        lf_dataset_new.append(lf_dataset)
        if (num_files == config.num_files):
            print ("going_out")
            break

    
    lf_dim = int(math.sqrt(lf.shape[0]))

    lf_dataset_new = np.array(lf_dataset_new)

    return lf_dim, lf_dataset_new

def get_lr_loss_list(lr_path, loss_path):
    try:
        lr_list = np.load(lr_path)
        lr_list = lr_list.tolist()
    except:
        lr_list = []

    try:
        loss_list = np.load(loss_path)
        loss_list = loss_list.tolist()
    except:
        loss_list = []
        
    return lr_list, loss_list


def get_all_data_folders(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    data_folders = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for category in categories:
        for scene in os.listdir(os.path.join(base_dir, category)):
            data_folder = os.path.join(*[base_dir, category, scene])
            if os.path.isdir(data_folder):
                data_folders.append(data_folder)

    return data_folders

def lf_from_Ychannel_only_new(data_folder, views,config):
    lf = []
    view_count = 0
    for view in views: 
        fpath = os.path.join(data_folder, view)
        view_count = view_count + 1
        img = cv2.imread(fpath)
#        img=mpimg.imread(fpath)
        if (config.channels == 1):
            img_Y = np.array((cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))[:,:,0])
        else:
            img_Y = img
        img_Y = np.reshape(img_Y, (np.shape(img_Y)[0],np.shape(img_Y)[1],config.channels))

        lf.append(img_Y)
#        if(view_count >=  9):
#            break  
    return np.array(lf)
  
    
def lf_from_slice_new(slice_):
    h = slice_.shape[1]
    w = slice_.shape[2]
    d = slice_.shape[0]
    c = slice_.shape[3]
    
    n_view = int(math.sqrt(d))
    
    col_lf = np.array([], dtype=np.uint8).reshape(0,w*n_view,c)
    
#    print ('h : {} \n w : {} \n depth : {} \n channel : {} \n n_view : {} \n'.format(h,w,d,c,n_view))
#    print ('col_lf : {}'.format(np.shape(col_lf)))
    
    for j in range(n_view):
        row_lf = np.array([], dtype=np.uint8).reshape(h,0,c)
#        print ('row_lf : {}'.format(np.shape(row_lf)))
        for i in range(n_view):
#            print ('index : {}'.format((n_view*j)+i))
            single_image = slice_[(n_view*j)+i,:,:,:]
#            print ('single_image : {}'.format(np.shape(single_image)))
            row_lf = np.hstack((row_lf,single_image))
#            cv2.imshow('row_only', slice_[:,:,(n_view*j)+i])
#            cv2.imshow('row_lf', row_lf)
#            cv2.waitKey(-1)

            
        col_lf = np.vstack((col_lf, row_lf))
#        cv2.imshow('col_lf', col_lf)
#        cv2.waitKey(-1)        
        
    return col_lf
    

def lf_from_patches_new(lf, patch_size, strides):
    lf_dataset = []
    
    H = np.shape(lf)[1]
#    W = np.shape(lf)[2]

    max_row_shift = max_col_shift = ((H - patch_size)//strides) + 1 
    
    row = col = 0
    for col_shift in range(0,max_col_shift):
        for row_shift in range(0, max_row_shift):
#            print ('row: {}  col:{}'.format(row,col))
            slice_3d =   lf[:,row:row + patch_size,col:col + patch_size ,:]    
#            print ('slice_3d_shape : {}'.format(np.shape(slice_3d)))
            patch_lf = lf_from_slice_new(slice_3d)
#            print ('patch_lf_shape : {}'.format(np.shape(patch_lf)))
            lf_dataset.append(patch_lf)
            col = col + strides
        row = row + 8
        col = 0
        
    return np.array(lf_dataset)

#def lf_from_slice(slice_):
#    h = slice_.shape[0]
#    w = slice_.shape[1]
#    d = slice_.shape[2]
#    
#    n_view = int(math.sqrt(d))
#    
#    col_lf = np.array([], dtype=np.uint8).reshape(0,w*n_view)
#    
#    for j in range(n_view):
#        row_lf = np.array([], dtype=np.uint8).reshape(h,0)
#        for i in range(n_view):
##            print ('index : {}'.format((n_view*j)+i))
#            row_lf = np.hstack((row_lf, slice_[:,:,(n_view*j)+i]))
##            cv2.imshow('row_only', slice_[:,:,(n_view*j)+i])
##            cv2.imshow('row_lf', row_lf)
##            cv2.waitKey(-1)
#
#            
#        col_lf = np.vstack((col_lf, row_lf))
##        cv2.imshow('col_lf', col_lf)
##        cv2.waitKey(-1)        
#        
#    return col_lf


#
#def lf_from_patches(lf, patch_size, strides):
#    
#    H = np.shape(lf)[0]
##    W = np.shape(lf)[1]
#
#    max_row_shift = max_col_shift = ((H - patch_size)//strides) + 1
#    
#    row = col = 0
#    for col_shift in range(0,max_col_shift):
#        for row_shift in range(0, max_row_shift):
#            slice_3d =   lf[row: row + patch_size,col: col + patch_size ,:]    
#            patch_lf = lf_from_slice(slice_3d)
#            lf_dataset.append(patch_lf)
#            col = col + strides
#        row = row + 8
#        col = 0
#        
#    return lf_dataset
#
#def lf_from_Ychannel_only(string_paths):
#    view_count = 0
#    for path in string_paths: 
#        view_count = view_count + 1
#        img = cv2.imread(path)
#        img_Y = np.array((cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))[:,:,0])
##    cv2.putText(img_Y,'{}'.format(view_count),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
##    cv2.imshow('img', img_Y)
##    cv2.waitKey(-1)
#        lf.append(img_Y)
#        if(view_count >=  9):
#            break
#        
#    return lf
#        
#def get_pathlist(pathlist):
#    string_paths = []
#    for path in pathlist:
#        path_in_str = str(path)
##    print (path_in_str)
#        string_paths.append(path_in_str)
#   
#    string_paths.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
#    return string_paths
#

