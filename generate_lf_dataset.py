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
import tensorflow as tf
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

lf_dataset = []
lf = []

def load_patched_data(path,num_files):
    print (path+"1.npy")
    data = np.load(path+"1.npy")
    print ("block shape : {}".format(data.shape))
    for i in range(2,num_files+1):
        print ("path : {}".format(path + str(i) + ".npy"))
        block = np.load(path + str(i) + ".npy")
        print ("block shape : {}".format(block.shape))
        data = np.append(data, block, axis = 0)
    return data

def load_data(config,path, dict_name, start, end):
    dict_path = path + dict_name + ".npy" 
    dict_ = np.load(dict_path)
    print (dict_)
    data = np.empty((0,1152,1152,1), np.float64)

    for index in range(start,end+1):#config.num_files):
#        index = index + 1
        name = dict_.item().get(index)+".npy"
        full_name = path + name
        print (full_name)
        block = np.load(full_name)
        data = np.append(data,block, axis = 0)
        print (data.shape)
       
    return data


def get_list_save_paths(config, lr_path):
    lr_save_path = config.save_path + lr_path + 'lr.npy'
    loss_path = config.save_path + lr_path + 'loss.npy'
    loss_train_path = config.save_path + lr_path + 'loss_train.npy'
    test_output_path = config.save_path + lr_path
    return lr_save_path,loss_path, loss_train_path, test_output_path
    

def write_model_information(FLAGS, file_path, lr,hidden_dims):
    with open(file_path,"w") as f:
        f.write("patch_size : " + str(FLAGS.patch) + "\n")
        f.write("channels : " + str(FLAGS.channels)+ "\n")
        f.write("allowed_lf_dim : " + str(FLAGS.allowed_lf_dim)+ "\n")
        f.write("batch_size : " + str(FLAGS.batch_size)+ "\n")
        f.write("num_files : " + str(FLAGS.num_files)+ "\n")
        f.write("learning_rate : " + str(lr)+ "\n")
        f.write("gradient_clip : " + str(FLAGS.gradient_clip)+ "\n")
        f.write("strides : " + str(FLAGS.strides)+ "\n")
        f.write("epochs : " + str(FLAGS.epochs)+ "\n")
        f.write("model : " + str(FLAGS.model)+ "\n")
        f.write("save_steps : " + str(FLAGS.save_steps)+ "\n")
        f.write("save_output_steps : " + str(FLAGS.save_output_steps)+ "\n")
        f.write("lstm_input : " + str([hidden_dims[0],hidden_dims[1],hidden_dims[2]])+ "\n")


def get_data_config(config):
    lf_dim, lf_dataset_new = get_data(config)
    H = W = lf_dim * config.patch_size
    lf_dataset_new = np.reshape(lf_dataset_new, (-1, H, W,config.channels))
    print ('lf_dataset_total : {}'.format(lf_dataset_new.shape))

    lf_dataset_new = normalize_image(lf_dataset_new)
    return lf_dim, lf_dataset_new

def get_data_config_new(config, save_path):
    lf_dim, lf_dataset_new = get_data_new(config,save_path)
    H = W = lf_dim * config.patch_size
    lf_dataset_new = np.reshape(lf_dataset_new, (-1, H, W,config.channels))
    print ('lf_dataset_total : {}'.format(lf_dataset_new.shape))
    
    return lf_dim, lf_dataset_new
    
def normalize_image(lf_dataset_new):
    mean = np.mean(lf_dataset_new)
    std = np.std(lf_dataset_new,dtype=np.float64)
    lf_dataset_new = (lf_dataset_new - mean)/(std)
    return lf_dataset_new

def get_data_new(config,save_path):
    base_dir = config.base_dir
    patch_size = config.patch_size
    strides = config.strides
    all_folders = get_all_data_folders(base_dir)
    lf_dataset_new = []

    num_files = 0
    train_filename_dict = {}

    for data_folder in all_folders:
        
        num_files = num_files + 1
#        print ('num_files : {}'.format(num_files))
        
        print (data_folder)
        name_folder = data_folder.split("/")[-1]
        print ("name_folder : {}".format(name_folder))
        train_filename_dict[num_files] = name_folder
        views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])
#        print ('views : {}'.format(np.shape(views)))
        lf = lf_from_Ychannel_only_new(data_folder, views, config)
        lf_dim = int(math.sqrt(lf.shape[0]))
#        print ('lf : {}'.format(np.shape(lf)))
#        lf_dataset = lf_from_patches_new(lf, patch_size, strides)
#        print ('lf_dataset_after_patches : {}'.format(np.shape(lf_dataset)))
#        
#        lf_dataset = normalize_image(lf_dataset)
#        H = W = lf_dim * config.patch_size
#        lf_dataset = np.reshape(lf_dataset, (-1, H, W,config.channels))
#        
#        if not os.path.exists(save_path + "{}.npy".format(name_folder)):
#            np.save(save_path + "{}.npy".format(name_folder),lf_dataset)
#            print ("Saved : {}".format(save_path + "{}.npy".format(name_folder)))
#        
#        lf_dataset_new.append(lf_dataset)
#        if (num_files == config.num_files):
#            print ("going_out")
#            break
        
    np.save("{}train_filename_dict.npy".format(save_path),train_filename_dict)
    lf_dataset_new = np.array(lf_dataset_new)

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

def get_lr_loss_list(lr_path, loss_path,loss_train_path):
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

    try:
        loss_train_list = np.load(loss_train_path)
        loss_train_list = loss_train_list.tolist()
    except:
        loss_train_list = []
                
    return lr_list, loss_list, loss_train_list


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
    
#    print ("H : {}  patch: {}   strides  {}".format(H,patch_size, strides))
#    print ("max_row_shift : {}".format(max_row_shift))
    
    
    row = col = 0
    for col_shift in range(0,max_col_shift):
#        print ("row : {}:{}   col:{}".format(row, row+patch_size, col))
        if (row+patch_size > H):
            break
        for row_shift in range(0, max_row_shift):
#            print ("row : {}  col :{}:{}".format(row, col, col+patch_size))
#            print ("col_shift : {}  row_shift :{}".format(col_shift, row_shift))
#            print ('row: {}  col:{}'.format(row,col))
            if (col+patch_size > H):
                break
            
            slice_3d =   lf[:,row:row + patch_size,col:col + patch_size ,:]    
#            print ('slice_3d_shape : {}'.format(np.shape(slice_3d)))
            patch_lf = lf_from_slice_new(slice_3d)
#            print ('patch_lf_shape : {}'.format(np.shape(patch_lf)))
            lf_dataset.append(patch_lf)
            col = col + strides

        row = row + strides
        col = 0
        
    return np.array(lf_dataset)



def gradient_tvars(self,g):
    for var in self.tvars:
        if not ("batch_normalization" in var.name):
#            print ("not included summary : **{}**{}".format(self.name,var.name))
            gradient = tf.gradients(self.loss, var, name = var.name[0:-2].split("/")[-1])
            summary = tf.summary.histogram(self.name + "_gradient_"+ var.name[0:-2], gradient)
            self.summary_list.append(summary)
        else:
            summary = tf.summary.histogram(self.name+"_batchnorm_" + var.name[0:-2],var)
            self.summary_list.append(summary)
        
#        elif(self.name == "Train"):
#            self.summary_list.append(var)
#        elif(self.name == "Test"):
#            self.summary_list.append(var)
#    
#def get_summary(m_train, m_test, g):
#    
#    gradients_histogram(m_train,g)
#    gradients_histogram(m_test,g)
#        
#    merge_train_summ = tf.summary.merge([m_train.training_loss_summary, 
#                                         m_train.output_stack_summary,
#                                         m_train.summary_batchnorm_conv1_b,
#                                         m_train.summary_batchnorm_conv2_b,
#                                         m_train.summary_lstm_input,
#                                         m_train.summary_lstm_output,
#                                         m_train.summary_decoder_input,
#                                         m_train.summary_out_bicubic,
#                                         m_train.summary_dec_batchnorm_conv1_b ,
#                                         m_train.output_layer_summary,
#                                         m_train.summary_gradient_Model_encoder_conv1_filter,
#                                         m_train.summary_gradient_Model_encoder_conv2_filter,
#                                         m_train.summary_gradient_Model_LSTM_lstm_cell_kernel,
#                                         m_train.summary_gradient_Model_decoder_dec_conv1_dec_conv1_filter,
#                                         m_train.summary_gradient_Model_out_level_out_conv_filter_yo,
#                                         m_train.summary_gradient_Model_encoder_conv1_bias_conv1,
#                                         m_train.summary_gradient_Model_encoder_conv2_bias_conv2,
#                                         m_train.summary_gradient_Model_LSTM_lstm_cell_bias,
#                                         m_train.summary_gradient_Model_decoder_dec_conv1_dec_bias_conv1,
#                                         m_train.summary_gradient_Model_out_level_out_conv_bias_yo
#    ])
#    
#    merge_test_summ = tf.summary.merge([m_test.training_loss_summary, 
#                                         m_test.output_stack_summary,
#                                         m_test.summary_batchnorm_conv1_b,
#                                         m_test.summary_batchnorm_conv2_b,
#                                         m_test.summary_lstm_input,
#                                         m_test.summary_lstm_output,
#                                         m_test.summary_decoder_input,
#                                         m_test.summary_out_bicubic,
#                                         m_test.summary_dec_batchnorm_conv1_b ,
#                                         m_test.output_layer_summary,
#                                         m_test.summary_gradient_Model_encoder_conv1_filter,
#                                         m_test.summary_gradient_Model_encoder_conv2_filter,
#                                         m_test.summary_gradient_Model_LSTM_lstm_cell_kernel,
#                                         m_test.summary_gradient_Model_decoder_dec_conv1_dec_conv1_filter,
#                                         m_test.summary_gradient_Model_out_level_out_conv_filter_yo, 
#                                         m_test.summary_gradient_Model_encoder_conv1_bias_conv1,
#                                         m_test.summary_gradient_Model_encoder_conv2_bias_conv2,
#                                         m_test.summary_gradient_Model_LSTM_lstm_cell_bias,
#                                         m_test.summary_gradient_Model_decoder_dec_conv1_dec_bias_conv1,
#                                         m_test.summary_gradient_Model_out_level_out_conv_bias_yo
#    ])
#    
#    return merge_train_summ,merge_test_summ
#    
#
#def gradients_histogram(self,g):
#    
##    self.training_loss_summary = tf.summary.scalar(self.name + "_loss_summary", self.loss)
##    self.output_stack_summary = tf.summary.image(self.name+"_output", tf.squeeze(self.output_stack),max_outputs = 10)
#        
##    self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
##    self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
##    self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
##    self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
##    self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input)            
##    self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)            
##    self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', self.dec_batchnorm_conv1_b)
##    self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output', self.output)       
#        
#    Model_encoder_conv1_filter = g.get_tensor_by_name("Model/encoder/conv1/conv1_filter:0")
#    gradient_Model_encoder_conv1_filter = tf.gradients(self.loss, Model_encoder_conv1_filter, 
#                                                             name = 'gradient_Model_encoder_conv1_filter_yo')
#    self.summary_gradient_Model_encoder_conv1_filter = tf.summary.histogram(self.name + "_" + 'summary_gradient_Model_encoder_conv1_filter', 
#                                                                                  gradient_Model_encoder_conv1_filter)  
#    
#    Model_encoder_conv2_filter = g.get_tensor_by_name("Model/encoder/conv2/conv2_filter:0")
#    gradient_Model_encoder_conv2_filter = tf.gradients(self.loss, Model_encoder_conv2_filter, 
#                                                             name = 'gradient_Model_encoder_conv2_filter_yo')
#    self.summary_gradient_Model_encoder_conv2_filter = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_encoder_conv2_filter', 
#                                                                                  gradient_Model_encoder_conv2_filter)  
#    
#    Model_LSTM_lstm_cell_kernel = g.get_tensor_by_name("Model/LSTM/lstm_cell/kernel:0")
#    gradient_Model_LSTM_lstm_cell_kernel = tf.gradients(self.loss, Model_LSTM_lstm_cell_kernel, 
#                                                             name = 'Model_LSTM_lstm_cell_kernel')
#    self.summary_gradient_Model_LSTM_lstm_cell_kernel = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_LSTM_lstm_cell_kernel', 
#                                                                                  gradient_Model_LSTM_lstm_cell_kernel)  
#        
#    Model_decoder_dec_conv1_dec_conv1_filter = g.get_tensor_by_name("Model/decoder/dec_conv1_/dec_conv1_filter:0")
#    gradient_Model_decoder_dec_conv1_dec_conv1_filter = tf.gradients(self.loss, Model_decoder_dec_conv1_dec_conv1_filter, 
#                                                             name = 'Model_decoder_dec_conv1_dec_conv1_filter')
#    self.summary_gradient_Model_decoder_dec_conv1_dec_conv1_filter = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_decoder_dec_conv1_dec_conv1_filter', 
#                                                                                  gradient_Model_decoder_dec_conv1_dec_conv1_filter)  
#    
#    Model_out_level_out_conv_filter_yo = g.get_tensor_by_name("Model/out_level/out_conv_filter_yo:0")
#    gradient_Model_out_level_out_conv_filter_yo = tf.gradients(self.loss, Model_out_level_out_conv_filter_yo, 
#                                                             name = 'Model_out_level_out_conv_filter_yo')
#    self.summary_gradient_Model_out_level_out_conv_filter_yo = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_out_level_out_conv_filter_yo', 
#                                                                                  gradient_Model_out_level_out_conv_filter_yo)  
#    
#    
#    Model_encoder_conv1_bias_conv1 = g.get_tensor_by_name("Model/encoder/conv1/bias_conv1:0")
#    gradient_Model_encoder_conv1_bias_conv1 = tf.gradients(self.loss, Model_encoder_conv1_bias_conv1, 
#                                                             name = 'Model_encoder_conv1_bias_conv1')
#    self.summary_gradient_Model_encoder_conv1_bias_conv1 = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_encoder_conv1_bias_conv1', 
#                                                                                  gradient_Model_encoder_conv1_bias_conv1)  
#    
#    Model_encoder_conv2_bias_conv2 = g.get_tensor_by_name("Model/encoder/conv2/bias_conv2:0")
#    gradient_Model_encoder_conv2_bias_conv2 = tf.gradients(self.loss, Model_encoder_conv2_bias_conv2, 
#                                                             name = 'Model_encoder_conv2_bias_conv2')
#    self.summary_gradient_Model_encoder_conv2_bias_conv2 = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_encoder_conv2_bias_conv2', 
#                                                                                  gradient_Model_encoder_conv2_bias_conv2)  
#    
#    
#    Model_LSTM_lstm_cell_bias = g.get_tensor_by_name("Model/LSTM/lstm_cell/bias:0")
#    gradient_Model_LSTM_lstm_cell_bias = tf.gradients(self.loss, Model_LSTM_lstm_cell_bias, 
#                                                             name = 'Model_LSTM_lstm_cell_bias')
#    self.summary_gradient_Model_LSTM_lstm_cell_bias = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_LSTM_lstm_cell_bias', 
#                                                                                  gradient_Model_LSTM_lstm_cell_bias)
#
#    Model_decoder_dec_conv1_dec_bias_conv1 = g.get_tensor_by_name("Model/decoder/dec_conv1_/dec_bias_conv1:0")
#    gradient_Model_decoder_dec_conv1_dec_bias_conv1 = tf.gradients(self.loss, Model_decoder_dec_conv1_dec_bias_conv1, 
#                                                             name = 'Model_decoder_dec_conv1_dec_bias_conv1')
#    self.summary_gradient_Model_decoder_dec_conv1_dec_bias_conv1 = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_decoder_dec_conv1_dec_bias_conv1', 
#                                                                                  gradient_Model_decoder_dec_conv1_dec_bias_conv1)
#
#    Model_out_level_out_conv_bias_yo = g.get_tensor_by_name("Model/out_level/out_conv_bias_yo:0")
#    gradient_Model_out_level_out_conv_bias_yo = tf.gradients(self.loss, Model_out_level_out_conv_bias_yo, 
#                                                             name = 'Model_out_level_out_conv_bias_yo')
#    self.summary_gradient_Model_out_level_out_conv_bias_yo = tf.summary.histogram(self.name + "_" +'summary_gradient_Model_out_level_out_conv_bias_yo', 
#                                                                                  gradient_Model_out_level_out_conv_bias_yo)


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

