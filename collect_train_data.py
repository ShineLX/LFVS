#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:23:08 2019

@author: rohit
"""

import numpy as np

dot =""

#train1_3 = np.load(dot + "/saver-500-2/processed_train_data_128/processed_train_data_128.npy")
#print ("1-3 shape : {}".format(train1_3.shape))
#print ("done")
#
#train10_12 = np.load(dot +"/saver-500-2/processed_train_data_128_10-12/processed_train_data_10-12128.npy")
#print ("10-12 shape : {}".format(train10_12.shape))
#print ("done")
#
#train_all = np.append(train1_3,train10_12,axis = 0)
#print ("all shape : {}".format(train_all.shape))
#
#train1_3 = np.zeros((2,2))
#train10_12 = np.zeros((2,2))

#
train_all = np.load(dot + "/saver-500-4/processed_train_data_128_4-9/processed_train_data_4-9128.npy") 
print ("4-9 shape : {}".format(train_all.shape))
print ("done")
#
#
#train_all = np.append(train_all,train4_9,axis = 0)
#print ("all shape : {}".format(train_all.shape))
#
#train4_9 = np.zeros((2,2))



save_path = dot + "/saver-1000/" 

length = train_all.shape[0]
print ("total_length : {}".format(length))
rand = np.arange(length)
np.random.shuffle(rand)

print (rand[1:10])


train_all_s = train_all[rand,:,:,:]
train_all = []

print ("train_all_s shape : {}".format(train_all_s.shape))


division = 3
size = length//division

for i in range(division):
    data = train_all_s[i*size:(i+1)*size,:,:,:]
    print ("index : {} : {}".format(i*size,(i+1)*size))
    print (data.shape)
    np.save(save_path + str(i+3+1) + ".npy", data)
    print ("saved as {}".format(save_path + str(i+3+1) + ".npy"))




