#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:45:48 2019

@author: rohit
"""

import numpy as np

train = np.load("/saver-1000/1.npy")
print ("train : {}".format(train.shape))
train_local = train[1:1000,:,:,:]
print ("train_local : {}".format(train_local.shape))


test = np.load("/saver-500-3/1.npy")
print ("test : {}".format(test.shape))
test_local = test[1:1000,:,:,:]
print ("test_local : {}".format(test_local.shape))


np.save("/saver-1000/train_local_128.npy",train_local)
np.save("/saver-1000/test_local_128.npy",test_local)

