#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:39:10 2019

@author: rohit
"""

import numpy as np
from pathlib import Path
import math
import os

import generate_lf_dataset

patch_size = 32
strides = 8

base_dir = "/Users/rohit/Downloads/Random/benchmark/training"
all_folders = generate_lf_dataset.get_all_data_folders(base_dir)
lf_dataset_new = []

for data_folder in all_folders:
    print (data_folder)
    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])
    print (views)
    lf = generate_lf_dataset.lf_from_Ychannel_only_new(data_folder, views)
    print ('lf : {}'.format(np.shape(lf)))
    lf_dataset = generate_lf_dataset.lf_from_patches_new(lf, patch_size, strides)
    print ('lf_dataset : {}'.format(np.shape(lf_dataset)))
    lf_dataset_new.append(lf_dataset)
    
lf_dataset_new = np.array(lf_dataset_new)
lf_dataset_new = np.reshape(lf_dataset_new, (-1,288,288,1))

lf_dim = int(math.sqrt(lf.shape[0]))
lf_dataset_new = ((lf_dataset_new - np.mean(lf_dataset_new))/(np.std(lf_dataset_new)))

