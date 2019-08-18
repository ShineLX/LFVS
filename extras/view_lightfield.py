#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:55:46 2019

@author: rohit
"""

import file_io
data_folder = "/Users/rohit/Downloads/Random/benchmark/training/dino"

LF = file_io.read_lightfield(data_folder)
param_dict = file_io.read_parameters(data_folder)
depth_map = file_io.read_depth(data_folder, highres=True)
disparity_map = file_io.read_disparity(data_folder, highres=False)