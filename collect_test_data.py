#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:57:04 2019

@author: rohit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:23:08 2019

@author: rohit
"""

import numpy as np

dot =""

data = np.load(dot + "/saver-500-3/processed_test_data_128/processed_test_data_128.npy")
print ("data shape : {}".format(data.shape))
print ("done")


length = data.shape[0]
print ("total_length : {}".format(length))
rand = np.arange(length)
np.random.shuffle(rand)

print (rand[1:10])


train_all_s = data[rand,:,:,:]

print ("train_all_s shape : {}".format(train_all_s.shape))

save_path = dot + "/saver-500-3/"
size = length//3

for i in range(3):
    data = train_all_s[i*size:(i+1)*size,:,:,:]
    print ("index : {} : {}".format(i*size,(i+1)*size))
    print (data.shape)
    np.save(save_path + str(i+1) + ".npy", data)
    print ("saved as {}".format(save_path + str(i+1) + ".npy"))




