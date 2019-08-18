#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:50:28 2019

@author: rohit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:32:44 2019

@author: rohit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:58:18 2019

@author: rohit
"""

import numpy as np
from pathlib import Path
import math
import tensorflow as tf 
import random
import os
import datetime


import generate_lf_dataset

strides = 8
patch_size = 32


hidden_size = 1024

is_training = True
batch_count = 0

train_percent = 0.80 
lf_size = 0

starter_learning_rate = 0.1
decay_steps = 100
decay_rate = 0.96

epochs = 30


batch_size  = test_batch_size = 100
dot = ""
#base_dir = "/Users/rohit/Downloads/Random/benchmark/training"
base_dir = dot + "/tmp/data/training/"
#path = dot + '/tmp/data/dino'
#path  = '/Users/rohit/Documents/dino'


try:
    lr_list = np.load(dot + '/saver/lr.npy')
    lr_list = lr_list.tolist()
except:
    lr_list = []

try:
    loss_list = np.load(dot +'/saver/loss.npy')
    loss_list = loss_list.tolist()
except:
    loss_list = []


output_list = []
index_dict = {}

all_folders = generate_lf_dataset.get_all_data_folders(base_dir)
lf_dataset_new = []

for data_folder in all_folders:
    print (data_folder)
    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])
#    print (views)
    lf = generate_lf_dataset.lf_from_Ychannel_only_new(data_folder, views)
    print ('lf : {}'.format(np.shape(lf)))
    lf_dataset = generate_lf_dataset.lf_from_patches_new(lf, patch_size, strides)
    print ('lf_dataset : {}'.format(np.shape(lf_dataset)))
    lf_dataset_new.append(lf_dataset)
    
    
lf_dim = int(math.sqrt(lf.shape[0]))

lf_dataset_new = np.array(lf_dataset_new)
lf_dataset_new = np.reshape(lf_dataset_new, (-1,lf_dim*patch_size,lf_dim*patch_size,1))
print ('lf_dataset_new : {}'.format(lf_dataset_new.shape))



print ('calculating mean....')
print (datetime.datetime.now())
mean = np.mean(lf_dataset_new)
print ('calculating std....')
print (datetime.datetime.now())
std = np.std(lf_dataset_new,dtype=np.float64)
print ('calculating....')
print (datetime.datetime.now())
lf_dataset_new = (lf_dataset_new - mean)/(std)
print ('done!')


def variable_get(shape, name):
    return tf.get_variable(name, shape, tf.float32)
        

def conv2D_batchnorm(input_, kernel_shape, kernel_name,strides, padding,bias_name,batch_norm_name, is_training):
    conv = tf.nn.relu(tf.nn.conv2d(input_, variable_get(kernel_shape, kernel_name),
                                                                            strides = strides, padding = padding,data_format='NHWC'))          
    conv_b = tf.nn.bias_add(conv, variable_get([kernel_shape[-1]],name = bias_name))
    
    
    batchnorm_conv_b = tf.keras.layers.BatchNormalization(axis = -1)(conv_b, training = is_training)
    
    
#    batchnorm_conv_b = tf.layers.batch_normalization(
#                            conv_b, training, name = batch_norm_name)
#    batchnorm_conv_b = tf.contrib.layers.batch_norm(conv_b, is_training = training)
    
    return batchnorm_conv_b
  


def layers(row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            batchnorm_conv1_b = conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3,16], 
                                                         kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv1",
                                                         batch_norm_name = "batch_norm_conv1",
                                                         is_training = is_training)

                   
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            batchnorm_conv2_b = conv2D_batchnorm(input_ = batchnorm_conv1_b, kernel_shape = [3,3,16,32], 
                                                         kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv2",
                                                         batch_norm_name = "batch_norm_conv2",
                                                         is_training = is_training)

                    
        with tf.variable_scope('conv_3',reuse=tf.AUTO_REUSE):
            batchnorm_conv3_b = conv2D_batchnorm(input_ = batchnorm_conv2_b, kernel_shape = [3,3,32,64], 
                                                         kernel_name = "conv3_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv3",
                                                         batch_norm_name = "batch_norm_conv3",
                                                         is_training = is_training)

        lstm_input = tf.reshape(batchnorm_conv3_b, [-1,4*4*64])
        lstm_input = tf.identity(lstm_input, name="lstm_input")
                
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(lstm_input, state)
            
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        decoder_input = tf.reshape(cell_output,[-1,4,4,64])
        decoder_input = tf.identity(decoder_input,name = "decoder_input")
                
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            out_bicubic = tf.image.resize_bicubic(decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                    
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv1_b = conv2D_batchnorm(input_ = out_bicubic, kernel_shape = [3,3,64,32], 
                                                         kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv1",
                                                         batch_norm_name = "dec_batch_norm_conv1",
                                                         is_training = is_training)
                    
        with tf.variable_scope('dec_conv2_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv2_b = conv2D_batchnorm(input_ = dec_batchnorm_conv1_b, kernel_shape = [3,3,32,16], 
                                                         kernel_name = "dec_conv2_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv2",
                                                         batch_norm_name = "dec_batch_norm_conv2",
                                                         is_training = is_training)
                
        with tf.variable_scope('dec_conv3_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv3_b = conv2D_batchnorm(input_ = dec_batchnorm_conv2_b, kernel_shape = [3,3,16,3], 
                                                         kernel_name = "dec_conv3_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv3",
                                                         batch_norm_name = "dec_batch_norm_conv3",
                                                         is_training = is_training)

        with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
            out_conv = tf.nn.relu(tf.nn.conv2d(
                            dec_batchnorm_conv3_b, variable_get([3,3,3,1], 'out_conv_filter'),strides = [1,1,1,1], 
                            padding = 'SAME'))
            output = tf.nn.bias_add(out_conv, variable_get([1],'out_conv_bias'))         
            
            return output, state

def get_keys(row, col):
    key_left = (str(row)+str(col-1)) 
    key_left_top = (str(row-1)+str(col-1))
    key_top = (str(row-1)+str(col))
    key_target = (str(row)+str(col))
    
    return key_left,key_left_top,key_top,key_target


def get_slices(lf_batch_holder, row, col, key_left,key_left_top,key_top,key_target):
    if ((row == 1) or (col == 1)):
        if((row == 1) and (col == 1)):
            left = lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size, 
                                        (col*patch_size)-patch_size:(col*patch_size),:]
            left_top = lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                            (col*patch_size)-patch_size:(col*patch_size),:]
            top = lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                       (col*patch_size):(col*patch_size)+patch_size,:]
                    
        elif((row != 1) and (col == 1)):
            left = lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size, 
                                        (col*patch_size)-patch_size:(col*patch_size),:]
            left_top = lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                            (col*patch_size)-patch_size:(col*patch_size),:]        
            top = output_list[index_dict[key_top]]

        elif((row == 1) and (col != 1)):
            left = output_list[index_dict[key_left]]
            left_top = lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                            (col*patch_size)-patch_size:(col*patch_size),:]        
            top = lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                       (col*patch_size):(col*patch_size)+patch_size,:]

                    
    else:
        left = output_list[index_dict[key_left]]
        left_top = output_list[index_dict[key_left_top]]
        top = output_list[index_dict[key_top]]
    
    return left, left_top, top

def prepare_input(row, col, lf_batch_holder, is_training):
    
    key_left,key_left_top,key_top,key_target = get_keys(row, col)
    print ('{} , {}, {} <=> {}'.format(key_left,key_left_top,key_top,key_target))
            
    left, left_top, top = get_slices(lf_batch_holder, row, col, key_left,key_left_top,key_top,key_target)

    seq_input = []
    
#    tf_batch_size = tf.shape(lf_batch_holder)[0]
    
    for batch in range(batch_size):
            seq_input.append(left[batch,:,:,:])
            seq_input.append(left_top[batch,:,:,:])
            seq_input.append(top[batch,:,:,:])
              
    seq_input = tf.convert_to_tensor(seq_input)
    seq_input = tf.reshape(seq_input, (-1,32,32,3))
    return seq_input, key_target
    
def model(copy_lf_batch_holder, is_training):
    
    loss = 0
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                    reuse = tf.AUTO_REUSE)

    
#    if (is_training):
    initial_state = cell.zero_state(batch_size, tf.float32)    
#    else :
#        initial_state = cell.zero_state(test_batch_size, tf.float32)    
        
    tf_resize_tensor = tf.constant([patch_size, patch_size], name= 'interpolate_to_size')

    
    for col in range(1,lf_dim):
            
        state = initial_state
        
        for row in range(1,lf_dim):
                        
            seq_input, key_target = prepare_input(row, col, copy_lf_batch_holder, is_training)
#            t_list.append(t)
            
            output, state_ = layers(row, col, seq_input, cell, state, tf_resize_tensor, is_training)
            state = state_
            
            output = tf.identity(output, name="output_from_layers")
            
            output_list.append(output)
            index = len(output_list) - 1
            index_dict[key_target] = index
            
            target = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size,
                                 col*patch_size:(col*patch_size)+patch_size,:]
            
            
            target = tf.identity(target, name="target_from_batchholder")
#            loss = loss + tf.nn.l2_loss(target-output, name = 'l2_loss')                     
            loss = loss + tf.reduce_sum(tf.pow((target-output),2))
#            break
            
        
    
    output_stack = tf.stack(output_list)
#    t_stack = tf.stack(t_list)
    
    return loss, output_stack, index_dict
 


def make_graph(g):
    with g.as_default():
        #with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
        copy_lf_batch_holder = tf.placeholder(tf.float32, shape = (None, lf_dim*patch_size,lf_dim*patch_size,1), name = 'lf_batch_data_holder')
    
        loss, output_stack, index_dict = model(copy_lf_batch_holder, is_training)
    
    
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars, name = 'train_gradients'),
                                       5, name = "clip_gradients_train")
    
        global_step=tf.train.get_or_create_global_step()
            
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=True)
            
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name = "gradient_descent_train")
    
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step = global_step, name = "apply_gradients_train")

        g.add_to_collection("loss_",loss)
        g.add_to_collection("output_stack",output_stack)
    
        print ('exporting_graph...')    
#        tf.train.export_meta_graph(dot +'/saver/models.meta')
        print ('graph_exported')
            
        return loss, train_op, output_stack, copy_lf_batch_holder, learning_rate
    
    

data_size = lf_dataset_new.shape[0]
train_size = int(train_percent * data_size)


#lf_dataset = np.reshape(lf_dataset, (-1,lf_dim*patch_size,lf_dim*patch_size,1))
lf_train_data = (lf_dataset_new[0:train_size,:,:,:])
lf_test_data =  (lf_dataset_new[train_size:data_size,:,:,:])

print ('lf_train_data', np.shape(lf_train_data))

random_example_index = random.randint(0,data_size)
########################
random_example = lf_dataset_new[1:3,:,:,:]
#random_example = np.reshape(random_example,(-1,lf_dim*patch_size,lf_dim*patch_size,1))
output_random_example = random_example

output_example_all = []

g = tf.Graph()

with g.as_default():
    
    tf_lf_train_placeholder = tf.placeholder(tf.float32, shape = (None, lf_dim*patch_size,lf_dim*patch_size,1), name = 'tf_lf_train_placeholder')
    
    print ("-----------Training")
    tf_lf_train_data = tf.data.Dataset.from_tensor_slices(tf_lf_train_placeholder)
    
    tf_lf_train_data = tf_lf_train_data.repeat(epochs).shuffle(buffer_size = 10000).batch(batch_size)
    batch_iter = tf_lf_train_data.make_initializable_iterator()
    next_batch = batch_iter.get_next()
    
    print ("-----------Test")
    tf_lf_test_data = tf.data.Dataset.from_tensor_slices(lf_test_data)
    
    tf_lf_test_data = tf_lf_test_data.repeat().shuffle(buffer_size = 10000).batch(test_batch_size)
    test_batch_iter = tf_lf_test_data.make_initializable_iterator()
    test_next_batch = test_batch_iter.get_next()
    
    


loss, train_op, output_stack, copy_lf_batch_holder,learning_rate = make_graph(g)
 
print ('***********ready to graph **************')

with g.as_default():
        
    saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir = dot +'/saver/checkpoint_saver', save_steps = 30)
    summary_hook = tf.train.SummarySaverHook(output_dir = dot +'/saver/summary',save_steps =2,scaffold=tf.train.Scaffold())
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    print ('**************** entering in session ****************')
    
    with tf.train.MonitoredTrainingSession(config = config,
                                                     hooks = [saver_hook],
                                                     checkpoint_dir = dot +'/saver/checkpoint_saver') as sess:
        print ('**************** inside the session ****************')
        sess.run(batch_iter.initializer, feed_dict={tf_lf_train_placeholder : lf_train_data})
        sess.run(test_batch_iter.initializer)
        
        while not sess.should_stop():
            g_step = g.get_tensor_by_name('global_step:0')
            print ('global_step : {}'.format(sess.run(g_step)))
        
            train_batch = sess.run(next_batch)
            test_batch = sess.run(test_next_batch)
            
            print ('train and test batch generated')
        
#            fetches = {'train_op': train_op,
#                       'loss': loss}
#        
            lr = sess.run(learning_rate)
            print ('learning_rate :{}'.format(lr))
            
            lr_list.append(lr)
        
    
            if(train_batch.shape[0] != batch_size):
                break
            
            output_list = []
            index_dict = {}
            
            is_training = True
            sess.run(train_op,feed_dict = {copy_lf_batch_holder : train_batch})

            print ('training_finished')

            output_list = []
            index_dict = {}
                        
            is_training = False
            
            loss_ = sess.run(loss, feed_dict = {copy_lf_batch_holder : test_batch})
          
#            output_random_stack = sess.run(output_stack, feed_dict = {copy_lf_batch_holder : random_example})
              
#            for row in range(1, lf_dim):
#                for col in range(1,lf_dim):
#                    key = (str(row)+str(col))
#                    index = index_dict[key]
#                    output_slice = output_random_stack[index,:,:,:,:]
                    
#                    output_random_example[:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size,:] = output_slice
                    
#            output_example_all.append(output_random_example)
#            np.save('output_example_all.npy', np.array(output_example_all))             
            
            print ('batch_count : {}'.format(batch_count))
            batch_count = batch_count + 1

            print ('loss : {}'.format(loss_))
            loss_list.append(loss_)
            
            np.save(dot +'/saver/loss.npy', np.array(loss_list))   
            
            np.save(dot + '/saver/lr.npy', np.array(lr_list))            

#            if (batch_count > 10):
#                break

    print ('************ phir - milenge ********')
#np.save('loss.npy', np.array(loss_list))    
#np.save('output_example_all.npy', np.array(output_example_all))     
  
