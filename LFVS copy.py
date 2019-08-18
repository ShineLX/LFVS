#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:58:18 2019

@author: rohit
"""

import numpy as np
from pathlib import Path
import cv2
import math
import tensorflow as tf 


def lf_from_slice(slice_):
    h = slice_.shape[0]
    w = slice_.shape[1]
    d = slice_.shape[2]
    
    n_view = int(math.sqrt(d))
    
    col_lf = np.array([], dtype=np.uint8).reshape(0,w*n_view)
    
    for j in range(n_view):
        row_lf = np.array([], dtype=np.uint8).reshape(h,0)
        for i in range(n_view):
#            print ('index : {}'.format((n_view*j)+i))
            row_lf = np.hstack((row_lf, slice_[:,:,(n_view*j)+i]))
#            cv2.imshow('row_only', slice_[:,:,(n_view*j)+i])
#            cv2.imshow('row_lf', row_lf)
#            cv2.waitKey(-1)

            
        col_lf = np.vstack((col_lf, row_lf))
#        cv2.imshow('col_lf', col_lf)
#        cv2.waitKey(-1)        
        
    return col_lf
    
    
#/Users/rohit/Downloads/Random/benchmark/training

path = './dino'

pathlist = Path(path).glob('*input_Cam*')

string_paths = []

for path in pathlist:
    path_in_str = str(path)
#    print (path_in_str)
    string_paths.append(path_in_str)
   
string_paths.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

lf = []

view_count = 0
error_count = 0
for path in string_paths:
    
    view_count = view_count + 1
    try:    
        img = cv2.imread(path)
        img_Y = np.array((cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))[:,:,0])
#    cv2.putText(img_Y,'{}'.format(view_count),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
#    cv2.imshow('img', img_Y)
        lf.append(img_Y)
    except:
        error_count = error_count + 1
        
#    cv2.waitKey(-1)
print ('error_count : {}'.format(error_count))

#lf = np.array(lf)
lf = np.moveaxis(lf, 0, -1)

strides = 8
patch_size = 32

H = np.shape(lf)[0]
W = np.shape(lf)[1]

max_row_shift = max_col_shift = ((H - patch_size)//strides) + 1

lf_dataset = []

lf_size = 0
row = col = 0
for col_shift in range(0,max_col_shift):
    for row_shift in range(0, max_row_shift):
        lf_size = lf_size + 1
        #print ('lf_size: {}'.format(lf_size))
        slice_3d =   lf[row: row + patch_size,col: col + patch_size ,:]    
        patch_lf = lf_from_slice(slice_3d)
        lf_dataset.append(patch_lf)
        col = col + strides
    row = row + 8
    col = 0
        
lf_dataset = np.float32(np.array(lf_dataset))

lf_dataset = ((lf_dataset - np.mean(lf_dataset))/(np.std(lf_dataset)))

lf_dim = int(math.sqrt(lf.shape[2]))

batch_size = 100

g = tf.Graph()

with g.as_default():
    
    
    def variable_get(shape, name):
            return tf.get_variable(name, shape, tf.float32)
        
        
    def conv2D_batchnorm(input_, kernel_shape, kernel_name,strides, padding,bias_name,batch_norm_name, training):
        conv1 = tf.nn.relu(tf.nn.conv2d(input_, variable_get(kernel_shape, kernel_name),
                                                                            strides = strides, padding = padding))          
        conv1_b = tf.nn.bias_add(conv1, variable_get([kernel_shape[-1]],name = bias_name))
#        batchnorm_conv1_b = tf.layers.batch_normalization(
#                            conv1_b, training, name = batch_norm_name)
        return conv1_b
    
    def deConv2D(input_,kernel_shape,kernel_name, output_shape,strides,padding, op_name,bias_name, training,batch_norm_name):
        dec_conv1 = tf.nn.relu(tf.nn.conv2d_transpose(
                            input_, variable_get(kernel_shape, kernel_name),
                            output_shape = output_shape,
                            strides = strides, padding = padding, name = op_name))          
        dec_conv1_b = tf.nn.bias_add(dec_conv1, variable_get([output_shape[-1]],bias_name))
#        dec_batchnorm_conv1_b = tf.layers.batch_normalization(
#                            dec_conv1_b, training = training, name = batch_norm_name)
        return dec_conv1_b
    
    dict_index_to_output_tensor = {}
    
    lf_batch_holder = tf.placeholder(tf.float32, shape = (None, 288,288), name = 'lf_batch_data_holder')
    copy_lf_batch_holder =   tf.identity(lf_batch_holder, name="copy_lf_batch_data_holder")
    
    h_copy_lf_batch_holder = int(copy_lf_batch_holder.shape[1])
    w_copy_lf_batch_holder = int(copy_lf_batch_holder.shape[2])
    
    copy_lf_batch_holder = tf.reshape(copy_lf_batch_holder, (-1,h_copy_lf_batch_holder,w_copy_lf_batch_holder,1))
    
    hidden_size = 1024
    
#    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
#                                                    reuse = tf.AUTO_REUSE)
    
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                    reuse = tf.AUTO_REUSE)


    
    initial_state = cell.zero_state(batch_size, tf.float32)

    loss = 0
    
    output_list = []
    index_dict = {}
    
    tf_resize_tensor = tf.constant([patch_size, patch_size], name= 'interpolate_to_size')
        

#    row = 0
#    
#    for col in range(lf_dim):
#        key = int(str(row) + str(col))
##        print (key)
#        dict_index_to_output_tensor[key] = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size,
#                                 col*patch_size:(col*patch_size)+patch_size,:]
#    
#    col = 0
#    
#    for row in range(1,lf_dim):
#        key = int(str(row) + str(col))
##        print (key)
#        dict_index_to_output_tensor[key] = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size,
#                                 col*patch_size:(col*patch_size)+patch_size,:]
#
 
    
    for col in range(1,lf_dim):
        
        state = initial_state
        
        for row in range(1,lf_dim):
            
            
                        
            key_left = (str(row)+str(col-1)) 
            key_left_top = (str(row-1)+str(col-1))
            key_top = (str(row-1)+str(col))
            key_target = (str(row)+str(col))
            
            print ('{} , {}, {} <=> {}'.format(key_left,key_left_top,key_top,key_target))
            
            if ((row == 1) or (col == 1)):
                if((row == 1) and (col == 1)):
                    left = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size, 
                                        (col*patch_size)-patch_size:(col*patch_size),:]
                    left_top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                            (col*patch_size)-patch_size:(col*patch_size),:]
                    top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                       (col*patch_size):(col*patch_size)+patch_size,:]
                    
                elif((row != 1) and (col == 1)):
                    left = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size, 
                                        (col*patch_size)-patch_size:(col*patch_size),:]
                    left_top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                            (col*patch_size)-patch_size:(col*patch_size),:]        
                    top = output_list[index_dict[key_top]]

                elif((row == 1) and (col != 1)):
                    left = output_list[index_dict[key_left]]
                    left_top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                            (col*patch_size)-patch_size:(col*patch_size),:]        
                    top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
                                       (col*patch_size):(col*patch_size)+patch_size,:]

                    
            else:
                left = output_list[index_dict[key_left]]
                left_top = output_list[index_dict[key_left_top]]
                top = output_list[index_dict[key_top]]
                                       
                    
                    
#            left_x_top =row*patch_size
#            left_x_bottom = (row*patch_size)+patch_size
#            left_y_top = (col*patch_size)-patch_size
#            left_y_bottom = (col*patch_size)
#            


#            
#        
#            
#            left = dict_index_to_output_tensor[key_left]
#            left_top = dict_index_to_output_tensor[key_left_top]
#            top = dict_index_to_output_tensor[key_top]
#            
#            
            
#            left = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size, 
#                                        (col*patch_size)-patch_size:(col*patch_size),:]
            
#            left_top_x_top = (row*patch_size)-patch_size
#            left_top_x_bottom = (row*patch_size)
#            left_top_y_top = (col*patch_size)-patch_size
#            left_top_y_bottom = col*patch_size
#            
#            left_top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
#                                            (col*patch_size)-patch_size:(col*patch_size),:]
    
#            top_x_top = (row*patch_size)-patch_size
#            top_x_bottom = (row*patch_size)
#            top_y_top = (col*patch_size)
#            top_y_bottom = (col*patch_size)+patch_size
#    
#            top = copy_lf_batch_holder[:,(row*patch_size)-patch_size:(row*patch_size), 
#                                       (col*patch_size):(col*patch_size)+patch_size,:]
              
#            
#            if(row == 1):
#                print ('\nleft = ({},{}:{},{})'.format(left_x_top,left_y_top,left_x_bottom,left_y_bottom))
#            
#                print ('left_top = ({},{}:{},{})'.format(left_top_x_top,left_top_y_top,left_top_x_bottom,left_top_y_bottom))
#            
#                print ('top = ({},{}:{},{})'.format(top_x_top,top_y_top,top_x_bottom,top_y_bottom))
#            
#      
            
            seq_input = []
            for batch in range(batch_size):
                seq_input.append(left[batch,:,:,:])
                seq_input.append(left_top[batch,:,:,:])
                seq_input.append(top[batch,:,:,:])
                
                
#            seq_input = np.array(seq_input)    
            seq_input = tf.convert_to_tensor(seq_input)
            
            seq_input = tf.reshape(seq_input, (-1,32,32,3))
            
            with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
                with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
                    batchnorm_conv1_b = conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3,16], 
                                                         kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv1",
                                                         batch_norm_name = "batch_norm_conv1",
                                                         training = True)
                    
#                    conv1 = tf.nn.relu(tf.nn.conv2d(seq_input, variable_get([3,3,3,16], 'conv1_filter'),
#                                                                            strides = [1,2,2,1], padding = 'SAME'))          
#                    conv1_b = tf.nn.bias_add(conv1, variable_get([16],'bias_conv1'))
#                    batchnorm_conv1_b = tf.layers.batch_normalization(
#                            conv1_b, training=True, name = 'batch_norm_conv1')
#                    
                with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
                    batchnorm_conv2_b = conv2D_batchnorm(input_ = batchnorm_conv1_b, kernel_shape = [3,3,16,32], 
                                                         kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv2",
                                                         batch_norm_name = "batch_norm_conv2",
                                                         training = True)
#                    
#                    
#                    conv2 = tf.nn.relu(tf.nn.conv2d(
#                            batchnorm_conv1_b, variable_get([3,3,16,32], 'conv2_filter'),strides = [1,2,2,1], 
#                            padding = 'SAME'))
#                    conv2_b = tf.nn.bias_add(conv2, variable_get([32],'conv2_bias'))
#                    batchnorm_conv2_b = tf.layers.batch_normalization(
#                            conv2_b, training=True,name = 'batch_norm_conv2')
                    
                with tf.variable_scope('conv_3',reuse=tf.AUTO_REUSE):
                    batchnorm_conv3_b = conv2D_batchnorm(input_ = batchnorm_conv2_b, kernel_shape = [3,3,32,64], 
                                                         kernel_name = "conv3_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv3",
                                                         batch_norm_name = "batch_norm_conv3",
                                                         training = True)
##                    
#                    
#                    conv3 = tf.nn.relu(tf.nn.conv2d(
#                            batchnorm_conv2_b, variable_get([3,3,32,64], 'conv3_filter'),strides = [1,2,2,1], 
#                            padding = 'SAME'))
#                    conv3_b = tf.nn.bias_add(conv3, variable_get([64],'conv3_bias'))
#                    batchnorm_conv3_b = tf.layers.batch_normalization(
#                            conv3_b, training=True,name = 'batch_norm_conv3')
#            
                lstm_input = tf.reshape(batchnorm_conv3_b, [-1,4*4*64])
                lstm_input = tf.identity(lstm_input, name="lstm_input")
                
            with tf.variable_scope('LSTM'):
                if not (row == 1 and col == 1): 
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(lstm_input, state)
            
            with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
                lstm_output = tf.reshape(cell_output,[-1,4,4,64])
                lstm_output = tf.identity(lstm_output,name = "lstm_output")
                
                with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
                    out_bicubic = tf.image.resize_bicubic(lstm_output,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                    
                with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
                    dec_batchnorm_conv1_b = conv2D_batchnorm(input_ = out_bicubic, kernel_shape = [3,3,64,32], 
                                                         kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv1",
                                                         batch_norm_name = "dec_batch_norm_conv1",
                                                         training = True)
                    
                with tf.variable_scope('dec_conv2_',reuse=tf.AUTO_REUSE):
                    dec_batchnorm_conv2_b = conv2D_batchnorm(input_ = dec_batchnorm_conv1_b, kernel_shape = [3,3,32,16], 
                                                         kernel_name = "dec_conv2_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv2",
                                                         batch_norm_name = "dec_batch_norm_conv2",
                                                         training = True)
                
                with tf.variable_scope('dec_conv3_',reuse=tf.AUTO_REUSE):
                    dec_batchnorm_conv3_b = conv2D_batchnorm(input_ = dec_batchnorm_conv2_b, kernel_shape = [3,3,16,3], 
                                                         kernel_name = "dec_conv3_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv3",
                                                         batch_norm_name = "dec_batch_norm_conv3",
                                                         training = True)
                
#                with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
#                    dec_batchnorm_conv1_b = deConv2D(input_ = lstm_output , kernel_shape = [3,3,32,64],
#                                                     kernel_name = "dec_conv1_filter",output_shape =[batch_size,8,8,32],
#                                                     strides  = [1,2,2,1],padding = "SAME", op_name = "dec_conv1_op",
#                                                     bias_name = "dec_bias_conv1", 
#                                                     training = True,batch_norm_name = "dec_batch_norm_conv1")
##                    
#                    
#                    dec_conv1 = tf.nn.relu(tf.nn.conv2d_transpose(
#                            lstm_output, variable_get([3,3,32,64], 'dec_conv1_filter'),output_shape = [batch_size,8,8,32],
#                            strides = [1,2,2,1], padding = 'SAME', name = 'dec_conv1_op'))          
#                    dec_conv1_b = tf.nn.bias_add(dec_conv1, variable_get([32],'dec_bias_conv1'))
#                    dec_batchnorm_conv1_b = tf.layers.batch_normalization(
#                            dec_conv1_b, training=True, name = 'dec_batch_norm_conv1')
                    
#                with tf.variable_scope('dec_conv2',reuse=tf.AUTO_REUSE):
#                    dec_batchnorm_conv2_b = deConv2D(input_ = dec_batchnorm_conv1_b , kernel_shape = [3,3,16,32],
#                                                     kernel_name = "dec_conv2_filter",output_shape =[batch_size,16,16,16],
#                                                     strides  = [1,2,2,1],padding = "SAME", op_name = "dec_conv2_op",
#                                                     bias_name = "dec_bias_conv2", 
#                                                     training = True,batch_norm_name = "dec_batch_norm_conv2")
##                    
##                    
#                    dec_conv2 = tf.nn.relu(tf.nn.conv2d_transpose(
#                            dec_batchnorm_conv1_b, variable_get([3,3,16,32], 'dec_conv2_filter'),
#                            output_shape = [batch_size,16,16,16],strides = [1,2,2,1], padding = 'SAME', 
#                            name = 'dec_conv2_op'))          
#                    dec_conv2_b = tf.nn.bias_add(dec_conv2, variable_get([16],'dec_conv2_bias'))
#                    dec_batchnorm_conv2_b = tf.layers.batch_normalization(
#                            dec_conv2_b, training=True,name = 'dec_batch_norm_conv2')
#                    
#                with tf.variable_scope('dec_conv_3',reuse=tf.AUTO_REUSE):
#                    dec_batchnorm_conv3_b = deConv2D(input_ = dec_batchnorm_conv2_b , kernel_shape = [3,3,3,16],
#                                                     kernel_name = "dec_conv3_filter",output_shape =[batch_size,32,32,3],
#                                                     strides  = [1,2,2,1],padding = "SAME", op_name = "dec_conv3_op",
#                                                     bias_name = "dec_bias_conv3", 
#                                                     training = True,batch_norm_name = "dec_batch_norm_conv3")
##                    
##                    
    
#                    dec_conv3 = tf.nn.relu(tf.nn.conv2d_transpose(
#                            dec_batchnorm_conv2_b, variable_get([3,3,3,16], 'dec_conv3_filter'),
#                            output_shape = [batch_size,32,32,3],strides = [1,2,2,1], padding = 'SAME', 
#                            name = 'dec_conv3_op'))          
#                    dec_conv3_b = tf.nn.bias_add(dec_conv3, variable_get([3],'dec_conv3_bias'))
#                    dec_batchnorm_conv3_b = tf.layers.batch_normalization(dec_conv3_b, training=True,name = 'dec_batch_norm_conv3')
#            
                with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
                    out_conv = tf.nn.relu(tf.nn.conv2d(
                            dec_batchnorm_conv3_b, variable_get([3,3,3,1], 'out_conv_filter'),strides = [1,1,1,1], 
                            padding = 'SAME'))
                    output = tf.nn.bias_add(out_conv, variable_get([1],'out_conv_bias'))  
        
            
            
#            dict_index_to_output_tensor[key_target] = output
            
#            dict_index_to_output_tensor = tf.convert_to_tensor(dict_index_to_output_tensor)
            
            
            output_list.append(output)
            index = len(output_list) - 1
            index_dict[key_target] = index
            
            target = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size,
                                 col*patch_size:(col*patch_size)+patch_size,:]
            
#            loss = loss + tf.nn.l2_loss(target-output, name = 'l2_loss')                     
            loss = loss + tf.reduce_sum(tf.pow((target-output),2))
            
            
#            padding_top = (row *patch_size)
#            padding_bottom = h_copy_lf_batch_holder - (padding_top + patch_size)
#            
#            padding_left = col * patch_size
#            padding_right = w_copy_lf_batch_holder - (padding_left + patch_size)
#                                 
#            paddings = tf.constant([[0, 0,], [padding_top, padding_bottom],[padding_left,padding_right],[0,0]])
#            
#            pad_output = tf.pad(output, paddings, "CONSTANT")
#            pad_target = tf.pad(target, paddings, "CONSTANT")
#            
#            copy_lf_batch_holder = copy_lf_batch_holder - pad_target + pad_output
            
    
    output_stack = tf.stack(output_list)
    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars, name = 'train_gradients'),
                                       5, name = "clip_gradients_train")
    optimizer = tf.train.GradientDescentOptimizer(0.01, name = "gradient_descent_train")
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step(), name = "apply_gradients_train")

    
    
#    opt = tf.train.GradientDescentOptimizer(0.01)

#    global_step = tf.train.get_or_create_global_step()    
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
#    with tf.control_dependencies(update_ops):
#        train = opt.minimize(loss, global_step=global_step)
    
    
    
    g.add_to_collection("loss_",loss)
    g.add_to_collection("output_stack",output_stack)
    
#    g.add_to_collection('dict_index_to_output_tensor',dict_index_to_output_tensor)
    
    
    print ('exporting_graph...')    
    tf.train.export_meta_graph('./models.meta')
    print ('graph_exported')
    
    
loss_list = [] 
batch_count = 0

with g.as_default():
    lf_data = tf.data.Dataset.from_tensor_slices(lf_dataset)
    
    lf_data = lf_data.batch(batch_size)
    batch_iter = lf_data.make_initializable_iterator()
    next_batch = batch_iter.get_next()
    
with g.as_default():
        
    saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir = './saver/checkpoint_saver', save_steps = 1)
    summary_hook = tf.train.SummarySaverHook(output_dir = './saver/summary',save_steps =2,scaffold=tf.train.Scaffold())
    
    with tf.train.MonitoredTrainingSession(hooks = [saver_hook, summary_hook],checkpoint_dir = './saver/checkpoint_saver') as sess:
        sess.run(batch_iter.initializer)
        
        while not sess.should_stop():
            g_step = g.get_tensor_by_name('global_step:0')
            print ('global_step : {}'.format(sess.run(g_step)))
        
            train_batch = sess.run(next_batch)
        
#            fetches = {'train_op': train_op,
#                       'loss': loss}
#        
            sess.run(train_op,feed_dict = {lf_batch_holder : train_batch})
            loss_ = sess.run(loss, feed_dict = {lf_batch_holder : train_batch})
          
            print ('batch_count : {}'.format(batch_count))
            batch_count = batch_count + 1

            print ('loss : {}'.format(loss_))
            loss_list.append(loss_)
#            if (batch_count > 10):
#                break

    
np.save('loss.npy', np.array(loss_list))    
    
#print ("VERSION", tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)))
            

#cv2.destroyAllWindows()
#cv2.waitKey(1)
#
#    

#
#WARNING:tensorflow:Issue encountered when serializing dict_index_to_output_tensor.
#Type is unsupported, or the types of the items don't match field type in CollectionDef. Note this is a warning and probably safe to ignore.
#'dict' object has no attribute 'name'
#WARNING:tensorflow:Issue encountered when serializing dict_index_to_output_tensor.