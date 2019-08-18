#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:19:26 2019

@author: rohit
"""

import tensorflow as tf


#our_model model 1
def layers_new(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    print ("inside our_model layers_new*****")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,16], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
            
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [3,3,16,32], 
                                                     kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            
        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
            
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
        
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
            
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
          
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            
            
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,32,16], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
            
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                                      self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            

    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,16,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
                
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        
        return self.output, state
    
#model_2
def layers_new_2(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    print ("inside_layer_2")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,3*self.config.channels], 
                                                     kernel_name = "conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
            
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)
            
            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)

#            lstm_input = tf.reshape(batchnorm_conv3_b, [self.config.batch_size,64,64,64])
        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
            
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)

        
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
        
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)

        

    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
            
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)

        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
            
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)

            
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,32,16], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)



    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,16,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)

        
        return self.output, state
        
    
#model 3
def layers_new_3(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    print ("inside_layer_3")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            batchnorm_conv2_b = self.conv2D_batchnorm(input_ = batchnorm_conv1_b, kernel_shape = [3,3,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

        lstm_input = tf.reshape(batchnorm_conv2_b, [self.config.batch_size,-1])
        lstm_input = tf.identity(lstm_input, name="lstm_input")
            
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(lstm_input, state)
                
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        decoder_input = tf.reshape(cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        decoder_input = tf.identity(decoder_input,name = "decoder_input")
                    
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            out_bicubic = tf.image.resize_bicubic(decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
    
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        out_conv = tf.nn.relu(tf.nn.conv2d(
                        dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        output = tf.nn.bias_add(out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        return output, state
    
def layers_new_4(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = [] 
     
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_4")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [5,5,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,4,4,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)
            
            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")
            
        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
            
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)

        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
                
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)

    
    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        
        return self.output, state
 
    
def layers_new_4_2(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = []
    
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_4_2")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [9,9,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,4,4,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")

        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
#        self.size_list.append(tf.shape(self.lstm_input))
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
                
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
#        self.size_list.append(tf.shape(self.cell_output))
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv1_b))
            self.size_info.append("dec_batchnorm_conv1_b")
    
    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        self.size_list.append(tf.shape(self.output))
        self.size_info.append("output")
        
        return self.output, state
 
    
def layers_new_5(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    print ("inside_layer_5")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)
            
            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
        
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [15,15,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,8,8,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",is_training = is_training)
          
            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
        
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
        
        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input_yo")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
          
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input_yo")
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)           
        
        with tf.variable_scope('bicubic_interpolate_yo',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp_yo')
            
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
            
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                                      self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            
            
    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter_yo'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias_yo'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        
        return self.output, state
    
def layers_new_6(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = []
    
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_6")
    with tf.variable_scope('encoder'):#,reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [17,17,32,128], 
                                                     kernel_name = "conv2_filter", strides = [1,8,8,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")

        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
#        self.size_list.append(tf.shape(self.lstm_input))
        
    with tf.variable_scope('LSTM'):
        print("***Scope : {}".format(tf.get_variable_scope().name))
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
                
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
#        self.size_list.append(tf.shape(self.cell_output))
        
    with tf.variable_scope('decoder'):#, reuse = tf.AUTO_REUSE):
        print("***Scope : {}".format(tf.get_variable_scope().name))
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,128,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv1_b))
            self.size_info.append("dec_batchnorm_conv1_b")
    
    with tf.variable_scope('out_level'):#,reuse=tf.AUTO_REUSE):
        print("***Scope : {}".format(tf.get_variable_scope().name))
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        self.size_list.append(tf.shape(self.output))
        self.size_info.append("output")
        
        return self.output, state
 
def layers_new_6_cuda(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = []
    
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_6")
    with tf.variable_scope('encoder'):#,reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [17,17,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,8,8,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")

        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [1,self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
#        self.size_list.append(tf.shape(self.lstm_input))
        
    with tf.variable_scope('LSTM'):
        print("***Scope : {}".format(tf.get_variable_scope().name))
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('CUDA_LSTM'):
            print ("This is  Cuda_6")
            self.cell_output, state = cell(
                        self.lstm_input,
                        initial_state=None if (row == 1) else state,
                        training = is_training)
                
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
#        self.size_list.append(tf.shape(self.cell_output))
        
    with tf.variable_scope('decoder'):#, reuse = tf.AUTO_REUSE):
        print("***Scope : {}".format(tf.get_variable_scope().name))
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_'):#,reuse=tf.AUTO_REUSE):
            print("***Scope : {}".format(tf.get_variable_scope().name))
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv1_b))
            self.size_info.append("dec_batchnorm_conv1_b")
    
    with tf.variable_scope('out_level'):#,reuse=tf.AUTO_REUSE):
        print("***Scope : {}".format(tf.get_variable_scope().name))
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        self.size_list.append(tf.shape(self.output))
        self.size_info.append("output")
        
        return self.output, state
    
def layers_new_7(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = []
    
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_7")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [7,7,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,4,4,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [5,5,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,4,4,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")

        self.lstm_input = tf.reshape(self.batchnorm_conv2_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
#        self.size_list.append(tf.shape(self.lstm_input))
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
                
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
#        self.size_list.append(tf.shape(self.cell_output))
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv1_b))
            self.size_info.append("dec_batchnorm_conv1_b")
    
    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv1_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        self.size_list.append(tf.shape(self.output))
        self.size_info.append("output")
        
        return self.output, state
    
    
def layers_new_8(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = []
    
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_8")
    with tf.variable_scope('encoder'):#,reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1') as scope:#,reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training,scope = scope)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2'):#,reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [3,3,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training,scope = scope)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")

        with tf.variable_scope('conv3'):#,reuse=tf.AUTO_REUSE):
            self.batchnorm_conv3_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv2_b, kernel_shape = [5,5,64,64], 
                                                     kernel_name = "conv3_filter", strides = [1,4,4,1], 
                                                     padding = "SAME",bias_name = "bias_conv3",
                                                     batch_norm_name = "batch_norm_conv3",
                                                     is_training = is_training,scope = scope)

            self.summary_batchnorm_conv3_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv3_b', self.batchnorm_conv3_b)
            self.summary_list.append(self.summary_batchnorm_conv3_b)
            self.size_list.append(tf.shape(self.batchnorm_conv3_b))
            self.size_info.append("batchnorm_conv3_b")

        self.lstm_input = tf.reshape(self.batchnorm_conv3_b, [self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
#        self.size_list.append(tf.shape(self.lstm_input))
        
    with tf.variable_scope('LSTM'):#, reuse = tf.AUTO_REUSE):
        if not (row == 1 and col == 1):
            print ("LSTM row and col not equals 1")
            tf.get_variable_scope().reuse_variables()
        (self.cell_output, state) = cell(self.lstm_input, state)
                
        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
        self.summary_list.append(self.summary_lstm_output)
#        self.size_list.append(tf.shape(self.cell_output))
        
    with tf.variable_scope('decoder'):#, reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate'):#,reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_'):#,reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,64], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training,scope = scope)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv1_b))
            self.size_info.append("dec_batchnorm_conv1_b")
   

        with tf.variable_scope('dec_conv2_'):#,reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.dec_batchnorm_conv1_b, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv2_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv2",
                                                     batch_norm_name = "dec_batch_norm_conv2",
                                                     is_training = is_training,scope = scope)
   
            self.summary_dec_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv2_b', 
                                                              self.dec_batchnorm_conv2_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv2_b))
            self.size_info.append("dec_batchnorm_conv2_b")
    
    
    with tf.variable_scope('out_level'):#,reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv2_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter',scope = scope),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias', scope = scope))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        self.size_list.append(tf.shape(self.output))
        self.size_info.append("output")
        
        return self.output, state
    

    
def layers_new_8_cuda(self,row, col, seq_input, cell,cuda_cell,cuda_state, state, tf_resize_tensor, is_training):
    self.size_list = []
    self.size_info = []
    
    self.size_list.append(tf.shape(seq_input))
    self.size_info.append("Input")
    
    print ("inside_layer_8")
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,32], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
                        
            self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
            self.summary_list.append(self.summary_batchnorm_conv1_b)

            self.size_list.append(tf.shape(self.batchnorm_conv1_b))
            self.size_info.append("batchnorm_conv1_b")
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv1_b, kernel_shape = [3,3,32,64], 
                                                     kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

            self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
            self.summary_list.append(self.summary_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.batchnorm_conv2_b))
            self.size_info.append("batchnorm_conv2_b")

        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            self.batchnorm_conv3_b = self.conv2D_batchnorm(input_ = self.batchnorm_conv2_b, kernel_shape = [5,5,64,64], 
                                                     kernel_name = "conv3_filter", strides = [1,4,4,1], 
                                                     padding = "SAME",bias_name = "bias_conv3",
                                                     batch_norm_name = "batch_norm_conv3",
                                                     is_training = is_training)

            self.summary_batchnorm_conv3_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv3_b', self.batchnorm_conv3_b)
            self.summary_list.append(self.summary_batchnorm_conv3_b)
            self.size_list.append(tf.shape(self.batchnorm_conv3_b))
            self.size_info.append("batchnorm_conv3_b")

        self.lstm_input = tf.reshape(self.batchnorm_conv3_b, [1,self.config.batch_size,-1])
        self.lstm_input = tf.identity(self.lstm_input, name="lstm_input")
        
        self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
        self.summary_list.append(self.summary_lstm_input)
#        self.size_list.append(tf.shape(self.lstm_input))
        
    with tf.variable_scope('CUDA_LSTM'):
        print ("This is  Cuda")
        self.cuda_outputs, self.cuda_output_states = cuda_cell(
                        self.lstm_input,
                        initial_state=None if (row == 1) else cuda_state,
                        training = is_training)
            #dropout=self.dropout if is_training else 0.
#        with tf.variable_scope('LSTM'):
#            if not (row == 1 and col == 1): 
#                tf.get_variable_scope().reuse_variables()
#            (cell_output, state) = cell(lstm_input, state)
            
        self.cell_output = self.cuda_outputs
        
        
#    with tf.variable_scope('LSTM'):
#        if not (row == 1 and col == 1): 
#            tf.get_variable_scope().reuse_variables()
#        (self.cell_output, state) = cell(self.lstm_input, state)
#                
#        self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
#        self.summary_list.append(self.summary_lstm_output)
#        self.size_list.append(tf.shape(self.cell_output))
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        self.decoder_input = tf.reshape(self.cell_output,[self.config.batch_size,self.config.hidden_dims[0],
                                                self.config.hidden_dims[1], self.config.hidden_dims[2]])
        self.decoder_input = tf.identity(self.decoder_input,name = "decoder_input")
              
        
        self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
        self.summary_list.append(self.summary_decoder_input)
        self.size_list.append(tf.shape(self.decoder_input))
        self.size_info.append("decoder_input")
        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            self.out_bicubic = tf.image.resize_bicubic(self.decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
            self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
            self.summary_list.append(self.summary_out_bicubic)
            self.size_list.append(tf.shape(self.out_bicubic))
            self.size_info.append("out_bicubic")
    
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = self.out_bicubic, kernel_shape = [3,3,64,64], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv1_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv1_b))
            self.size_info.append("dec_batchnorm_conv1_b")
   

        with tf.variable_scope('dec_conv2_',reuse=tf.AUTO_REUSE):
            self.dec_batchnorm_conv2_b = self.conv2D_batchnorm(input_ = self.dec_batchnorm_conv1_b, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv2_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv2",
                                                     batch_norm_name = "dec_batch_norm_conv2",
                                                     is_training = is_training)
   
            self.summary_dec_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv2_b', 
                                                              self.dec_batchnorm_conv2_b)
            self.summary_list.append(self.summary_dec_batchnorm_conv2_b)
            self.size_list.append(tf.shape(self.dec_batchnorm_conv2_b))
            self.size_info.append("dec_batchnorm_conv2_b")
    
    
    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        self.out_conv = tf.nn.relu(tf.nn.conv2d(
                        self.dec_batchnorm_conv2_b, self.variable_get([3,3,32,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        self.output = tf.nn.bias_add(self.out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
        self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
        self.summary_list.append(self.output_layer_summary)
        self.size_list.append(tf.shape(self.output))
        self.size_info.append("output")
        
        return self.output, state, self.cuda_output_states
    
    