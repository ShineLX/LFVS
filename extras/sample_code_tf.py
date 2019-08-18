#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:46:03 2019

@author: rohit
"""

# Import `tensorflow`
import tensorflow as tf
import numpy as np

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)


ones = np.ones((50,32,32))
twos = np.ones((50,32,32))
twos = twos*2

ones = np.reshape(ones, (50,32,32,1))
twos = np.reshape(twos, (50,32,32,1))
#ones = ones.astype(np.float32)
#twos = twos.astype(np.float32)


tf.reset_default_graph()
g = tf.Graph()


def variable_get(shape, name):
        return tf.get_variable(name, shape, tf.float32)
        

with g.as_default():
    target = tf.placeholder(tf.float32, shape = [None, 32,32,1], name = "target_placeholder")
    input1 = tf.placeholder(tf.float32, shape = [None, 32,32,1], name = "input_placeholder")
#    input1 = tf.reshape(input1,[None,32,32,-1])
    
    conv = tf.nn.relu(tf.nn.conv2d(input1, variable_get([3,3,1,4], "kernel1"),
                                                                            strides = [1,1,1,1], padding = "SAME",data_format='NHWC'))          
    conv_b = tf.nn.bias_add(conv,variable_get([4],name = "bias1"))
    
    conv2 = tf.nn.relu(tf.nn.conv2d(conv_b, variable_get([3,3,4,1], "kernel2"),
                                                                            strides = [1,1,1,1], padding = "SAME",data_format='NHWC'))          
    conv2_b = tf.nn.bias_add(conv2,variable_get([1],name = "bias2"))
    
    loss = tf.reduce_sum(tf.pow((conv2_b-target),2))
    loss = tf.identity(loss, name = "loss_yo")
    loss_summ = tf.summary.scalar("loss_summary", loss)
    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars, name = 'get_gradients_yo'),
                                      5, name = "clip_gradients_yo")
    
#        tf.summary.histogram(name+"_gradients", self.grads)
        
    global_step=tf.train.get_or_create_global_step()
#    global_step = tf.identity(global_step, name = "global_step_yo")      
        #self.learning_rate = tf.train.exponential_decay(self.config.starter_learning_rate, self.global_step,
#                                           self.config.decay_steps, self.config.decay_rate, staircase=False)
            
        
    optimizer = tf.train.GradientDescentOptimizer(0.01, name = "gradient_descent_optimizer_yo")
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step = global_step, name = "apply_gradients_yo")
            
with g.as_default():
    ones_t = tf.convert_to_tensor(ones, dtype=tf.float32)
    twos_t = tf.convert_to_tensor(twos, dtype=tf.float32)

    ones_t = tf.identity(ones_t, name = "input_ones_yo")
    twos_t = tf.identity(twos_t, name = "output_twos_yo")

    data = tf.data.Dataset.from_tensor_slices((ones_t, twos_t))        
    data = data.batch(10)
    iter1 = data.make_initializable_iterator(shared_name = "initializable_iter_yo")
    next1 = iter1.get_next()
    
#    train_writer = tf.summary.FileWriter("/Users/rohit/Documents/LFVS/extras/writer")
    merge_summary = tf.summary.merge_all()
    merge_summary = tf.summary.merge([loss_summ])

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("/Users/rohit/Documents/LFVS/extras/writer",sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(iter1.initializer)
        for i in range(5):
            batch = sess.run(next1)
            batch = np.array(batch)
            print (batch.shape)  
            train_batch = batch[0,:,:,:]
            test_batch = batch[1,:,:,:]
            
            fetch = {"loss" : loss, 
                     "train_op" : train_op,
                     "summary": merge_summary}
            
            fetch_res = sess.run(fetch, feed_dict = {input1 : train_batch, target: test_batch})
            
            loss_train = fetch_res["loss"]
            loss_summary = fetch_res["summary"]
            
            print ("loss_train : {}".format(loss_train))            
            
            train_writer.add_summary(loss_summary,i)
            train_writer.flush()
            
            
            
            
        

        
    
    

