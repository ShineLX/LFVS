#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:03:17 2019

@author: rohit
"""



import tensorflow as tf


tf.reset_default_graph()

t = tf.Graph()
with t.as_default():
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    saver = tf.train.Saver()
    sess = tf.Session()
    
    try:
        ckpt = tf.train.get_checkpoint_state("/Users/rohit/Documents/LFVS/saver-restore-model/")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("restored")
    except:
        sess.run(tf.global_variables_initializer())   
        print ("initialized")
#    saver.restore(sess, '/Users/rohit/Documents/LFVS/saver-restore-model/sess_data')
    w1_ = sess.run(w1)
    w2_ = sess.run(w2)
#    w1_ = w1_ + 1
#    w2_ = w2_ + 1
    print (w1_)
    print (w2_)
    saver.save(sess, '/Users/rohit/Documents/LFVS/saver-restore-model/sess_data', global_step  = 100)

    
    
"""
w1_
Out[32]: array([-0.2948895 ,  0.35861558], dtype=float32)

w2_
Out[33]: 
array([-1.5645419 , -2.7687674 , -0.9228906 ,  0.28880548, -0.7388    ],
      dtype=float32)

"""