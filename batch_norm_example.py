#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:13:30 2019

@author: rohit
"""

import tensorflow as tf

k = tf.Graph()

with k.as_default():
    x = tf.placeholder(tf.float32, [None, 1], 'x')
    is_training = tf.placeholder_with_default(False, (), 'is_training')
    y = tf.layers.batch_normalization(x, training = is_training)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        y = tf.identity(y)



with k.as_default():
    x_1 = tf.placeholder(tf.float32, [None, 1], 'x')
    is_training_1 = tf.placeholder_with_default(False, (), 'is_training')
    y_1 = tf.layers.batch_normalization(x, training = is_training)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        y = tf.identity(y)



with k.as_default():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
#    y_1 = sess.run([y, update_ops], feed_dict={x: [[-10], [0], [10]],is_training: True})[0]
#    y_2 = sess.run(y, feed_dict={x: [[-10]],is_training: False})
 
    for _ in range(100):
        y_1 = sess.run([y], feed_dict={x: [[-10], [0], [10]], is_training: True})[0]
    y_2 = sess.run(y, feed_dict={x: [[-10]], is_training: False})    
    print (y_1)
    print (y_2)
#    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
#        out = sess.run([tf.get_variable('batch_normalization/moving_mean'),
#                    tf.get_variable('batch_normalization/moving_variance')])
#    moving_average, moving_variance = out
#    y_out = sess.run(y, feed_dict={x: [[-10], [0], [10]]})
#    print (y_out)
    sess.close()
    
    