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
import tensorflow as tf 

import generate_lf_dataset


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

flags = tf.flags
logging = tf.logging

dot = ""

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

FLAGS = flags.FLAGS


class TrainConfigSmall(object):
  """Small config."""
  strides = 8
  patch_size = 32
  hidden_size = 1024
  starter_learning_rate = 0.1
  decay_steps = 150
  decay_rate = 0.96
  epochs = 15
  batch_size = 100
  channels = 1
  gradient_clip = 5 
  is_training = True
  base_dir =  dot + "/tmp/data/training/"
  save_path = dot + "/saver/"
#  base_dir = "/Users/rohit/Downloads/Random/benchmark/training"

class TestConfig(object):
  """Small config."""
  strides = 8
  patch_size = 32
  hidden_size = 1024
  epochs = ""
  batch_size = 100
  channels = 1
  is_training = False
  base_dir = dot + "/tmp/data/test/"
  save_path = dot + "/saver/"


def get_trainconfig():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = TrainConfigSmall()
  return config

def get_testconfig():
    config = TestConfig()
    return config

strides = 8
patch_size = 32
train_percent = 0.80
batch_count = 0

train_config = get_trainconfig()
test_config = get_testconfig()

lr_path = train_config.save_path + 'lr.npy'
loss_path = train_config.save_path + 'loss.npy'
lr_list, loss_list = generate_lf_dataset.get_lr_loss_list(lr_path, loss_path)

lf_dim, lf_train_data = generate_lf_dataset.get_data_config(train_config)
print ('lf_train_data', np.shape(lf_train_data))

lf_dim, lf_test_data = generate_lf_dataset.get_data_config(test_config)
print ('lf_test_data', np.shape(lf_test_data))
  
train_config.lf_dim = test_config.lf_dim = lf_dim 


class Model(object):
    def __init__(self, is_training, config, name):
        self.name = name
        self.config = config
        self.output_list = []
        self.index_dict = {}
        self.loss = 0
        self.lr_list = lr_list
        self.loss_list = loss_list
        
        H = W = self.config.lf_dim*self.config.patch_size
        
        self.copy_lf_batch_holder = tf.placeholder(tf.float32, shape = (None,H,W,config.channels), name = name + 'data_placeholder')
    
        self.struct(self.copy_lf_batch_holder, is_training)
        
        if(not is_training):
            return
        
        self.tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars, name = 'train_gradients'),
                                      self.config.gradient_clip, name = "clip_gradients_train")
    
        self.global_step=tf.train.get_or_create_global_step()      
        self.learning_rate = tf.train.exponential_decay(self.config.starter_learning_rate, self.global_step,
                                           self.config.decay_steps, self.config.decay_rate, staircase=False)
            
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name = "gradient_descent_train")
    
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars),
                global_step = self.global_step, name = "apply_gradients_train")

        print ('exporting_graph...')    
#        tf.train.export_meta_graph(dot +'/saver/models.meta')
        print ('graph_exported')


    def struct(self,copy_lf_batch_holder, is_training):
            
        cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                    reuse = not is_training)

        self.initial_state = cell.zero_state(self.config.batch_size, tf.float32)   
        
        
        tf_resize_tensor = tf.constant([self.config.patch_size, self.config.patch_size], name= self.name + 'interpolate_to_size')

        for col in range(1,self.config.lf_dim):
            
            state = self.initial_state
        
            for row in range(1,self.config.lf_dim):
                        
                seq_input, key_target = self.prepare_input(row, col, copy_lf_batch_holder, is_training)
            
                output, state_ = self.layers(row, col, seq_input, cell, state, tf_resize_tensor, is_training)
                state = state_
            
                output = tf.identity(output, name="output_from_layers")
            
                self.output_list.append(output)
                index = len(self.output_list) - 1
                self.index_dict[key_target] = index
            
                target = copy_lf_batch_holder[:,row*patch_size:(row*patch_size)+patch_size,
                                 col*patch_size:(col*patch_size)+patch_size,:]
            
                target = tf.identity(target, name="target_original")
                self.loss = self.loss + tf.reduce_sum(tf.pow((target-output),2))
                
        self.output_stack = tf.stack(self.output_list)
        
    
    def prepare_input(self,row, col, lf_batch_holder, is_training):
    
        key_left,key_left_top,key_top,key_target = self.get_keys(row, col)
        print ('{} , {}, {} <=> {}'.format(key_left,key_left_top,key_top,key_target))
            
        left, left_top, top = self.get_slices(lf_batch_holder, row, col, key_left,key_left_top,key_top,key_target)

        seq_input = []

        for batch in range(self.config.batch_size):
            seq_input.append(left[batch,:,:,:])
            seq_input.append(left_top[batch,:,:,:])
            seq_input.append(top[batch,:,:,:])

        seq_input = tf.convert_to_tensor(seq_input, name = "sequence_input_tensor")
        seq_input = tf.reshape(seq_input, (-1,self.config.patch_size,self.config.patch_size,3*self.config.channels))
        return seq_input, key_target
    
    
    def get_keys(self,row, col):
        key_left = (str(row)+str(col-1)) 
        key_left_top = (str(row-1)+str(col-1))
        key_top = (str(row-1)+str(col))
        key_target = (str(row)+str(col))
    
        return key_left,key_left_top,key_top,key_target

    def get_slices(self,lf_batch_holder, row, col, key_left,key_left_top,key_top,key_target):
        
        rw_ptch = row*self.config.patch_size
        cl_ptch = col*self.config.patch_size
        ptch = self.config.patch_size

        left = lf_batch_holder[:,rw_ptch:(row+1)*ptch,(col-1)*ptch:cl_ptch,:]
        left_top = lf_batch_holder[:,(row-1)*ptch:rw_ptch, 
                                            (col-1)*ptch:cl_ptch,:]
        top = lf_batch_holder[:,(row-1)*ptch:rw_ptch, 
                                       cl_ptch:(col+1)*ptch,:]
        
        if ((row == 1) or (col == 1)):
            if((row == 1) and (col == 1)):
                left = left
                left_top = left_top
                top = top
                    
            elif((row != 1) and (col == 1)):
                left = left
                left_top = left_top      
                top = self.output_list[self.index_dict[key_top]]

            elif((row == 1) and (col != 1)):
                left = self.output_list[self.index_dict[key_left]]
                left_top = left_top      
                top = top
                    
        else:
            left = self.output_list[self.index_dict[key_left]]
            left_top = self.output_list[self.index_dict[key_left_top]]
            top = self.output_list[self.index_dict[key_top]]
    
        return left, left_top, top


    def layers(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):
    
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
                batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3,16], 
                                                         kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv1",
                                                         batch_norm_name = "batch_norm_conv1",
                                                         is_training = is_training)
                
            with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
                batchnorm_conv2_b = self.conv2D_batchnorm(input_ = batchnorm_conv1_b, kernel_shape = [3,3,16,32], 
                                                         kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                         padding = "SAME",bias_name = "bias_conv2",
                                                         batch_norm_name = "batch_norm_conv2",
                                                         is_training = is_training)

                    
            with tf.variable_scope('conv_3',reuse=tf.AUTO_REUSE):
                batchnorm_conv3_b = self.conv2D_batchnorm(input_ = batchnorm_conv2_b, kernel_shape = [3,3,32,64], 
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
                dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = out_bicubic, kernel_shape = [3,3,64,32], 
                                                         kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv1",
                                                         batch_norm_name = "dec_batch_norm_conv1",
                                                         is_training = is_training)
                    
            with tf.variable_scope('dec_conv2_',reuse=tf.AUTO_REUSE):
                dec_batchnorm_conv2_b = self.conv2D_batchnorm(input_ = dec_batchnorm_conv1_b, kernel_shape = [3,3,32,16], 
                                                         kernel_name = "dec_conv2_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv2",
                                                         batch_norm_name = "dec_batch_norm_conv2",
                                                         is_training = is_training)
                
            with tf.variable_scope('dec_conv3_',reuse=tf.AUTO_REUSE):
                dec_batchnorm_conv3_b = self.conv2D_batchnorm(input_ = dec_batchnorm_conv2_b, kernel_shape = [3,3,16,3], 
                                                         kernel_name = "dec_conv3_filter", strides = [1,1,1,1], 
                                                         padding = "SAME",bias_name = "dec_bias_conv3",
                                                         batch_norm_name = "dec_batch_norm_conv3",
                                                         is_training = is_training)

        with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
            out_conv = tf.nn.relu(tf.nn.conv2d(
                            dec_batchnorm_conv3_b, self.variable_get([3,3,3,1], 'out_conv_filter'),strides = [1,1,1,1], 
                            padding = 'SAME'))
            output = tf.nn.bias_add(out_conv, self.variable_get([1],'out_conv_bias'))         
            
            return output, state

    def variable_get(self,shape, name):
        return tf.get_variable(name, shape, tf.float32)
        

    def conv2D_batchnorm(self,input_, kernel_shape, kernel_name,strides, padding,bias_name,batch_norm_name, is_training):
        conv = tf.nn.relu(tf.nn.conv2d(input_, self.variable_get(kernel_shape, kernel_name),
                                                                            strides = strides, padding = padding,data_format='NHWC'))          
        conv_b = tf.nn.bias_add(conv, self.variable_get([kernel_shape[-1]],name = bias_name))
        batchnorm_conv_b = tf.keras.layers.BatchNormalization(axis = -1)(conv_b, training = is_training) 
        return batchnorm_conv_b
    


class Input(object):
  """The input data."""

  def __init__(self, config, name=None):  
    self.batch_size = config.batch_size
    self.epoch_size = config.epochs
    H = W = config.lf_dim*config.patch_size
    
    self.tf_lf_placeholder = tf.placeholder(tf.float32, shape = (None,H,W,config.channels), name = name + '_placeholder')
    self.tf_lf_data = tf.data.Dataset.from_tensor_slices(self.tf_lf_placeholder)
 
    if (config.is_training):
        self.tf_lf_data = self.tf_lf_data.shuffle(buffer_size = 10000).repeat(config.epochs).batch(config.batch_size)
    else:
        self.tf_lf_data = self.tf_lf_data.shuffle(buffer_size = 10000).repeat().batch(config.batch_size)

    self.batch_iter = self.tf_lf_data.make_initializable_iterator()
    self.next_batch = self.batch_iter.get_next()


def export_ops(m):
    g.add_to_collection("loss_",m.loss)
    g.add_to_collection("output_stack",m.output_stack)
    g.add_to_collection("train_op",m.train_op)
    g.add_to_collection("lr",m.learning_rate)
    

g = tf.Graph()

with g.as_default():
    with tf.name_scope("Train"):
        train_input = Input(train_config,'Train_input')
        print ('train_input_done')
        with tf.variable_scope("Model", reuse=None):
            m_train = Model(is_training = True, config = train_config, name = 'Train')
            m_train.train_input = train_input
            export_ops(m_train)
    
    print ('train_model_done')
    
    with tf.name_scope("Test"):
        test_input = Input(test_config,'Test_input')
        print ('test_input_done')
        with tf.variable_scope("Model", reuse=True):
            m_test = Model(is_training = False,config = test_config, name = "Test")
            m_test.test_input = test_input
     
print ('***********ready to graph **************')


with g.as_default():
        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir = train_config.save_path +'checkpoint_saver', save_steps = 10)
        summary_hook = tf.train.SummarySaverHook(output_dir = train_config.save_path + 'summary',save_steps =2,scaffold=tf.train.Scaffold())    
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        
        print ('**************** entering in session ****************')
        
        with tf.train.MonitoredTrainingSession(config = config,hooks = [saver_hook],
                                           checkpoint_dir = train_config.save_path + 'checkpoint_saver') as sess:
            sess.run(m_train.train_input.batch_iter.initializer, feed_dict={m_train.train_input.tf_lf_placeholder : lf_train_data})
            sess.run(m_test.test_input.batch_iter.initializer, feed_dict={m_test.test_input.tf_lf_placeholder : lf_test_data})
            
            print ('**************** inside the session ****************')
            
            while not sess.should_stop():
                
                g_step_no = sess.run(m_train.global_step)
                print ('global_step : {}'.format(g_step_no))
                
                train_batch = sess.run(m_train.train_input.next_batch)
                test_batch = sess.run(m_test.test_input.next_batch)
                
                print ("train_batch_shape :{}".format(train_batch.shape))
                print ("test_batch_shape :{}".format(test_batch.shape))
                
                lr = sess.run(m_train.learning_rate)
                print ('learning_rate :{}'.format(lr))                
                m_train.lr_list.append(lr)
                
                if(train_batch.shape[0] != train_config.batch_size):
                    print ("data_finished")
                    break
                
                sess.run(m_train.train_op,feed_dict = {m_train.copy_lf_batch_holder : train_batch})
                
                loss_ = sess.run(m_test.loss, feed_dict = {m_test.copy_lf_batch_holder : test_batch})
                
                print ('loss : {}'.format(loss_))
                m_train.loss_list.append(loss_)
                
                            
                np.save(loss_path, np.array(m_train.loss_list))   
            
                np.save(lr_path, np.array(m_train.lr_list))            

                batch_count = batch_count + 1
#                if (batch_count > 10):
#                    break

        print ('************ phir - milenge ********')
                
