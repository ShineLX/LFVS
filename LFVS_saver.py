#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 02:04:02 2019

@author: rohit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:48:15 2019

@author: rohit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:57:32 2019

@author: rohit
"""

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
import os

tf.reset_default_graph()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("dot", "",
                        "extension")
flags.DEFINE_bool("GPU", "False",
                        "Runnin local or remote")
flags.DEFINE_bool("use_cudaLSTM", "False",
                        "Use basic lstm cell or cudnn LSTM")

flags.DEFINE_integer("patch", 128,
                     "patch_size to be used")
flags.DEFINE_integer("channels", 1,
                     "number of channels")
flags.DEFINE_integer("allowed_lf_dim", 2,
                     "number of channels")
flags.DEFINE_integer("batch_size", 100,
                     "number of examples in one batch")
flags.DEFINE_integer("epochs", 3,
                     "number of channels")
flags.DEFINE_integer("num_files", 1,
                     "number of files to load data from")

flags.DEFINE_string("save_path", "/saver-extra/checkpoint_saver/",
                        "save path for tensorflow data")
flags.DEFINE_float("learning_rate", 0.001,
                     "learning_rate")
flags.DEFINE_integer("gradient_clip", 5,
                     "gradient clip")
flags.DEFINE_integer("strides", 8,
                     "strides")
flags.DEFINE_integer("save_steps", 1000,
                     "checkpoint_save_steps")
flags.DEFINE_integer("summary_save_steps", 100,
                     "summary_save_steps")
flags.DEFINE_integer("save_output_steps", 2000,
                     "save test output in every number of these steps")
flags.DEFINE_string("model", "model_6_cudalstm_8_128_layers_batchnorm_efficient_files_1",
                     "model_info")
flags.DEFINE_list("lr_list", [0.0001],#[0.1,0.01,0.001,0.0001,0.00001],
                     "list of learning rates")
flags.DEFINE_float("dropout_rate", 1.0,
                     "dropout_rate")

flags.DEFINE_string("path_train_data", "/saver-1000/",
                        "path for tensorflow data")
flags.DEFINE_string("path_test_data", "/saver-500-3/",
                        "path for tensorflow data")


FLAGS = flags.FLAGS

hidden_dims = [int(FLAGS.patch/16),int(FLAGS.patch/16),128]

class TrainConfigSmall(object):
  """Small config."""
  strides = FLAGS.strides
  patch_size = FLAGS.patch
  hidden_size = hidden_dims[0]*hidden_dims[1]*hidden_dims[2] 
  starter_learning_rate = 0.05
  decay_steps = 100
  decay_rate = 0.96
  epochs = FLAGS.epochs
  batch_size = FLAGS.batch_size
  channels = FLAGS.channels
  gradient_clip = FLAGS.gradient_clip
  is_training = True
  if (FLAGS.dot == "."):
      base_dir = "/Users/rohit/Downloads/Random/benchmark/training"
  else:
      base_dir =  FLAGS.dot + "/saver-2/training/"    
  save_path = FLAGS.dot + FLAGS.save_path + str(FLAGS.model) + "/" 
  num_files = FLAGS.num_files
  train_single_image = True
  allowed_lf_dim = FLAGS.allowed_lf_dim
  learning_rate = FLAGS.learning_rate
  hidden_dims = hidden_dims
  dropout_rate = 1.0
  use_cudaLSTM = FLAGS.use_cudaLSTM


class TestConfig(object):
  """Small config."""
  strides = FLAGS.strides
  patch_size = FLAGS.patch
  hidden_size = hidden_dims[0]*hidden_dims[1]*hidden_dims[2] #(FLAGS.patch/4)*(FLAGS.patch/4)*32
  epochs = FLAGS.epochs
  batch_size = FLAGS.batch_size
  channels = FLAGS.channels
  is_training = False
  if (FLAGS.dot == "."):
      base_dir = "/Users/rohit/Downloads/Random/benchmark/test"
  else:
      base_dir =  FLAGS.dot + "/saver-2/test/"
  save_path = FLAGS.dot + FLAGS.save_path + str(FLAGS.model) + "/"
  num_files = FLAGS.num_files
  train_single_image = True
  allowed_lf_dim = FLAGS.allowed_lf_dim
  hidden_dims = hidden_dims
  dropout_rate = 1.0
  save_output_step = FLAGS.save_output_steps
  use_cudaLSTM = FLAGS.use_cudaLSTM

def get_trainconfig():
  """Get model config."""
  config = TrainConfigSmall()
  return config

def get_testconfig():
    config = TestConfig()
    return config

batch_count = 0

train_config = get_trainconfig()
test_config = get_testconfig()

if not(FLAGS.GPU):
    lf_train_data = generate_lf_dataset.load_patched_data(FLAGS.dot + FLAGS.path_train_data, FLAGS.num_files)
    print ("lf_train_data shape : {}".format(lf_train_data.shape))

    lf_test_data = generate_lf_dataset.load_patched_data(FLAGS.dot + FLAGS.path_test_data,FLAGS.num_files)
    print ("lf_test_data shape : {}".format(lf_test_data.shape))
else:
    lf_train_data = np.load(FLAGS.dot + FLAGS.path_train_data + "train_local_128.npy")
    print ("lf_train_data shape : {}".format(lf_train_data.shape))
    
    lf_test_data = np.load(FLAGS.dot + FLAGS.path_test_data + "test_local_128.npy")
    print ("lf_test_data shape : {}".format(lf_test_data.shape))
    

lf_dim = 9
train_config.lf_dim = test_config.lf_dim = lf_dim 


class Model(object):
    
    
    if(FLAGS.use_cudaLSTM):
        from _layers import layers_new_6_cuda
    else:
        from _layers import layers_new_6
    
    def __init__(self, is_training, config, name):
        
        self.batch_normlist = []
        self.summary_list = []
        self.name = name
        self.config = config
        self.output_list = []
        self.index_dict = {}
#        self.loss = 0
        
        H = W = self.config.allowed_lf_dim*self.config.patch_size
                
        self.copy_lf_batch_holder = tf.placeholder(tf.float32, shape = (None,H,W,config.channels), name = name + 'data_placeholder')
        self.dropout = tf.placeholder_with_default(1.0, shape = (), name = "dropout_holder_yo")
                
        self.struct(self.copy_lf_batch_holder, is_training)
        
        self.network_size_list = tf.stack(self.size_list)
        
        self.tvars = tf.trainable_variables()
                
        self.training_loss_summary = tf.summary.scalar(self.name + "_loss_summary", self.loss)
        self.summary_list.append(self.training_loss_summary)
            
        self.quarter_zero_tensor = tf.zeros([self.config.batch_size, H/2,H/2,self.config.channels], tf.float32)        
        
        self.output_stack = tf.reshape(self.output_stack, [self.config.batch_size, self.config.patch_size, self.config.patch_size, self.config.channels])
        
        self.zero_output = tf.concat(([(self.quarter_zero_tensor),(self.output_stack)]),1)
        self.in_zero_out  = tf.concat(([(self.copy_lf_batch_holder), self.zero_output]),2)

        self.summary_in_zero_out = tf.summary.image(self.name+"_input_zero_output", (self.in_zero_out),max_outputs = 20)    
        self.summary_list.append(self.summary_in_zero_out)
         
        self.output_stack_summary = tf.summary.image(self.name+"_output_stack", (self.output_stack),max_outputs = 10)    
        self.summary_list.append(self.output_stack_summary)
        
        

        if(not is_training):
            return
        
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars, name = 'get_gradients_yo'),
                                      self.config.gradient_clip, name = "clip_gradients_yo")
    
        self.global_step=tf.train.get_or_create_global_step()      
        #self.learning_rate = tf.train.exponential_decay(self.config.starter_learning_rate, self.global_step,
#                                           self.config.decay_steps, self.config.decay_rate, staircase=False)
            
        self.learning_rate = self.config.learning_rate
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name = "gradient_descent_yo")
    
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars),
                global_step = self.global_step, name = "apply_gradients_yo")

    def struct(self,copy_lf_batch_holder, is_training):
            
        print("***Scope : {}".format(tf.get_variable_scope().name))
        self.cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                    reuse = not is_training)

        self.initial_state = self.cell.zero_state(self.config.batch_size, tf.float32)    
        
        tf_resize_tensor = tf.constant([self.config.patch_size, self.config.patch_size], name= self.name + 'interpolate_to_size')


        self.cuda_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers = 1,
            num_units = self.config.hidden_size,
            direction="unidirectional"
            #dropout=self.dropout if is_training else 0.,
            # kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        

        for col in range(1,self.config.lf_dim):

            if(self.config.use_cudaLSTM):
                cuda_state = []
            else:
                state = self.initial_state
        
            for row in range(1,self.config.lf_dim):
                        
                seq_input, key_target = self.prepare_input(row, col, copy_lf_batch_holder, is_training)
            
#                output, state_ = self.layers(row, col, seq_input, cell, state, tf_resize_tensor, is_training)
#                output, state_ = self.layers_new_6(row, col, seq_input, self.cell, state, tf_resize_tensor, is_training)
                if (self.config.use_cudaLSTM):
                    output, cuda_state_ = self.layers_new_6_cuda(row, col, seq_input,self.cuda_cell,cuda_state, tf_resize_tensor, is_training)
                    cuda_state = cuda_state_
                else:
                    output, state_ = self.layers_new_6(row, col, seq_input, self.cell, state, tf_resize_tensor, is_training)
                    state = state_
                
                output = tf.identity(output, name="output_from_layers_yo")
            
                self.output_list.append(output)
                index = len(self.output_list) - 1
                self.index_dict[key_target] = index
            
                target = copy_lf_batch_holder[:,row*self.config.patch_size:(row*self.config.patch_size)+self.config.patch_size,
                                 col*self.config.patch_size:(col*self.config.patch_size)+self.config.patch_size,:]
            
                target = tf.identity(target, name="target_original_yo")
                
                if (row == 1 and col == 1):
                    self.loss = tf.reduce_sum(tf.pow((target-output),2))
                else:
                    self.loss = self.loss + tf.reduce_sum(tf.pow((target-output),2))
                
                if(self.config.train_single_image):
                    break
            
            if(self.config.train_single_image):
                break
                
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


    def variable_get(self,shape, name):
#        scope.reuse_variables()
#        print ("This is the scope : {}".format(scope))
        var =  tf.get_variable(name, shape, tf.float32)
        print (var.name)
        return var
        

    def conv2D_batchnorm(self,input_, kernel_shape, kernel_name,strides, padding,bias_name,batch_norm_name, is_training):
        conv = (tf.nn.conv2d(input_, self.variable_get(kernel_shape, kernel_name),
                                                                            strides = strides, padding = padding,data_format='NHWC'))          
        conv_b = tf.nn.bias_add(conv, self.variable_get([kernel_shape[-1]],name = bias_name))
        
#        batchnorm_conv_b = tf.keras.layers.BatchNormalization(axis = -1)(conv_b, training = is_training) 
        
        batchnorm_conv_b = tf.layers.batch_normalization(conv_b,axis = -1, training = is_training) 
        
        
        conv_relu = tf.nn.relu(batchnorm_conv_b)
        return conv_relu
    
#        conv = tf.nn.relu(tf.nn.conv2d(input_, self.variable_get(kernel_shape, kernel_name),
#                                                                            strides = strides, padding = padding,data_format='NHWC'))          
#        conv_b = tf.nn.bias_add(conv, self.variable_get([kernel_shape[-1]],name = bias_name))
#        batchnorm_conv_b = tf.keras.layers.BatchNormalization(axis = -1)(conv_b, training = is_training) 
#        return batchnorm_conv_b
    


class Input(object):
  """The input data."""

  def __init__(self, config, name=None):  
    self.batch_size = config.batch_size
    self.epoch_size = config.epochs
    
    H = W = config.allowed_lf_dim*config.patch_size
    
    self.tf_lf_placeholder = tf.placeholder(tf.float32, shape = (None,H,W,config.channels), name = name + '_placeholder_yo')
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
    
lr_list = FLAGS.lr_list
      
for lr in lr_list:     
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("Train"):
            train_input = Input(train_config,'Train_input')
            print ('train_input_done')
            with tf.variable_scope(FLAGS.model, reuse=None):
                m_train = Model(is_training = True, config = train_config, name = 'Train')
                m_train.train_input = train_input
#                export_ops(m_train)
        
        print ('train_model_done') 
        
        with tf.name_scope("Test"):
            test_input = Input(test_config,'Test_input')
            print ('test_input_done')
            with tf.variable_scope(FLAGS.model, reuse= tf.AUTO_REUSE):
                m_test = Model(is_training = False,config = test_config, name = "Test")
                m_test.test_input = test_input
                
                
#        merge_train_summ,merge_test_summ = generate_lf_dataset.get_summary(m_train, m_test,g)
        
        if not(FLAGS.use_cudaLSTM):
            generate_lf_dataset.gradient_tvars(m_train,g)
            generate_lf_dataset.gradient_tvars(m_test,g)
        
        m_train.model_info_dict = {}
        
        
#        train_batchnorm_summary = tf.summary.merge(m_train.batch_normlist)
#        test_batchnorm_summary = tf.summary.merge(m_test.batch_normlist)
        
        
        train_summary_merge = tf.summary.merge(m_train.summary_list)
        test_summary_merge = tf.summary.merge(m_test.summary_list)
        
        print ('***********ready to graph **************')     
                        
        lr_path = "lr_" + str(lr) + "/"
        print ("now going for lr : {}".format(lr))
    
        lr_save_path,loss_path, loss_train_path, test_output_path = generate_lf_dataset.get_list_save_paths(train_config, lr_path)
        lr_list, loss_list, loss_train_list = generate_lf_dataset.get_lr_loss_list(lr_save_path, loss_path,loss_train_path)

#        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir = train_config.save_path + lr_path, save_steps = FLAGS.save_steps)
#        summary_hook = tf.train.SummasrySaverHook(output_dir = train_config.save_path + lr_path + 'summary',save_steps =2,scaffold=tf.train.Scaffold())    
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        
        if not os.path.exists(train_config.save_path + lr_path):
            os.makedirs(train_config.save_path + lr_path)
            
        
        print ('**************** entering in session ****************')
        saver = tf.train.Saver(max_to_keep = 2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
#            saver.restore(sess, os.getcwd() + train_config.save_path[1:] + lr_path +"model.ckpt")
#            saver.restore(sess, os.getcwd() + train_config.save_path[1:] + lr_path)
            sess.run(m_train.train_input.batch_iter.initializer, feed_dict={m_train.train_input.tf_lf_placeholder : lf_train_data})
            sess.run(m_test.test_input.batch_iter.initializer, feed_dict={m_test.test_input.tf_lf_placeholder : lf_test_data})
            
            generate_lf_dataset.write_model_information(FLAGS, m_train.config.save_path + lr_path + "model_information.txt", lr, hidden_dims)
            train_writer = tf.summary.FileWriter(train_config.save_path + lr_path + "tensorboard_train/", sess.graph)
            test_writer = tf.summary.FileWriter(train_config.save_path + lr_path + "tensorboard_test/", sess.graph)
            print ('**************** inside the session ****************')
            
            while (True):
                
                g_step_no = sess.run(m_train.global_step)
                print ('global_step : {}'.format(g_step_no))
                
                try:
                    train_batch = sess.run(m_train.train_input.next_batch)
                    test_batch = sess.run(m_test.test_input.next_batch)
                except:
                    print ("training_data_finished")
                    break
                
                if(train_batch.shape[0] != train_config.batch_size):
                    print ("data_finished")
                    break
          
                
                if (g_step_no == 0):
                    size_results = sess.run(m_train.network_size_list,feed_dict = {m_train.copy_lf_batch_holder : train_batch})
                    for layers in range(size_results.shape[0]):
                        print ("======>Size_list : {}   :  {}".format(size_results[layers,:], m_train.size_info[layers]))
                        m_train.model_info_dict[m_train.size_info[layers]] =  size_results[layers,:]
                    with open(m_train.config.save_path + lr_path + "model_information.txt","a") as f:
                        f.write("model_info_dict : " + str(m_train.model_info_dict) + "\n")    
                    
#                in_out =  sess.run(m_train.in_zero_out,feed_dict = {m_train.copy_lf_batch_holder : train_batch})
#                print ("------- in_out_shape : {}".format(in_out.shape))
                        
                fetch_train = { "loss_training" : m_train.loss,
                         "train_op" : m_train.train_op
                        }
                
                train_results = sess.run(fetch_train,feed_dict = {m_train.copy_lf_batch_holder : train_batch})
                
                loss_t = train_results["loss_training"]
                print ('loss_training : {}'.format(loss_t))
#                loss_train_list.append(loss_t)
#                np.save(loss_train_path, np.array(loss_train_list))   
                
                
#                train_loss_again = sess.run(m_train.loss,feed_dict = {m_train.copy_lf_batch_holder : train_batch})
#                print ("train_loss_again : {}".format(train_loss_again))
                
                
                fetch_loss = {"test_output" : m_test.output_stack,
                              "test_loss" : m_test.loss
                              }
                
#                test_loss_train_data = sess.run(m_test.loss, feed_dict = {m_test.copy_lf_batch_holder : train_batch}) 
#                print ("test_loss_train_data : {}".format(test_loss_train_data))
                
                
                test_results = sess.run(fetch_loss, feed_dict = {m_test.copy_lf_batch_holder : test_batch}) 

                loss_ = test_results["test_loss"]
                print ('loss_test : {}'.format(loss_))
#                loss_list.append(loss_)                            
#                np.save(loss_path, np.array(loss_list))   
                
                test_out = np.squeeze(test_results["test_output"])
                print ("test_output_shape : {}".format(np.shape(test_out)))
                
                if((g_step_no % m_test.config.save_output_step) == 0):   
                    if not os.path.exists(test_output_path + "/output/output_" + str(g_step_no) ):
                        os.makedirs(test_output_path + "/output/output_" + str(g_step_no))
                    np.save(test_output_path + "/output/output_" + str(g_step_no) + "/"+"test_output_" + str(g_step_no) + ".npy", np.array(test_out))   
                    np.save(test_output_path + "/output/output_" + str(g_step_no) + "/"+"test_input_" + str(g_step_no) + ".npy", np.array(test_batch))

                if (g_step_no % FLAGS.save_steps == 0):
                    save_path = saver.save(sess, train_config.save_path + lr_path +"model.ckpt", global_step = g_step_no)
#                    save_path = saver.save(sess, train_config.save_path + lr_path, global_step = g_step_no)
                    print("Model saved in path: %s" % save_path)
                    
                if (g_step_no % FLAGS.summary_save_steps == 0):
                    train_summary = sess.run(train_summary_merge, feed_dict = {m_train.copy_lf_batch_holder : train_batch})
                    train_writer.add_summary(train_summary,g_step_no)
                    train_writer.flush()
                
                    test_summary = sess.run(test_summary_merge, feed_dict = {m_test.copy_lf_batch_holder : test_batch})
                    test_writer.add_summary(test_summary,g_step_no)
                    test_writer.flush()
                    print("Summary saved at {}".format(train_config.save_path + lr_path + "tensorboard_"))
                    
                batch_count = batch_count + 1
                
            train_writer.close()
            test_writer.close()
                
#                if (batch_count > 10):
#                    break

        print ('************ phir - milenge ********')
                
     
#----------------------------------------------------------------------------------------------------------------------------------
"""
#    merged = tf.summary.merge([m_train.loss_summary,m_train.output_summary,m_train.summary_batchnorm_conv1_b,
#                               m_train.summary_batchnorm_conv2_b, m_train.summary_batchnorm_conv3_b, m_train.summary_lstm_input,
#                               m_train.summary_lstm_output,m_train.summary_decoder_input,m_train.summary_bicubic_output,
#                               m_train.summary_dec_batchnorm_conv1_b,m_train.summary_dec_batchnorm_conv2_b,
#                               m_train.summary_dec_batchnorm_conv3_b,m_train.summary_output,
#            m_test.loss_summary,m_test.output_summary,m_test.summary_batchnorm_conv1_b,m_test.summary_batchnorm_conv2_b,
#            m_test.summary_batchnorm_conv3_b, m_test.summary_lstm_input,m_test.summary_lstm_output,m_test.summary_decoder_input,
#            m_test.summary_bicubic_output,m_test.summary_dec_batchnorm_conv1_b,m_test.summary_dec_batchnorm_conv2_b,
#            m_test.summary_dec_batchnorm_conv3_b,m_test.summary_output])
#   
"""       
#----------------------------------------------------------------------------------------------------------------------------------
"""
def layers(self,row, col, seq_input, cell, state, tf_resize_tensor, is_training):

    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
            batchnorm_conv1_b = self.conv2D_batchnorm(input_ = seq_input, kernel_shape = [3,3,3*self.config.channels,16], 
                                                     kernel_name = "conv1_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv1",
                                                     batch_norm_name = "batch_norm_conv1",
                                                     is_training = is_training)
            
#                self.summary_batchnorm_conv1_b = tf.summary.histogram('batchnorm_conv1_b', batchnorm_conv1_b)
            
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
            batchnorm_conv2_b = self.conv2D_batchnorm(input_ = batchnorm_conv1_b, kernel_shape = [3,3,16,32], 
                                                     kernel_name = "conv2_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv2",
                                                     batch_norm_name = "batch_norm_conv2",
                                                     is_training = is_training)

#                self.summary_batchnorm_conv2_b = tf.summary.histogram('batchnorm_conv2_b', batchnorm_conv2_b)    
            
        with tf.variable_scope('conv_3',reuse=tf.AUTO_REUSE):
            batchnorm_conv3_b = self.conv2D_batchnorm(input_ = batchnorm_conv2_b, kernel_shape = [3,3,32,64], 
                                                     kernel_name = "conv3_filter", strides = [1,2,2,1], 
                                                     padding = "SAME",bias_name = "bias_conv3",
                                                     batch_norm_name = "batch_norm_conv3",
                                                     is_training = is_training)

#                self.summary_batchnorm_conv3_b = tf.summary.histogram('batchnorm_conv3_b', batchnorm_conv3_b)
            
#            lstm_input = tf.reshape(batchnorm_conv3_b, [self.config.batch_size,64,64,64])
        lstm_input = tf.reshape(batchnorm_conv3_b, [self.config.batch_size,-1])
        lstm_input = tf.identity(lstm_input, name="lstm_input")
            
#            self.summary_lstm_input = tf.summary.histogram('lstm_input', lstm_input)
        
    with tf.variable_scope('LSTM'):
        if not (row == 1 and col == 1): 
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(lstm_input, state)
        
#            self.summary_lstm_output = tf.summary.histogram('lstm_output', cell_output)
        
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        decoder_input = tf.reshape(cell_output,[self.config.batch_size,tf.cast(self.config.patch_size/8, tf.int32),tf.cast(self.config.patch_size/8, tf.int32),64])
        decoder_input = tf.identity(decoder_input,name = "decoder_input")
            
#            self.summary_decoder_input = tf.summary.histogram('decoder_input', decoder_input)
        
        with tf.variable_scope('bicubic_interpolate',reuse=tf.AUTO_REUSE):
            out_bicubic = tf.image.resize_bicubic(decoder_input,tf_resize_tensor,align_corners=False,name='op_bicubic_interp')
                
#            self.summary_bicubic_output = tf.summary.histogram('out_bicubic', out_bicubic)                
            
        with tf.variable_scope('dec_conv1_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv1_b = self.conv2D_batchnorm(input_ = out_bicubic, kernel_shape = [3,3,64,32], 
                                                     kernel_name = "dec_conv1_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv1",
                                                     batch_norm_name = "dec_batch_norm_conv1",
                                                     is_training = is_training)
            
#                self.summary_dec_batchnorm_conv1_b = tf.summary.histogram('dec_batchnorm_conv1_b', dec_batchnorm_conv1_b)
            
            
        with tf.variable_scope('dec_conv2_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv2_b = self.conv2D_batchnorm(input_ = dec_batchnorm_conv1_b, kernel_shape = [3,3,32,16], 
                                                     kernel_name = "dec_conv2_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv2",
                                                     batch_norm_name = "dec_batch_norm_conv2",
                                                     is_training = is_training)
            
#                self.summary_dec_batchnorm_conv2_b = tf.summary.histogram('dec_batchnorm_conv2_b', dec_batchnorm_conv2_b)
            
        with tf.variable_scope('dec_conv3_',reuse=tf.AUTO_REUSE):
            dec_batchnorm_conv3_b = self.conv2D_batchnorm(input_ = dec_batchnorm_conv2_b, kernel_shape = [3,3,16,3], 
                                                     kernel_name = "dec_conv3_filter", strides = [1,1,1,1], 
                                                     padding = "SAME",bias_name = "dec_bias_conv3",
                                                     batch_norm_name = "dec_batch_norm_conv3",
                                                     is_training = is_training)
            
#                self.summary_dec_batchnorm_conv3_b = tf.summary.histogram('dec_batchnorm_conv3_b', dec_batchnorm_conv3_b)

    with tf.variable_scope('out_level',reuse=tf.AUTO_REUSE):
        out_conv = tf.nn.relu(tf.nn.conv2d(
                        dec_batchnorm_conv3_b, self.variable_get([3,3,3,self.config.channels], 'out_conv_filter'),strides = [1,1,1,1], 
                        padding = 'SAME'))
        output = tf.nn.bias_add(out_conv, self.variable_get([self.config.channels],'out_conv_bias'))         
        
#            self.summary_output = tf.summary.histogram('final_output', output)
        
        return output, state
    
"""

"""
#
#lf_dim, lf_train_data = generate_lf_dataset.get_data_config(train_config)
#print ('-----lf_train_data', np.shape(lf_train_data))
#lf_train_data = lf_train_data[:,0:train_config.allowed_lf_dim*train_config.patch_size,
#                              0:train_config.allowed_lf_dim*train_config.patch_size,:]
#print ('lf_train_data', np.shape(lf_train_data))
#
#
#if not os.path.exists(train_config.save_path + "train_data"):
#    os.makedirs(train_config.save_path + "train_data")
#np.save(train_config.save_path + "train_data/lf_train_data.npy", lf_train_data)
#
#lf_dim, lf_test_data = generate_lf_dataset.get_data_config(test_config)
#print ('-----lf_test_data', np.shape(lf_test_data))
#lf_test_data = lf_test_data[:,0:test_config.allowed_lf_dim*test_config.patch_size,
#                            0:test_config.allowed_lf_dim*test_config.patch_size,:]
#print ('lf_test_data', np.shape(lf_test_data))
#
#if not os.path.exists(train_config.save_path + "test_data"):
#    os.makedirs(train_config.save_path + "test_data")
#np.save(train_config.save_path + "test_data/lf_test_data.npy", lf_test_data)


"""

"""

model_4_lr_0_01_test_loss = np.load("loss.npy")
model_4_lr_0_01_train_loss = np.load("loss_train.npy")


model_4_lr_0_01_test_loss = model_4_lr_0_01_test_loss[0:2000]
model_4_lr_0_01_train_loss = model_4_lr_0_01_train_loss[0:2000]

pylab.plot(model_4_lr_0_01_test_loss, '-b', label='model_4_lr_0_01_test_loss')
pylab.plot(model_4_lr_0_01_train_loss, '-g', label='train')
pylab.legend(loc='upper left')
pylab.show()

"""
"""
flags.DEFINE_string("dot", "",
                        "extension")
flags.DEFINE_integer("patch", 128,
                     "patch_size to be used")
flags.DEFINE_integer("channels", 1,
                     "number of channels")
flags.DEFINE_integer("allowed_lf_dim", 2,
                     "number of channels")
flags.DEFINE_integer("batch_size", 50,
                     "number of examples in one batch")
flags.DEFINE_integer("epochs", 7,
                     "number of channels")
flags.DEFINE_integer("num_files", 50,
                     "number of files to load data from")
flags.DEFINE_string("save_path", "/saver/checkpoint_saver/",
                        "save path for tensorflow data")
flags.DEFINE_float("learning_rate", 0.001,
                     "learning_rate")
flags.DEFINE_integer("gradient_clip", 5,
                     "gradient clip")
flags.DEFINE_integer("strides", 8,
                     "strides")
flags.DEFINE_integer("save_steps", 1000,
                     "checkpoint_save_steps")
flags.DEFINE_integer("summary_save_steps", 100,
                     "summary_save_steps")
flags.DEFINE_integer("save_output_steps", 1000,
                     "save test output in every number of these steps")
flags.DEFINE_string("model", "model_7",
                     "model_info")
flags.DEFINE_list("lr_list", [0.1,0.01,0.001,0.0001,0.00001],
                     "list of learning rates")
flags.DEFINE_float("dropout_rate", 1.0,
                     "dropout_rate")
FLAGS = flags.FLAGS
"""
"""
#
#train_path = FLAGS.dot + FLAGS.save_path_train_data + "train_data/"
#test_path = FLAGS.dot + FLAGS.save_path_test_data + "test_data/"
#
#lf_train_whole_data = generate_lf_dataset.load_data(train_config,train_path, dict_name = "train_filename_dict", start = 4, end = 9)
#lf_test_whole_data = generate_lf_dataset.load_data(train_config,train_path, dict_name = "train_filename_dict",start = 10, end = 12)
##lf_test_whole_data = generate_lf_dataset.load_data(test_config,test_path, dict_name = "test_filename_dict")
#
#print ("***************")
#save_processed_train_data_path = FLAGS.dot + FLAGS.save_path_processed_training_data
#save_processed_test_data_path = FLAGS.dot + FLAGS.save_path_processed_test_data
#
#try:
#    if not os.path.exists(save_processed_train_data_path):
#        os.makedirs(save_processed_train_data_path)    
#        lf_train_data_slice_128 = np.empty((0,train_config.patch_size*train_config.allowed_lf_dim,
#	                                  train_config.patch_size*train_config.allowed_lf_dim,train_config.channels), np.float64)
#        for i in range(8):
#            for j in range(8):
#                print (i,j)
#                patch = lf_train_whole_data[:,i*train_config.patch_size:(i+2)*train_config.patch_size,
#	                                    j*train_config.patch_size:(j+2)*train_config.patch_size,:]
#                lf_train_data_slice_128 = np.append(lf_train_data_slice_128,patch, axis = 0)
#                if (j == 7):
#                    np.save(save_processed_train_data_path + "processed_train_data_4-9" + np.str(train_config.patch_size) 
#	                    + ".npy",lf_train_data_slice_128)
#                    print ("processed data saved at {},{}".format(i,j))
#                    print ("shape of lf_train_data_slice_128 : {}".format(lf_train_data_slice_128.shape))
#    else:
#        lf_train_data_slice_128 = np.load(save_processed_train_data_path + "processed_train_data_4-9" +
#                                      np.str(train_config.patch_size) + ".npy")
#        
#except:
#    print ("lol..error..train")
#   
#print ("lf_train_data_slice_128 : {}".format(lf_train_data_slice_128.shape))
#lf_train_data = lf_train_data_slice_128
#
#try:
#    if not os.path.exists(save_processed_test_data_path):
#        os.makedirs(save_processed_test_data_path)    
#        lf_test_data_slice_128 = np.empty((0,test_config.patch_size*test_config.allowed_lf_dim,
#	                                  test_config.patch_size*test_config.allowed_lf_dim,test_config.channels), np.float64)
#        for i in range(8):
#            for j in range(8):
#                print (i,j)
#                patch = lf_test_whole_data[:,i*test_config.patch_size:(i+2)*test_config.patch_size,
#	                                    j*test_config.patch_size:(j+2)*test_config.patch_size,:]
#                lf_test_data_slice_128 = np.append(lf_test_data_slice_128,patch, axis = 0)
#                if (j == 7):
#                    np.save(save_processed_test_data_path + "processed_train_data_10-12" + np.str(test_config.patch_size) 
#	                    + ".npy",lf_test_data_slice_128)
#                    print ("processed test data saved at {},{}".format(i,j))
#                    print ("shape of lf_test_data_slice_128 : {}".format(lf_test_data_slice_128.shape))
#    else:
#        lf_test_data_slice_128 = np.load(save_processed_test_data_path + "processed_train_data_10-12" +
#                                      np.str(test_config.patch_size) + ".npy")
#except:
#    print ("lol..error..test")
#
#print ("lf_test_data_slice_128 : {}".format(lf_test_data_slice_128.shape))
#lf_test_data = lf_test_data_slice_128

"""

"""

#flags.DEFINE_string("save_path_processed_training_data", "/saver-500-4/processed_train_data_128_4-9/",
#                        "save path for training data only patches")
#flags.DEFINE_string("save_path_processed_test_data", "/saver-500-3/processed_train_data_128_10-12/",
#                        "save path for test data only patches")

"""
#[(A-Z)(a-z)]+_batchnorm_conv\d_[a-z]+
#[A-Za-z_]+_gradient_Model_encoder_[a-z0-9_]+


