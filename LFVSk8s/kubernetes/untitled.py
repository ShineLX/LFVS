
          class Model(object):
          def __init__(self, is_training, config, name):
              self.name = name
              self.config = config
              self.output_list = []
              self.index_dict = {}
              self.loss = 0
              self.lr_list = lr_list
              self.loss_list = loss_list
          
              H = W = self.config.allowed_lf_dim*self.config.patch_size
          
              self.copy_lf_batch_holder = tf.placeholder(tf.float32, shape = (None,H,W,config.channels), name = name + 'data_placeholder')
      
              self.struct(self.copy_lf_batch_holder, is_training)
          
              self.loss_summary = tf.summary.scalar(name+"_loss", self.loss)
  
          
              if(not is_training):
                  return
          
              self.tvars = tf.trainable_variables()
              self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars, name = 'train_gradients'),
                                        self.config.gradient_clip, name = "clip_gradients_train")
      
  
          
              self.global_step=tf.train.get_or_create_global_step()      
              self.learning_rate = self.config.learning_rate
              self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name = "gradient_descent_train")
      
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      
              with tf.control_dependencies(update_ops):
              self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars),
                  global_step = self.global_step, name = "apply_gradients_train")

              print ('exporting_graph...')    

              print ('graph_exported')


