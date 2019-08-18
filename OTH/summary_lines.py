self.summary_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" + 'batchnorm_conv1_b', self.batchnorm_conv1_b)
self.summary_list.append(self.summary_batchnorm_conv1_b)



self.summary_batchnorm_conv2_b = tf.summary.histogram(self.name + "_" +'batchnorm_conv2_b', self.batchnorm_conv2_b)
self.summary_list.append(self.summary_batchnorm_conv2_b)

self.summary_lstm_input = tf.summary.histogram(self.name + "_" +'lstm_input', self.lstm_input)
self.summary_list.append(self.summary_lstm_input)


self.summary_lstm_output = tf.summary.histogram(self.name + "_" +'lstm_output', self.cell_output)
self.summary_list.append(self.summary_lstm_output)


self.summary_decoder_input = tf.summary.histogram(self.name + "_" +'decoder_input', self.decoder_input) 
self.summary_list.append(self.summary_decoder_input)


self.summary_out_bicubic = tf.summary.histogram(self.name + "_" +'out_bicubic', self.out_bicubic)  
self.summary_list.append(self.summary_out_bicubic)


self.summary_dec_batchnorm_conv1_b = tf.summary.histogram(self.name + "_" +'dec_batchnorm_conv1_b', 
                                                              self.dec_batchnorm_conv1_b)
self.summary_list.append(self.summary_dec_batchnorm_conv1_b)

self.output_layer_summary = tf.summary.histogram(self.name + "_" +'output_layers', self.output)   
self.summary_list.append(self.output_layer_summary)




