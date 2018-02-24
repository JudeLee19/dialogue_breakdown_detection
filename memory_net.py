import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
import numpy as np


class Memory_net():
    
    def __init__(self, obs_size, nb_hidden=128, num_class=3):
        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.num_class = num_class
        
        def __graph__():
            tf.reset_default_graph()
            
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
            init_state_c_, init_state_h_ = (tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2))
            system_features = tf.placeholder(tf.float32, [300], name='system_features')
            ground_label = tf.placeholder(tf.int32, name='ground_truth_action')

            # input projection
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden],
                                 initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden],
                                 initializer=tf.constant_initializer(0.))

            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi

            lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)

            lstm_op, state = lstm_f(inputs=projected_features, state=(init_state_c_, init_state_h_))

            # reshape LSTM's state tuple (2,128) -> (1,256)
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))
            
            # (256, 1)
            transposed_hidden_state = tf.transpose(state_reshaped)
            
            # output: 1 x 300 => 현재 시스템 메모리
            system_encoding = tf.expand_dims(system_features, 0)
            W_system = tf.get_variable('W_system', [300, 256], initializer=xav())
            
            current_system_attention_score = tf.matmul(tf.matmul(system_encoding, W_system), transposed_hidden_state)
            
            # 이전 시스템 메모리 값들
            prev_system_encodings = tf.placeholder(tf.float32, [None, 300])
            prev_system_attention_scores = tf.matmul(tf.matmul(prev_system_encodings, W_system), transposed_hidden_state)
            
            # output : [number of prev_utter + current_utter, 1]
            system_attention_scores = tf.concat([prev_system_attention_scores, current_system_attention_score], 0)
            transposed_system_attention_scores = tf.transpose(system_attention_scores)
            
            # [number of prev_utter + current_utter]
            system_attention_weights = tf.nn.softmax(transposed_system_attention_scores)
            
            # [number of prev_utter + current_utter, 300]
            system_encodings = tf.concat([prev_system_encodings, system_encoding], 0)
            
            weighted_system_encodings = tf.matmul(system_attention_weights, system_encodings)
            
            concatenated_features = tf.concat([state_reshaped, weighted_system_encodings], 1)

            # output projection
            Wo = tf.get_variable('Wo', [556, num_class],
                                 initializer=xav())
            bo = tf.get_variable('bo', [num_class],
                                 initializer=tf.constant_initializer(0.))
            # get logits
            logits = tf.matmul(concatenated_features, Wo) + bo

            probs = tf.squeeze(tf.nn.softmax(logits))

            # prediction
            prediction = tf.arg_max(probs, dimension=0)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ground_label)
            train_op = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

            self.loss = loss
            self.prediction = prediction
            self.probs = probs
            self.logits = logits
            self.state = state
            self.train_op = train_op

            # attach placeholders
            self.features_ = features_
            self.system_features = system_features
            self.init_state_c_ = init_state_c_
            self.init_state_h_ = init_state_h_
            self.ground_label = ground_label
            
            self.prev_system_encodings = prev_system_encodings
            self.system_encodings = system_encodings
            
        __graph__()
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.sess = sess

        # set init state to zeros
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)
        
        # 이전 시스템 값들.
        self.prev_system_encodings_ = np.zeros([1, 300], dtype=np.float32)

    def forward(self, user_embedding, system_embedding):
        probs, prediction, state_c, state_h, system_encodings = self.sess.run(
            [self.probs, self.prediction, self.state.c, self.state.h, self.system_encodings],
            feed_dict={
                self.features_: user_embedding.reshape([1, self.obs_size]),
                self.system_features: system_embedding,
                self.init_state_c_: self.init_state_c,
                self.init_state_h_: self.init_state_h,
                self.prev_system_encodings: self.prev_system_encodings_
            })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        self.prev_system_encodings_ = system_encodings
        
        return prediction
    
    def train_step(self, user_embedding, system_embedding, ground_label):
        _, loss_value, state_c, state_h, system_encodings = self.sess.run(
            [self.train_op, self.loss, self.state.c, self.state.h, self.system_encodings],
            feed_dict={
                self.features_: user_embedding.reshape([1, self.obs_size]),
                self.system_features : system_embedding,
                self.ground_label: [ground_label],
                self.init_state_c_: self.init_state_c,
                self.init_state_h_: self.init_state_h,
                self.prev_system_encodings: self.prev_system_encodings_
            })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        self.prev_system_encodings_ = system_encodings
        
        return loss_value
    
    def reset_state(self):
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.prev_system_encodings_ = np.zeros([1, 300], dtype=np.float32)