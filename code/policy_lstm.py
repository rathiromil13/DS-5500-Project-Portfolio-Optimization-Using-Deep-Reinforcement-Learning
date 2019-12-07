from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn

class PolicyLSTM(object):
    '''
    Using this class we will build policy for the LSTM network.
    '''

    def __init__(self, ohlc_feature_num, ticker_num, num_trading_periods, sess, optimizer, trading_cost, cash_bias_init, interest_rate, equiweight_vector, adjusted_rewards_alpha, num_filter_layer):

        # parameters
        self.ohlc_feature_num = ohlc_feature_num
        self.ticker_num = ticker_num
        self.num_trading_periods =  num_trading_periods
        self.trading_cost = trading_cost
        self.cash_bias_init = cash_bias_init
        self.interest_rate = interest_rate
        self.equiweight_vector = equiweight_vector
        self.adjusted_rewards_alpha = adjusted_rewards_alpha 
        self.optimizer = optimizer
        self.sess = sess
        self.num_filter_layer = num_filter_layer
        layers=2
        lstm_neurons = 20

        self.X_t = tf.placeholder(tf.float32, [None, self.ohlc_feature_num, self.ticker_num, self.num_trading_periods])
        self.weights_previous_t = tf.placeholder(tf.float32, [None, self.ticker_num + 1])
        self.pf_previous_t = tf.placeholder(tf.float32, [None, 1])
        self.daily_returns_t = tf.placeholder(tf.float32, [None, self.ticker_num]) 
        cash_bias = tf.get_variable('cash_bias', shape=[1, 1, 1, 1], initializer = tf.constant_initializer(self.cash_bias_init))
        shape_X_t = tf.shape(self.X_t)[0]
        self.cash_bias = tf.tile(cash_bias, tf.stack([shape_X_t, 1, 1, 1]))


        def lstm(X_t):
            network = tf.transpose(X_t, [0, 1, 3, 2])
            network = network / network[:, :, -1, 0, None, None]

            for layer_number in range(layers):
                resultlist = []
                reuse = False

                for i in range(self.ticker_num):
                    if i > 0:
                        reuse = True

                    result = tflearn.layers.lstm(X_t[:,:,:, i],
                                                    lstm_neurons,
                                                    dropout=0.3,
                                                    scope="lstm"+str(layer_number),
                                                    reuse=reuse)
            

                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, 1, self.ticker_num, lstm_neurons])
                # print('dhsegfhebgfhewf', network.shape)
            return network

        def policy_output(network, cash_bias):
            with tf.variable_scope("Convolution_Layer"):
                self.conv = tf.layers.conv2d(
                    inputs = network,
                    activation = tf.nn.relu,
                    filters = 1,
                    strides = (num_filter_layer + 1, 1),
                    kernel_size = (1, 1),
                    padding = 'same')

            with tf.variable_scope("Policy-Output"):
                tensor_squeeze = tf.squeeze(tf.concat([cash_bias, self.conv], axis=2), [1,3])
                self.action = tf.nn.softmax(tensor_squeeze)
            return self.action


        def reward(shape_X_t, action_chosen, interest_rate, weights_previous_t, pf_previous_t, daily_returns_t, trading_cost):
            #Calculating reward for current Portfolio
            with tf.variable_scope("Reward"):
                cash_return = tf.tile(tf.constant(1 + interest_rate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, daily_returns_t], axis=1)
                pf_vector_t = action_chosen * pf_previous_t
                pf_vector_previous = weights_previous_t * pf_previous_t

                total_trading_cost = trading_cost * tf.norm(pf_vector_t - pf_vector_previous, ord=1, axis=1) * tf.constant(1.0, shape=[1])
                total_trading_cost = tf.expand_dims(total_trading_cost, 1)

                zero_vector = tf.tile(tf.constant(np.array([0.0] * ticker_num).reshape(1, ticker_num), shape=[1, ticker_num], dtype=tf.float32), tf.stack([shape_X_t, 1]))
                cost_vector = tf.concat([total_trading_cost, zero_vector], axis=1)

                pf_vector_second_t = pf_vector_t - cost_vector
                final_pf_vector_t = tf.multiply(pf_vector_second_t, y_t)
                portfolio_value = tf.norm(final_pf_vector_t, ord=1)
                self.instantaneous_reward = (portfolio_value - pf_previous_t) / pf_previous_t
                
            #Calculating Reward for Equiweight portfolio
            with tf.variable_scope("Reward-Equiweighted"):
                cash_return = tf.tile(tf.constant(1 + interest_rate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, daily_returns_t], axis=1)
  
                pf_vector_eq = self.equiweight_vector * pf_previous_t
        
                portfolio_value_eq = tf.norm(tf.multiply(pf_vector_eq, y_t), ord=1)
                self.instantaneous_reward_eq = (portfolio_value_eq - pf_previous_t) / pf_previous_t

            #Calculating Adjusted Rewards
            with tf.variable_scope("Reward-adjusted"):
                self.adjusted_reward = self.instantaneous_reward - self.instantaneous_reward_eq - self.adjusted_rewards_alpha * tf.reduce_max(action_chosen)
                
            return self.adjusted_reward


        self.lstm_layer = lstm(self.X_t)
        self.action_chosen = policy_output(self.lstm_layer, self.cash_bias)
        self.adjusted_reward = reward(shape_X_t, self.action_chosen, self.interest_rate, self.weights_previous_t, self.pf_previous_t, self.daily_returns_t, self.trading_cost)
        self.train_op = optimizer.minimize(-self.adjusted_reward)

    def compute_weights(self, X_t_, weights_previous_t_):
        # tf.print(self.action_chosen)
        return self.sess.run(tf.squeeze(self.action_chosen), feed_dict={self.X_t: X_t_, self.weights_previous_t: weights_previous_t_})

    def train_lstm(self, X_t_, weights_previous_t_, pf_previous_t_, daily_returns_t_):
        """
        training the neural network
        """
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.weights_previous_t: weights_previous_t_,
                                                self.pf_previous_t: pf_previous_t_,
                                                self.daily_returns_t: daily_returns_t_})
