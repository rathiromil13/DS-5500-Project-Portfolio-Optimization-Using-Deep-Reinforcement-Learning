import numpy as np 
import tensorflow as tf

#dataset information
data_path = 'crypto_data_input.npy'
ticker_list = ['BTC', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP']

data = np.load(data_path)
ohlc_features_num = data.shape[0]
ticker_num = data.shape[1]
trading_days_captured = data.shape[2]

print('number of ohlc features : ' + str(ohlc_features_num))
print('number of crypto currencies considered : ' + str(ticker_num))
print('number of trading days captured : ' + str(trading_days_captured))

#hyper parameters of the CNN network
num_filters_layer_1 = 2
num_filters_layer_2 = 20
kernel_size = (1, 3)

#train-validation-test split
train_data_ratio = 0.6
training_steps = 0.6 * trading_days_captured
validation_steps = 0.2 * trading_days_captured
test_steps = 0.2 * trading_days_captured

#hyper parameters for RL framework
training_batch_size = 50
beta_pvm = 5e-5  
num_trading_periods = 10

weight_vector_init = np.array(np.array([1] + [0] * ticker_num))
portfolio_value_init = 10000
weight_vector_init_test = np.array(np.array([1] + [0] * ticker_num))
portfolio_value_init_test = 10000
num_episodes = 1
num_batches = 1
equiweight_vector = np.array(np.array([1/(ticker_num + 1)] * (ticker_num + 1)))
#probability of exploitation in the RL framework (acting greedily)
epsilon = 0.8
#used while calculating the adjusted rewards
adjusted_rewards_alpha = 0.1

#hyper parameters for the optimizer
l2_reg_coef = 1e-8
adam_opt_alpha = 9e-2
optimizer = tf.train.AdamOptimizer(adam_opt_alpha)

#hyper parameters for trading 
trading_cost = 1/100000
interest_rate = 0.02/250
cash_bias_init = 0.7