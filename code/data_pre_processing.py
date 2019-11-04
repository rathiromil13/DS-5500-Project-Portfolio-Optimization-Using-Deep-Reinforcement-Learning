import numpy as np
import pandas as pd


#Number of days stocks are traded in a year
nbr_trading_days = 252

#reading stock data
stocks_data_file_path = 'stocks_data.csv'
stocks_data = pd.read_csv(stocks_data_file_path)
stocks_data  = stocks_data.sort_values(['ticker', 'Date'])
print('stocks considered : ' + str(np.unique(stocks_data['ticker'])))

#Calculating Mean Returns and Covariance Matrix
pivot_table = pd.pivot_table(stocks_data, values='Close', index=['Date'],columns=['ticker'])
returns = pivot_table.pct_change().dropna()
stocks_returns_mean = returns.mean()
stocks_covariance_matrix = returns.cov()


#reading crypto data
crypto_data_file_path = 'crypto_data.csv'
crypto_data = pd.read_csv(crypto_data_file_path)
crypto_data  = crypto_data.sort_values(['crypto_ticker', 'Date'])
print('crypto currenicies considered : ' + str(np.unique(crypto_data['crypto_ticker'])))

smallest_ticker_data_points = 100000000
for ticker in np.unique(crypto_data['crypto_ticker']):
    if len(crypto_data['Date'].loc[crypto_data['crypto_ticker']==ticker]) < smallest_ticker_data_points:
        smallest_ticker_data_points = len(crypto_data['Date'].loc[crypto_data['crypto_ticker']==ticker])
        smallest_date = crypto_data['Date'].loc[crypto_data['crypto_ticker']==ticker].min()
    
crypto_data_filtered = crypto_data.loc[crypto_data['Date'] >= smallest_date]
crypto_data_filtered  = crypto_data_filtered.sort_values(['crypto_ticker', 'Date'])

pivot_table = pd.pivot_table(crypto_data_filtered, values='Close', index=['Date'],columns=['crypto_ticker'])
returns_crypto = pivot_table.pct_change().dropna()
crypto_returns_mean = returns_crypto.mean()
crypto_covariance_matrix = returns_crypto.cov()


def get_input_data(data_table, ticker_column_name):
    open_values = list()
    close_values = list()
    high_values = list()
    low_values = list()

    for ticker in np.unique(data_table[ticker_column_name]):
        open_value_list = data_table['Open'].loc[data_table[ticker_column_name]==ticker]
        open_values.append(open_value_list[1:].reset_index(drop = True) / open_value_list[:-1].reset_index(drop = True))
        close_values.append(data_table['Close'].loc[data_table[ticker_column_name]==ticker][:-1] / open_value_list[:-1])
        high_values.append(data_table['High'].loc[data_table[ticker_column_name]==ticker][:-1] / open_value_list[:-1])
        low_values.append(data_table['Low'].loc[data_table[ticker_column_name]==ticker][:-1] / open_value_list[:-1])
    
    return np.array([close_values,
                     high_values,
                     low_values,
                     open_values])


stocks_input_array = get_input_data(stocks_data, 'ticker')
np.save('stocks_data_input.npy', stocks_input_array)
print('shape of stocks data input : ' + str(stocks_input_array.shape))

crypto_input_array = get_input_data(crypto_data_filtered, 'crypto_ticker')
np.save('crypto_data_input.npy', crypto_input_array)
print('shape of crypto data input : ' + str(crypto_input_array.shape))