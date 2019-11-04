# Portfolio Optimization using Deep Reinforcement Learning
### Bishishta Mukherjee, Konanki Sai Charan, Manvita Markala, Romil Rathi
### November 4, 2019

# Summary

Financial portfolio optimization is the process of redistributing funds into multiple financial vehicles at a given timestamp in order to maximize returns while minimizing the risk at the same time. The goal in this project is to provide a solution framework which deals with this complex financial engineering problem. The framework will be implemented using a combination of machine learning and Reinforcement Learning (RL) techniques. Built RL framework will be trained and tested on stocks and crypto currency trading data.

The dataset consists of:

(i)  Historical trading data for 15 stocks from S&P 500 portfolio<sup>1</sup> from 2005 to present

(ii) Discretized historical trading data for 6 Cryptocurrencies from CoinMarketCap<sup>2</sup> from 2015 to present

(ii) Continuous historical trading data for ~10 Cryptocurrencies from Poloniex Exchange<sup>3</sup>


The price data follows the format of Open, High, Low and Close (OHLC) for a given time frame. Open is the price at which the stock begins trading, High is the highest value it attains, Low is the lowest value throughout the day and Close is the closing value. For stocks data and discretized cryptocurrency data, the time frame considered is one trading day whereas for continuous cryptocurrency data, time frame will be about 30 minutes. Usually, open price is equal to close price for the previous day, but cryptocurrency follows high frequency trading, thus we might not see open price for one day same as closing price for previous days. 

To build a better portfolio optimizing agent than the one obtained in phase 1, where the solution framework was built using Convolutional Neural Network (CNN), another framework will be built using Long Short-Term Memory (LSTM). *We will be using goal oriented algorithms like deep Q-learning and Recurrent Reinforcement Learning when training the above mentioned neural network models and make them learn how to maximize the return profit over time.*


# Proposed plan of research

To scrape the financial price data, assets or holdings are selected in a way that they range across different stock sectors so as to minimize the downside risk. Some of the selected holdings are Apple Inc. from information technology sector, Boeing from industrial sector, State Street Corp. from financial sector and Facebook from communication sector. This diversification of portfolio is the most important factor in reducing the overall risk involved because when portfolio is diversified, if one asset value goes down, other might go up which rebalances it. In order to manage the sparsity in allocation of each asset in the portfolio, a small subset of financial price data is selected here.

Two major concerns have been handled during the data preprocessing step. First being data values for different stock and crypto currencies have varying ranges which might introduce bias in the models, to handle these fluctuating values, normalization was performed as follows: _close value /open value_. This ensures that the price of all the stocks fall within the same scale. Second being missing data values which were handled in two steps : 
1) Data pertaining to national holidays and weekends have been removed
2) Remaining missing values, if any, of open, low, high and close have been replaced with associated close values of previous timestamp.

To build the proposed framework, three different traditional deep learning methods [Convolution Neural Network (CNN), Recurrent Neural Network (RNN) and Long Short term Memory (LSTM)] will be used and tested against each other as the input of the neural network models is the unpredictable time series data, The assumption is that RNN might be more useful than CNN because of its bi-directional nature. But in the case of RNNs, vanishing gradient problem might arise and therefore LSTM should be used, which in general case is an improved model and shows better results. Hence, in order to experiment with the assumptions made and conclude which framework works the best in this scenario, performances of each of the three frameworks built will be evaluated using back testing and benchmarks.

The problem statement at a higher level is a RL problem where a given portfolio has certain number of assets and we are trying to allocate a certain ratio of total investment (weight) on each asset after each timestamp so as to find the optimized weight which will in turn maximize the reward by increasing our profits. The agent-environment setup is similar to the paper<sup>1</sup>. Here, the environment is the financial market, state at any timestamp t would be historical stock prices and previous timestamp’s portfolio. Action would correspond to a vector containing allocation information amongst each asset for the obtained portfolio and reward would be the profit return of investment after rebalancing the portfolio by allocation obtained in form of action.We would use the above proposed setup as a basis for the algorithms like deep Q-learning and Recurrent Reinforcement Learning to train the models.

Data science tools<sup>8</sup> used are pandas, tensorflow, beautifulsoup, scikit-learn, plot.ly, matplotlib, tableau, seaborn and anaconda.


# Preliminary results

In order to build a portfolio optimizer, 15 assets for stock data based on lesser correlation and 6 assets for cryptocurrencies based on trading volume were considered. A deep Reinforcement learning method was applied and a Convolution Neural Network model was built. The results from the model have been analysed and will be considered as a baseline for the second phase of the project.

On testing the performance of optimizing agent over stocks data, the final weights allocated to different assets can be seen in figure 1. The highest weight was assigned to Coca-Cola Co. followed by Verizon Communications Inc and Walmart Inc. The least weight was assigned Simon Property Group Inc followed by Xcel Energy Inc.

![wt_vectors_stocks[]{label="fig:wt_vectors_stocks"}](figures/test_result_plots/wt_vector_stocks.png)
###### Figure 1: Weights obtained for stocks portfolio

As seen in Figure 2 that represents the cumulative portfolio value across test steps, the value returned by the built optimising agent increased with each test step and performed better when compared to the values returned by the agent assigning equal weights amongst all the assets. 
The optimizing agent exhibited an increase of 25-30% in cumulative portfolio value over the initial portfolio value.


![cpv_stocks[]{label="fig:cpv_stocks"}](figures/test_result_plots/cpv_stocks.png)
###### Figure 2: Cumulative Portfolio Value for Stocks portfolio

The mean sharpe ratio obtained for the optimizing agent in case of stocks data is 0.735. As mentioned earlier since the sharpe ratio is less than 1, the results obtained in figure 2 is misleading and the trained agent gives us sub-optimal results for stocks data.

On testing the performance of optimizing agent over crypto data, the final weights allocated to different assets can be seen in figure 3. The highest weight was assigned to Litecoin followed by XRP(Ripple) and Ethereum. The least weight was assigned to Monero followed by NEM(XEM).

![wt_vectors_crypto[]{label="fig:wt_vectors_crypto"}](figures/test_result_plots/wt_vector_crypto.png)
###### Figure 3: Weights obtained for Crypto portfolio

As seen in Figure 4 that represents the cumulative portfolio value across test steps, the value returned by the built optimising agent increased with each test step and performed better when compared to the values returned by the agent assigning equal weights amongst all the assets. 
The optimizing agent exhibited an increase of 300-400% in cumulative portfolio value over the initial portfolio value. The equiweight agent also performed well here exhibiting an increase of around 200% in cumulative portfolio value over the initial portfolio value


![cpv_crypto[]{label="fig:cpv_crypto"}](figures/test_result_plots/cpv_crypto.png)
###### Figure 4: Cumulative Portfolio Value for Crypto portfolio

The mean sharpe ratio obtained for the optimizing agent in case of crypto currency data is 1.652. As mentioned earlier, since the sharpe ratio is greater than 1, the results obtained in figure 4 is acceptable and the trained agent gives us nearly optimal results for crypto currency data.


# References

[1] Historical stocks data of 15 assets from SP 500 portfolio from 2005 to present. https://www.barchart.com

[2] Historical data for 6 Cryptocurrencies from CoinMarketCap from 2015 to present. https://coinmarketcap.com

[3] 10 Cryptocurrency Data from Poloniex Exchange. https://poloniex.com/

[4] Chi Zhang, Corey Chen, Limian Zhang https://www-scf.usc.edu/~zhan527/post/cs599/

[5] Olivier Jin, Hamza El-Saawy Portfolio Management using Reinforcement Learning Dept. of Computer Science, Stanford, USA.

[6] Zhengyao Jiang, Dixing Xu, and Jinjun Liang A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.
Xi’an Jiaotong-Liverpool University, Suzhou, SU 215123, P. R. China.

[7] Code reference: https://github.com/selimamrouni/Deep-Portfolio-Management-Reinforcement-Learning
