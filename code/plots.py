import matplotlib.pyplot as plt
import numpy as np

#function to plot cumulative portfolio values
def plot_cpv(final_pf_values_opt, final_pf_values_eq, initial_portfolio_value, plot_name):
	plt.plot(final_pf_values_opt, label = 'optimizing agent')
	plt.plot(final_pf_values_eq, label = 'equiweight agent')
	plt.plot([initial_portfolio_value] * len(final_pf_values_opt), label = 'no investment')
	plt.legend()
	plt.title('cumulative portfolio values over test steps')
	plt.xlabel('test steps')
	plt.ylabel('cumulative portfolio value')
	plt.savefig('figures/test_result_plots/' + plot_name)

#function to plot final wt vectors assigned
def plot_wts_assigned(wt_vector, num_stocks, ticker_list, plot_name):
	plt.bar(np.arange(num_stocks + 1), wt_vector)
	plt.title('Final Portfolio weights (test set)')
	plt.xticks(np.arange(num_stocks + 1), ['Cash'] + ticker_list, rotation=45)
	plt.xlabel('tickers')
	plt.savefig('figures/test_result_plots/' + plot_name)

