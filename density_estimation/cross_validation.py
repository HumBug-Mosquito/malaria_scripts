import math
import numpy as np
import itertools
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from sklearn.cross_validation import KFold
from collections import defaultdict

def k_folds(data, num_folds):
	"""returns a list of training-test pairs. Each pair 
	contains k-1 folds of training data and 1 fold of 
	test data, formed by selecting different combinations 
	from folds

	Parameters
	----------
	data : (n x d) ndarray 
		Array of n data points of dimension d.
	num_folds : int
		The number of folds to split the data into.

	"""
	size = data.shape[0]
	kf = KFold(size, n_folds=num_folds)
	combinations = [[data[train, :], data[test, :]] for train, test in kf]
	return combinations

def log_likelihood(pair, bandwidth):
	"""returns the likelihood of the test data having arisen
	a kernel trained on the training data with the given 
	bandwidth

	Parameters
	----------
	pair : list
		A list with two entries which contain the training
		and test data. 
	bandwidth : float
		This bandwidth is a scalar which is multiplied by the
		identity matrix to form a bandwidth matrix.  This 
		matrix is then used to perform density estimation.
	"""
	kernel = gaussian_kde(pair[0].T, bandwidth)
	probabilities = kernel.evaluate(pair[1].T)
	log_lhood = np.log(np.prod(probabilities))
	return log_lhood

def neg_log_likelilhood(data):
	"""returns the a negative log likelihood function object
	which can be passed to the optimizer to find the optimal 
	bandwidth.

	Parameters
	----------
	data : (n x d) ndarray 
		Array of n data points of dimension d
	"""

	def neg_log_llhood(bandwidth):
		neg = -1 * log_likelihood(data, bandwidth)
		return neg
	return neg_log_llhood

def cross_validated_bandwidths(data, num_folds):
	"""returns a list of optimal bandwidths that are calculated
	using cross validation.

	Parameters
	----------
	data : (n x d) ndarray 
		Array of n data points of dimension d.
	num_folds : int
		The number of folds used to perform the cross validation.
	"""
	combinations = k_folds(data, num_folds)
	validated_bandwidths = []
	for pair in combinations:
		neg_log_llhood = neg_log_likelilhood(pair)	
		optimal = minimize_scalar(neg_log_llhood, method='brent')
		validated_bandwidths.append(optimal['x'])
	return validated_bandwidths

def average_bandwith(data, num_runs=100, num_folds=5):
	"""returns the optimal bandwith (on average) computed over the number 
	of runs specified in num_runs using k-fold validation.

	Parameters
	----------
	data : (n x d) ndarray 
		Array of n data points of dimension d.
	num_runs : int
		The number of runs over which the average is calculated.
	num_folds : int
		The number of folds used to perform the cross validation.
	"""
	bandwidths = []
	for run in range(num_runs):
		bandwidths.extend(cross_validated_bandwidths(data, num_folds))
	average = sum(bandwidths) / len(bandwidths)
	return average