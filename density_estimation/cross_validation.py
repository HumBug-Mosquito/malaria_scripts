import math
import numpy as np
import itertools
from Kde import Kde
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
	kde = Kde(pair[0], bandwidth)
	log_lhood = kde.log_likelihood(pair[1])
	return log_lhood

def neg_log_likelilhood(pair):
	"""returns the a negative log likelihood function object
	which can be passed to the optimizer to find the optimal 
	bandwidth.

	Parameters
	----------
	pair : list
		A list with two entries which contain the training
		and test data. 
	"""
	def neg_log_llhood(bandwidth):
		neg = -1 * log_likelihood(pair, bandwidth)		
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
	pairs = k_folds(data, num_folds)
	validated_bandwidths = []
	for pair in pairs:
		neg_log_llhood = neg_log_likelilhood(pair)	
		optimal = minimize_scalar(neg_log_llhood, bounds=(0.001, 10), method='bounded')
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
		np.random.shuffle(data)
		validated_bandwidths = cross_validated_bandwidths(data, num_folds)
		bandwidths.extend(validated_bandwidths)
	average = sum(bandwidths) / len(bandwidths)
	return (average, bandwidths)

def cross_validated_likelihoods(data, num_folds, bandwidth):
	"""returns a list of negative log likelihoods that are 
	calculated using cross validation.

	Parameters
	----------
	data : (n x d) ndarray 
		Array of n data points of dimension d.
	num_folds : int
		The number of folds used to perform the cross validation.
	bandwidth : float
		the bandwidth used to construct the kernel.
	"""
	pairs = k_folds(data, num_folds)
	validated_likelihoods = []
	for pair in pairs:
		neg_log_llhood = neg_log_likelilhood(pair)	
		neg_llhood = neg_log_llhood(bandwidth)
		validated_likelihoods.append(neg_llhood)
	return validated_likelihoods

def average_log_likelihood(data, bandwidth, num_runs=100, num_folds=5):
	"""returns the average negative log-likelihood of the data arising
	from KDE using the given bandwidth. The average is computed 
	over 'num_runs' runs of k-fold validation where k is specified 
	by num_folds. 

	Parameters
	----------
	data : (n x d) ndarray 
		Array of n data points of dimension d.
	bandwidth : float
		the bandwidth used to construct the kernel.
	num_runs : int
		The number of runs over which the average is calculated.
	num_folds : int
		The number of folds used to perform the cross validation.
	"""
	llikelihoods = []
	for run in range(num_runs):
		np.random.shuffle(data)
		llikelihoods.extend(cross_validated_likelihoods(data, num_folds, bandwidth))
	average = sum(llikelihoods) / len(llikelihoods)
	return average
