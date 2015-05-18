import math
import numpy as np
import itertools
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from sklearn.cross_validation import KFold
from collections import defaultdict

def split_into_folds(data, num_folds):
	"""returns list of k-folds of shuffled data. Note that the 
	last fold will may contain a slightly different number of 
	data points if the size of data does not divide by num_folds.

	Parameters
	----------
	data : array
		An n x d array where n is the number of data points 
		and d is the dimension of the data.
	num_folds : int
		The number of folds into which the dataset 
		will be split.
	"""
	data_size = data.shape[0]
	fold_size = math.ceil(data_size / num_folds)
	indices = np.arange(data_size)
	np.random.shuffle(indices)
	folds = []
	for i in range(num_folds):
		if i < num_folds - 1:
			fold = data[indices[ i * fold_size: (i + 1) * fold_size], :]
		else: # ensures that the last fold includes all remaining data
			fold = data[indices[ i * fold_size:], :]
		folds.append(fold)
	return folds

def fold_combinations(folds):
	"""returns a list of dictionaries, each containing k-1 folds of
	training data and 1 fold of test data, formed by selecting
	different combinations from folds"""
	combinations = []
	fold_indices = range(len(folds))
	for i in fold_indices:
		split_data = {}
		training_folds = []
		for j in fold_indices:
			if i == j:
				test_fold = folds[j]
			else:
				training_folds.extend(folds[j])
		# split_data['test'] = test_data
		# split_data['train'] = np.vstack(tuple(training_folds))
		combinations.append([np.array(training_folds), test_fold])
	return combinations

def k_folds(data, num_folds):
	"""returns a list of dictionaries, each containing k-1 folds of
	training data and 1 fold of test data, formed by selecting
	different combinations from folds"""
	size = data.shape[0]
	kf = KFold(size, n_folds=num_folds)
	combinations = [[data[train, :], data[test, :]] for train, test in kf]
	return combinations

def log_likelihood(data, bandwidth):
	"""returns the likelihood of the test data having arisen
	a kernel trained on the training data with the given 
	bandwidth

	Parameters
	----------
	data : dictionary
		A dictionary with keys 'train' and 'test' containing arrays
		of the training and test data.
	bandwidth : float
		The bandwidth used to train the kernel.
	"""
	kernel = gaussian_kde(data['train'].T, bandwidth)
	probabilities = kernel.evaluate(data['test'].T)
	log_lhood = np.log(np.prod(probabilities))
	return log_lhood

def neg_log_likelilhood(data):
	"""returns the a negative log likelihood function which can be
	passed to the optimizer to find the optimal bandwidth."""
	def neg_log_llhood(bandwidth):
		neg = -1 * log_likelihood(data, bandwidth)
		return neg
	return neg_log_llhood

def cross_validated_bandwidths(data, num_folds):
	validated_bandwidths = []
	folds = split_into_folds(data, num_folds)
	combinations = k_folds(folds)

	for comb in combinations:
		neg_log_llhood = neg_log_likelilhood(comb)	
		optimal = minimize_scalar(neg_log_llhood, method='brent')
		validated_bandwidths.append(optimal['x'])
	return validated_bandwidths

def average_bandwith(data, num_runs=100, num_folds=5):
	"""returns the best bandwith (on average) computed over the number 
	of runs specified in num_runs using k-fold validation."""
	bandwidths = []
	for run in range(num_runs):
		bandwidths.extend(cross_validated_bandwidths(data, num_folds))
	average = sum(bandwidths) / len(bandwidths)
	return average