"""Add parent directory to path"""
import os,sys,inspect
currentdir_loc = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(currentdir_loc)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import numpy as np
from cross_validation import split_into_folds
from cross_validation import fold_combinations, k_folds

def sample_fold(scalar, rows, cols):
	"""returns numpy array of dimension rows x cols
	in which every entry has the value of scalar"""
	return scalar * np.ones((rows, cols))

def assert_lists_of_ndarrays_equal(list1, list2):
	for item1, item2 in zip(list1, list2):
		np.testing.assert_array_equal(item1, item2)

class TestFolds(unittest.TestCase):

	def test_k_folds(self):
		"""check that fold_combinations returns a list of every
		possible training-test pairing formed by joining k-1 folds 
		into the training set and the remaining fold into the test 
		set."""
		fold1 = sample_fold(0,5,2)
		fold2 = sample_fold(1,5,2)
		fold3 = sample_fold(2,5,2)
		data = np.concatenate((fold1, fold2))
		data = np.concatenate((data, fold3))
		expected_combs = [
				[np.concatenate((fold2,fold3), axis=0), fold1],
				[np.concatenate((fold1,fold3), axis=0), fold2],
				[np.concatenate((fold1,fold2), axis=0), fold3]]
		combinations = k_folds(data, num_folds=3)
		for pairing1, pairing2 in zip(expected_combs, combinations):
			assert_lists_of_ndarrays_equal(pairing1, pairing2)

	def test_k_folds_sizes(self):
		"""check that k_folds returns training/test pairs of the 
		desired sizes"""
		data = np.zeros((23,2))
		expected_combs = [
				[np.zeros((18,2)), np.zeros((5,2))],
				[np.zeros((18,2)), np.zeros((5,2))],
				[np.zeros((18,2)), np.zeros((5,2))],
				[np.zeros((19,2)), np.zeros((4,2))],
				[np.zeros((19,2)), np.zeros((4,2))]]
		combinations = k_folds(data, num_folds=5)
		for pairing1, pairing2 in zip(expected_combs, combinations):
			assert_lists_of_ndarrays_equal(pairing1, pairing2)


if __name__ == "__main__":
	unittest.main()