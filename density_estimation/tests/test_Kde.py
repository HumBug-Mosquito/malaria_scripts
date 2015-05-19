"""Add parent directory to path"""
import os,sys,inspect
currentdir_loc = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(currentdir_loc)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import numpy as np
from Kde import Kde

"""Define the required accuracy for the numerical tests,
(measured in number of matching decimal places)."""
ACCURACY = 7 

class TestKde(unittest.TestCase):
	
	def test_normalizing_constant(self):
		"""check that normalizing_constant returns the correct
		value for a given bandwidth."""
		train_data = np.array([[1,0],[2,0],[3,0]])
		bandwidth = 0.25
		kde = Kde(train_data, bandwidth)
		constant = kde.normalizing_constant()
		expected_constant = 2.54647908
		self.assertAlmostEqual(expected_constant, constant, ACCURACY)

	def test_density(self):
		"""check that density calculates probabilities
		correctly."""
		train_data = np.array([[1,0],[2,0]])
		bandwidth = 0.5
		kde = Kde(train_data, bandwidth)
		test_point = np.array([[2,0]])
		probability = kde.density(test_point)
		expected_probability = 0.36138844478748799
		self.assertAlmostEqual(expected_probability, probability, ACCURACY)

	def test_log_likelihood(self):
		"""check that log_likelihood returns correct values."""
		train_data = np.array([[1,0],[2,0]])
		bandwidth = 0.5
		points = np.array([[3,0],[4,0]])
		kde = Kde(train_data, bandwidth)
		log_lhood = kde.log_likelihood(points)
		expected_log_lhood = -12.286938687661852
		self.assertAlmostEqual(expected_log_lhood, log_lhood)

if __name__ == "__main__":
	unittest.main()