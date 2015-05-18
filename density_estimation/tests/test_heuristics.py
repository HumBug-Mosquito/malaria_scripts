"""Add parent directory to path"""
import os,sys,inspect
currentdir_loc = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(currentdir_loc)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import numpy as np
from scipy.stats import gaussian_kde
from heuristics import mean_distance, mean_min_distance, scott, silverman

"""Define the required accuracy for the numerical tests,
(measured in number of matching decimal places)."""
ACCURACY = 7 

class TestDistances(unittest.TestCase):

	def test_mean_distance(self):
		points = np.array([[0,0],
						   [1,1],
						   [2,2]])
		expected_mean_dist = np.sqrt(2) * (4 / 3)
		mean_dist = mean_distance(points)
		self.assertAlmostEqual(expected_mean_dist, mean_dist, ACCURACY)

	def test_mean_min_distance(self):
		points = np.array([[0,0],
						   [1,1],
						   [2,2],
						   [3,3],
						   [4,4]])
		expected_mean_min_dist = np.sqrt(2)
		mean_min_dist = mean_min_distance(points)
		self.assertAlmostEqual(expected_mean_min_dist, mean_min_dist, ACCURACY)

	def test_calculate_scott(self):
		points = np.array([[0,0],
						   [2,1],
						   [3,2],
						   [4,3],
						   [5,4]])
		kernel = gaussian_kde(points.T, bw_method = 'scott')
		expected_factor = kernel.factor ** 2
		factor = scott(points)
		self.assertAlmostEqual(expected_factor, factor)

	def test_calulate_silverman(self):
		points = np.array([[0,0],
						   [2,1],
						   [3,2],
						   [4,3],
						   [5,4]])
		kernel = gaussian_kde(points.T, bw_method = 'silverman')
		expected_factor = kernel.factor ** 2
		factor = silverman(points)
		self.assertAlmostEqual(expected_factor, factor)


if __name__ == "__main__":
	unittest.main()