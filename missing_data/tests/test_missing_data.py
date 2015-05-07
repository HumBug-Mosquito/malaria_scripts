"""Add parent directory to path"""
import os,sys,inspect
currentdir_loc = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(currentdir_loc)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import numpy as np
from missing_data import find_pixel_neighbourhood, pixel_average 
from missing_data import missing_coordinates, replace_missing_values
from missing_data import missing_ratio

class TestMissingCoordinates(unittest.TestCase):

	def test_missing_coordinates(self):
		"""checks that the correct tuple of x_coords and y_coords 
		of missing data is being returned"""
		data = np.array([[3,4,5],
					     [4,-999,-999],
					     [2,-999,5]])
		expected_coordinates = (np.array([1, 1, 2]), np.array([1, 2, 1]))
		coords = missing_coordinates(data)
		np.testing.assert_array_equal(expected_coordinates, coords)

	def test_missing_data_ratio_with_one_missing_pixel(self):
		"""check that missing_data_ratio works correctly"""
		data = np.array([[3,4,5],
					     [4,-999,3],
					     [2,2,5]])
		expected_missing_ratio = 1. / 9
		ratio = missing_ratio(data)
		self.assertEqual(expected_missing_ratio, ratio)

	def test_missing_data_ratio_with_multiple_missing_pixels(self):
		"""check that missing_data_ratio works correctly"""
		data = np.array([[3,4,5],
					     [4,-999,-999],
					     [-999,2,5]])
		expected_missing_ratio = 1. / 3
		ratio = missing_ratio(data)
		self.assertEqual(expected_missing_ratio, ratio)

class TestMissingDataNeighbourhoods(unittest.TestCase):

	def assert_dict_values_of_numpys_equal(self, dict1, dict2):
		"""checks that the set of numpy arrays forming the values 
		in two dictionaries are the same ASSUMING THEY HAVE THE SAME KEYS"""
		for key in dict1.keys():
			np.testing.assert_array_equal(dict1[key], dict2[key])

	def test_find_pixel_neighbourhood_interior(self):
		"""check neighbourhood for pixels not on boundary"""
		pixel_grid_dim = np.array([400, 300])
		pixel_loc = np.array([200, 150])
		expected_neighbours = {
			'left_nbor': np.array([199, 150]),
			'right_nbor': np.array([201, 150]),
			'up_nbor': np.array([200, 149]),
			'down_nbor': np.array([200, 151])
		}
		neighbours = find_pixel_neighbourhood(pixel_loc, pixel_grid_dim)
		self.assert_dict_values_of_numpys_equal(expected_neighbours, neighbours)

	def test_find_pixel_neighbourhood_left_boundary(self):
		"""check neighbourhood for pixels on left boundary
		does not contain a left neighbour."""
		pixel_grid_dim = np.array([400, 300])
		pixel_loc = np.array([0, 150])
		expected_neighbours = {
			'right_nbor': np.array([1, 150]),
			'up_nbor': np.array([0, 149]),
			'down_nbor': np.array([0, 151])
		}
		neighbours = find_pixel_neighbourhood(pixel_loc, pixel_grid_dim)
		self.assertEqual(expected_neighbours.keys(), neighbours.keys())
		self.assert_dict_values_of_numpys_equal(expected_neighbours, neighbours)

	def test_find_pixel_neighbourhood_right_boundary(self):
		"""check neighbourhood for pixels on right boundary
		does not contain a right neighbour."""
		pixel_grid_dim = np.array([400, 300])
		pixel_loc = np.array([399, 150])
		expected_neighbours = {
			'left_nbor': np.array([398, 150]),
			'up_nbor': np.array([399, 149]),
			'down_nbor': np.array([399, 151])
		}
		neighbours = find_pixel_neighbourhood(pixel_loc, pixel_grid_dim)
		self.assertEqual(expected_neighbours.keys(), neighbours.keys())
		self.assert_dict_values_of_numpys_equal(expected_neighbours, neighbours)

	def test_find_pixel_neighbourhood_top_boundary(self):
		"""check neighbourhood for pixels on top boundary
		does not contain a top neighbour."""
		pixel_grid_dim = np.array([400, 300])
		pixel_loc = np.array([200, 0])
		expected_neighbours = {
			'left_nbor': np.array([199, 0]),
			'right_nbor': np.array([201, 0]),			
			'down_nbor': np.array([200, 1])
		}
		neighbours = find_pixel_neighbourhood(pixel_loc, pixel_grid_dim)
		self.assertEqual(expected_neighbours.keys(), neighbours.keys())
		self.assert_dict_values_of_numpys_equal(expected_neighbours, neighbours)

	def test_find_pixel_neighbourhood_bottom_boundary(self):
		"""check neighbourhood for pixels on bottom boundary
		does not contain a 'down' neighbour."""
		pixel_grid_dim = np.array([400, 300])
		pixel_loc = np.array([200, 299])
		expected_neighbours = {
			'left_nbor': np.array([199, 299]),
			'right_nbor': np.array([201, 299]),
			'up_nbor': np.array([200, 298]),
		}
		neighbours = find_pixel_neighbourhood(pixel_loc, pixel_grid_dim)
		self.assertEqual(expected_neighbours.keys(), neighbours.keys())
		self.assert_dict_values_of_numpys_equal(expected_neighbours, neighbours)

class TestDataAveraging(unittest.TestCase):

	def test_complete_pixel_average(self):
		"""check neighbourhood average is correct when no neighbours contain 
		missing data."""
		data = np.array([[3,4,5],
					     [4,-999,6],
					     [2,8,5]])
		pixel_loc = np.array([1,1])
		neighbours = {
			'left_nbor': np.array([0, 1]),
			'right_nbor': np.array([2, 1]),
			'up_nbor': np.array([1, 0]),
			'down_nbor': np.array([1, 1])
		}
		expected_average = int(5.5) # All values must be integers
		average = pixel_average(pixel_loc, neighbours, data)
		self.assertEqual(expected_average, average)

	def test_partial_pixel_average(self):
		"""check neighbourhood average is correct when some neighbours contain 
		missing data."""
		data = np.array([[3,4,5],
					     [4,-999,-999],
					     [2,-999,5]])
		pixel_loc = np.array([1,1])
		neighbours = {
			'left_nbor': np.array([0, 1]),
			'right_nbor': np.array([2, 1]),
			'up_nbor': np.array([1, 0]),
			'down_nbor': np.array([1, 1])
		}
		expected_average = 4
		average = pixel_average(pixel_loc, neighbours, data)
		self.assertEqual(expected_average, average)

	def test_partial_pixel_average(self):
		"""check neighbourhood average is correct when some neighbours contain 
		missing data."""
		data = np.array([[3,4,5],
					     [4,-999,-999],
					     [2,-999,5]])
		pixel_loc = np.array([1,1])
		neighbours = {
			'left_nbor': np.array([0, 1]),
			'right_nbor': np.array([2, 1]),
			'up_nbor': np.array([1, 0]),
			'down_nbor': np.array([1, 1])
		}
		expected_average = 4
		average = pixel_average(pixel_loc, neighbours, data)
		self.assertEqual(expected_average, average)

	def test_land_average(self):
		"""check land average is correct when all neighbours contain 
		missing data."""
		data = np.array([[1,-999,1],
					     [-999,-999,-999],
					     [1,-999,1]])
		pixel_loc = np.array([1,1])
		neighbours = {
			'left_nbor': np.array([0, 1]),
			'right_nbor': np.array([2, 1]),
			'up_nbor': np.array([1, 0]),
			'down_nbor': np.array([1, 1])
		}
		expected_average = 1
		average = pixel_average(pixel_loc, neighbours, data)
		self.assertEqual(expected_average, average)

class TestDataReplacement(unittest.TestCase):

	def test_single_missing_value_replacement(self):
		"""check neighbourhood average is correct when no neighbours contain 
		missing data."""
		missing_data = np.array([[3,4,5],
						         [4,-999,6],
						         [2,8,5]])
		expected_data = np.array([[3,4,5],
					     		  [4,5,6],
					     		  [2,8,5]])
		data = replace_missing_values(missing_data)
		np.testing.assert_array_equal(expected_data, data)

	def test_multiple_missing_value_replacement(self):
		"""check neighbourhood average is correct when no neighbours contain 
		missing data."""
		missing_data = np.array([[3,4,5],
						         [4,-999,-999],
						         [2,-999,5]])
		expected_data = np.array([[3,4,5],
					     		  [4,4,5],
					     		  [2,3,5]])
		data = replace_missing_values(missing_data)
		np.testing.assert_array_equal(expected_data, data)

		
if __name__ == "__main__":
	unittest.main()

