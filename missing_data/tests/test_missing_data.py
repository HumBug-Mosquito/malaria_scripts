"""Add parent directory to path"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import numpy as np
from missing_data import find_pixel_neighbourhood

class TestMissingData(unittest.TestCase):

	def assert_dict_values_of_numpys_equal(self, dict1, dict2):
		"""compares the set of numpy arrays forming the values 
		in two dictionaries are the same (ignores dictionary keys)"""
		np.testing.assert_array_equal(dict1.values(), dict2.values())

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
		
if __name__ == "__main__":
	unittest.main()

