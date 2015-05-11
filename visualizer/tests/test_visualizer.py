"""Add parent directory to path"""
import os, sys,inspect
currentdir_loc = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(currentdir_loc)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import numpy as np
from visualizer import import_numpy_files


def assert_dict_value_type_equal(dict1, dict2):
	"""returns true if sample values from each dictionary
	share the same type."""
	value1 = list(dict1.values())[0]
	value2 = list(dict2.values())[0]
	tc = unittest.TestCase('__init__')
	tc.assertEqual(type(value1), type(value2))

class TestVisualizer(unittest.TestCase):

	def setUp(self):
		"""loads some numpy data into memory to allow
		visualization methods to be tested."""
		self.sample_path = currentdir + '/sample_data/'
		self.sample_file = self.sample_path + '2000_02_benin.npy'
		self.sample_data = {
			'2000_02': np.load(self.sample_file)
		}

	def test_import_numpy_files(self):
		"""check that a dict is created with the correct
		keys and with gdal objects as values."""
		data = import_numpy_files(path=self.sample_path)
		keys = data.keys()
		expected_keys = self.sample_data.keys()
		self.assertEqual(expected_keys, keys)
		assert_dict_value_type_equal(self.sample_data, data)

	def tearDown(self):
		self.sample_data = None

if __name__ == '__main__':
	unittest.main()