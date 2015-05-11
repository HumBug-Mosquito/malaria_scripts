"""Add parent directory to path"""
import os,sys,inspect
currentdir_loc = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(currentdir_loc)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import gdal
import numpy as np
from country_window_processor import import_rst_files, get_geotransform 
from country_window_processor import pixel_coordinates

def assert_dict_value_type_equal(dict1, dict2):
	"""returns true if sample values from each dictionary
	share the same type."""
	value1 = list(dict1.values())[0]
	value2 = list(dict2.values())[0]
	tc = unittest.TestCase('__init__')
	tc.assertEqual(type(value1), type(value2))


class TestCountryWindow(unittest.TestCase):

	def setUp(self):
		"""loads a small sample_data into memory to allow
		each of the processing methods to be tested."""
		self.sample_path = currentdir + '/sample_data/'
		self.sample_file = self.sample_path + '2000_02_ins_pt05deg.rst'
		self.sample_data = {
			'2000_02': gdal.Open(self.sample_file)
		}
		self.sample_geotransform = (-180.0, 0.05, 0.0, 90.0, 0.0, -0.05)
		self.sample_pixel_bbox = {
			'x': 3613,
			'y': 1550,
			'width': 65,
			'height': 128
		}

	def test_import_rst_files(self):
		"""check that a dict is created with the correct
		keys and with gdal objects as values."""
		data = import_rst_files(self.sample_path)
		keys = data.keys()
		expected_keys = self.sample_data.keys()
		self.assertEqual(expected_keys, keys)
		assert_dict_value_type_equal(self.sample_data, data)

	def test_get_geotransform(self):
		"""check that the correct geotransform is returned from 
		the raster data."""
		geotransform = get_geotransform(self.sample_data)
		self.assertEqual(self.sample_geotransform, geotransform)

	def test_pixel_coordinates(self):
		"""check that the bbox latitude/longitude coordinates are 
		correctly converted into pixel coordinates."""
		deg_bbox = {
			'top_left_lat': 12.5,
			'top_left_lon': 0.65,
			'width': 3.25, 
			'height': 6.4, 
		}
		pixel_bbox = pixel_coordinates(self.sample_geotransform, deg_bbox)
		self.assertEqual(self.sample_pixel_bbox, pixel_bbox)

	def tearDown(self):
		self.sample_data = None

if __name__ == '__main__':
	unittest.main()