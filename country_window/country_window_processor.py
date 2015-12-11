"""country_window_processor provides methods to convert global raw raster data
in the Idrisi A.1 format into numpy arrays of pixel values for a bounding box 
surrounding the country of interest."""

import gdal, os
import numpy as np
from time import sleep

"""Set up filesystem paths for data processing"""
BASE_PATH = '/Users/samuelalbanie/aims_course/project_one/Geography_Data/insolation'
"""Provide path to directory containing raw .rst and .rdc files"""
IMPORT_PATH = BASE_PATH + '/uncompressed/'
"""Provide destination for processed data """
EXPORT_PATH = BASE_PATH + '/benin/'

"""Provide coordinates in lat/lon for a bounding box (also known as 
a window) for the country of interest (in this case Benin)."""
BENIN_BBOX = {
	'top_left_lat': 12.5,
	'top_left_lon': 0.65,
	'width': 3.25, 
	'height': 6.4, 
}

def import_rst_files(path=IMPORT_PATH):
	"""Assumes filenames take the form 'YYYY_MM_NAME.rst' 
	where NAME can be any value. Returns a dictionary 
	using dates as keys with gdal object values."""
	dataset = {}
	fnames = next(os.walk(path))[2]
	fnames = [fname for fname in fnames if fname[-4:] == ".rst"]
	for fname in fnames:
		dataset[fname[:7]] = gdal.Open(path + fname)
		sleep(0.1)
	return dataset

def import_tiff_files(path=IMPORT_PATH):
	"""Assumes filenames take the form 'NAME_YYYY.tif' 
	where NAME can be any value. Returns a dictionary 
	using dates as keys with gdal object values."""
	dataset = {}
	fnames = next(os.walk(path))[2]
	fnames = [fname for fname in fnames if fname[-4:] == ".tif"]
	for fname in fnames:
		dataset[fname[-8:-4]] = gdal.Open(path + fname)
		sleep(0.1)
	return dataset

def import_bil_files(path=IMPORT_PATH):
	"""Assumes filenames take the form 'NAMEYYag.bil' 
	where NAME can be any value. Returns a dictionary 
	using dates as keys with gdal object values."""
	dataset = {}
	fnames = next(os.walk(path))[2]
	fnames = [fname for fname in fnames if fname[-4:] == ".bil"]
	for fname in fnames:
		dataset[fname[5:7]] = gdal.Open(path + fname)
		sleep(0.1)
	return dataset

def get_geotransform(dataset):
	"""returns a tuple containing the geo matrix for
	the raster data."""
	gdal_instance = list(dataset.values())[0]
	geotransform = gdal_instance.GetGeoTransform()
	return geotransform

def pixel_coordinates(geotransform, deg_bbox=BENIN_BBOX):
    """returns the pixel coordinates for the bounding box."""
    origin_x, origin_y = geotransform[0], geotransform[3]
    pixel_width, pixel_height = geotransform[1], geotransform[5]
    x = int((deg_bbox['top_left_lon'] - origin_x) / pixel_width)
    y = int((deg_bbox['top_left_lat'] - origin_y) / pixel_height)
    width = int(deg_bbox['width'] / pixel_width)
    height = int(deg_bbox['height'] / pixel_width)
    pixel_bbox = {}
    pixel_bbox['x'], pixel_bbox['y'] = x, y
    pixel_bbox['width'], pixel_bbox['height'] = width, height
    return pixel_bbox

def export_numpy(pixel_data, path):
	"""Saves processed data as numpy files in the given 
	directory"""
	for key in pixel_data.keys():
		np.save(path + key + '_benin', pixel_data[key])


	
