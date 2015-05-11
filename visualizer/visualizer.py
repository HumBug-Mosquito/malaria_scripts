"""visualizer provides methods to visualize the pixel array data
in the form of plots and animations."""

import os
import fiona
import numpy as np

"""Set up filesystem paths for data processing"""
BASE_PATH = '/Users/samuelalbanie/aims_course/project_one/Geography_Data/insolation'
"""Provide path to directory containing numpy files" of data"""
IMPORT_PATH = BASE_PATH + '/benin/'
"""Provide location of country shape files"""
SHAPE_PATH = '/Users/samuelalbanie/Downloads/BEN_adm/BEN_adm0.shp'
"""Provide the gdal geo matrix for the orignal raster data"""
GEO_MATRIX = (-180.0, 0.05, 0.0, 90.0, 0.0, -0.05)

def import_numpy_files(path=IMPORT_PATH):
	"""Returns a dictionary using dates as keys and
	numpy arrays of data as values."""
	data = {}
	fnames = next(os.walk(path))[2]
	fnames = [fname for fname in fnames if fname[-3:] == 'npy']
	for fname in fnames:
		data[fname[:7]] = np.load(path + fname)
	return data

def separate_coords(coords):
	"""returns two separate lists of x and y coordinates"""
	x_coords, y_coords = [], []
	for (x,y) in coords[0][0]:
		x_coords.append(x)
		y_coords.append(y)
	return (x_coords, y_coords)

def world2Pixel(geoMatrix, x_coords, y_coords):
    """ Uses a geomatrix to calculate the 
    pixel location of a geospatial coordinate"""
    upper_left_x, upper_left_y = geoMatrix[0], geoMatrix[3]
    xDist, yDist = geoMatrix[1], geoMatrix[5]
    rtnX, rtnY = geoMatrix[2], geoMatrix[4]
    pixel_x, pixel_y = [], []
    for x,y in zip(x_coords, y_coords):
        pixel_x.append(np.round((x - upper_left_x) / xDist).astype(np.int))
        pixel_y.append(np.round((upper_left_y - y) / xDist).astype(np.int))
    return (pixel_x, pixel_y)

def country_outline(path=SHAPE_PATH):
	"""returns points that trace out an outline of the country 
	shape file specified."""
	country_polygon =  fiona.open(path)[0]
	coords = country_polygon['geometry']['coordinates']
	(x_coords, y_coords) = separate_coords(coords)
	(line_x, line_y) = world2Pixel(GEO_MATRIX, x_coords, y_coords)
	return (line_x, line_y)