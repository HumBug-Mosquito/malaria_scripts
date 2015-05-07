"""missing_data provides methods to replace missing values in a raster
dataset.  It assumes that the dataset consists of a single grid of pixel
values stored in a numpy array."""

import numpy as np


"""Missing data is indicated by the value -999 in the IDRISI
Raster A.1 format."""
MISSING = -999

def missing_coordinates(data):
    """Returns an np array of coordinates where data is missing."""
    missing_mask = (data == MISSING)
    return np.where(missing_mask == True)

def missing_ratio(data):
    """Returns the ratio of the number of missing pixels to the 
    number of total pixels"""
    missing_mask = (data == MISSING)
    ratio =  (float(missing_mask.sum()) / data.size)
    return ratio

def find_pixel_neighbourhood(pixel_loc, pixel_grid_dim):
    """Returns a numpy array of pixels neigbouring the location 
    at 'pixel_loc' in a pixel grid with dimensions 'pixel_grid_dim'.
    If pixel_loc is on the boundary of the pixel grid, neighbours 
    that lie outside the boundary are not included."""
    neighbours = {}
    if pixel_loc[0] != 0:
        left_nbor = pixel_loc - np.array([1, 0]) # shift left
        neighbours['left_nbor'] = left_nbor
    if pixel_loc[0] != pixel_grid_dim[0] - 1:
        right_nbor = pixel_loc + np.array([1, 0]) # shift right
        neighbours['right_nbor'] = right_nbor
    if pixel_loc[1] != 0:
        up_nbor = pixel_loc - np.array([0, 1]) # shift up
        neighbours['up_nbor'] = up_nbor
    if pixel_loc[1] != pixel_grid_dim[1] - 1:
        down_nbor = pixel_loc + np.array([0, 1]) # shift down
        neighbours['down_nbor'] = down_nbor
    return neighbours

def remove_missing_neighbours(neighbours, data):
    """Removes any neighbours of pixel_loc that contain missing data"""
    data_neighbours = {}
    for key in neighbours.keys():
        loc = neighbours[key]
        if data[loc[0], loc[1]] != MISSING:
            data_neighbours[key] = neighbours[key]
    return data_neighbours

def pixel_average(pixel_loc, neighbours, data):
    """Calculates pixel_loc with the average of non-missing
    data in the neighbourhood. If no data is availabe amongst 
    the neighbours, the average land value is used."""
    data_neighbours = remove_missing_neighbours(neighbours, data)
    average = None
    if data_neighbours:
        average = neighbourhood_average(pixel_loc, data_neighbours, data)
    else:
        average = land_average(data)
    return average

def land_average(data):
    """returns the average value of all pixels on land"""
    land_data = data[np.where(data > MISSING)] 
    return np.average(land_data)

def neighbourhood_average(pixel_loc, neighbours, data):
    """returns the average of a pixels neighbours"""
    data_count = 0 
    data_sum = 0
    for loc in neighbours.values():
        data_count += 1
        data_sum += data[loc[0], loc[1]]
    average = int(data_sum / data_count)
    return average

def replace_missing_values(data):
    """Takes in a grid of pixels 'data' (as a numpy array) and finds the
    neighbourhoods of missing pixel data (which are represented by 
    a value of -999)."""
    pixel_grid_dim = data.shape
    missing_coords = missing_coordinates(data)
    original_data = np.copy(data) 
    for pixel_loc in zip(missing_coords[0], missing_coords[1]):
        neighbours = find_pixel_neighbourhood(pixel_loc, pixel_grid_dim)
        data[pixel_loc] = pixel_average(pixel_loc, neighbours, original_data)
    return data