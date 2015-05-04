"""missing_data provides methods to replace missing values in a raster
dataset.  It assumes that the dataset consists of a single grid of pixel
values stored in a numpy array."""

import numpy as np

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