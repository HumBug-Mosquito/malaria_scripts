import numpy as np
from scipy.spatial.distance import euclidean

def mean_distance(data):
	"""returns the mean pairwise distance between data points
	(useful as sanity check for the bandwidth)."""
	distances = []
	for i in range(data.shape[0]):
	    for j in range(data.shape[0]):
	        if i != j:
	            distances.append(euclidean(data[j,:].flatten(), data[i,:].flatten()))
	mean_dist = sum(distances) / len(distances)
	return mean_dist

def mean_min_distance(data):
	"""returns the mean distance to the next nearest point for 
	each point in data."""
	min_distances = []
	for i in range(data.shape[0]):
		distances = []
		for j in range(data.shape[0]):
			if i != j:
				distances.append(euclidean(data[j,:].flatten(), data[i,:].flatten()))
		min_distances.append(min(distances))
	mean_min_dist = sum(min_distances) / len(min_distances)
	return mean_min_dist

def scott(data):
	"""returns scotts rule of thumb value for the optimal
	bandwidth on the given data."""
	dim = data.shape[1]
	num_points = data.shape[0]
	scott_factor = num_points ** ( - 2 / (dim + 4))
	return scott_factor

def silverman(data):
	"""returns silverman rule of thumb value for the optimal
	bandwidth on the given data."""
	dim = data.shape[1]
	silverman_factor = scott(data) * ( 4 / (dim + 2)) ** (2 / (dim + 4))
	return silverman_factor