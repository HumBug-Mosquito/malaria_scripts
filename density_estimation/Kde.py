import numpy as np

class Kde:

	def __init__(self, train_data, bandwidth=1):
		self.train_data = train_data
		self.bandwidth = bandwidth

	def normalizing_constant(self):
		"""returns the normalizing constant associated 
		with each Gaussian kernel"""
		dim = self.train_data.shape[1]
		return 1 / (2 * np.pi * ((self.bandwidth) ** 2)) ** (dim / 2) 

	def density(self, x):
		"""returns the probability of the point x by constructing 
		Gaussian kernels of the bandwidth specifice in 'init()' 
		over the training points."""
		N = len(self.train_data)
		points = list(self.train_data)
		dists = [np.linalg.norm(x-point)**2 for point in points]
		exps = [np.exp(-dist / (2 * (self.bandwidth ** 2))) for dist in dists]
		unnormalized_sum = sum(exps)
		probability = (1 / N) * self.normalizing_constant() * unnormalized_sum
		return probability

	def log_likelihood(self, points):
		"""returns the log likelihood of 'points' arising from
		the distribution described by density()"""
		point_set = list(points)
		log_probabilities = [np.log(self.density(point)) for point in point_set]
		return sum(log_probabilities)