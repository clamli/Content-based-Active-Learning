import numpy as np

class MatrixFactorization:
	def __init__(self, R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
		self.R = R                                     # user-item matrix
		self.K = K                                     # feature number
		self.steps = steps                             # iterate time
		self.alpha = alpha                             # parameter1
		self.beta = beta                               # parameter2
		self.threshold = threshold                     # error threshold
		self.P = np.random.rand(len(R), K)          # user latent factor
		self.Q = np.random.rand(len(R[0]), K)       # item latent factor


	def matrix_factorization(self):
		self.Q = self.Q.T
		for step in range(self.steps):
			for i in range(len(self.R)):
				for j in range(len(self.R[i])):
					if self.R[i][j] > 0:
						eij = self.R[i][j] - np.dot(self.P[i, :], self.Q[:, j])
						for k in range(self.K):
							self.P[i][k] = self.P[i][k] + self.alpha * (2 * eij * self.Q[k][j] - self.beta * self.P[i][k])
							self.Q[k][j] = self.Q[k][j] + self.alpha * (2 * eij * self.P[i][k] - self.beta * self.Q[k][j])
			pR = np.dot(self.P, self.Q)
			e = 0
			for i in range(len(self.R)):
				for j in range(len(self.R[i])):
					if self.R[i][j] > 0:
						e = e + pow(self.R[i][j] - pR[i][j], 2)
						for k in range(self.K):
							e = e + (self.beta/2) * (pow(self.P[i][k], 2) + pow(self.Q[k][j], 2))
			if e < self.threshold:
				break
		self.Q = self.Q.T
		return self.P, self.Q, np.dot(self.P, self.Q.T), e


	def calculate_RMSE(self, oRate, pRate):
		
