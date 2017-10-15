import numpy as np

class MatrixFactorization:
    def __init__(self, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
        # self.R = R                                     # user-item matrix
        self.K = K                                     # feature number
        self.steps = steps                             # iterate time
        self.alpha = alpha                             # parameter1
        self.beta = beta                               # parameter2
        self.threshold = threshold                     # error threshold

    def matrix_factorization(self, R):
        self.P = np.random.rand(R.shape[1], self.K)         # user latent factor
        self.Q = np.random.rand(R.shape[0], self.K).T       # item latent factor 
        print("MF iterations")
        for step in range(self.steps):
            print(step)
            x = sparse.find(R)
            row = x[0]
            col = x[1]
#             print(row)
#             print(col)
            for i,j in zip(row, col):
#                 print(i)
#                 print(j)
                eij = R[i,j] - np.dot(self.P[j, :], self.Q[:, i])
                for k in range(self.K):
                    self.P[j][k] = self.P[j][k] + self.alpha * (2 * eij * self.Q[k][i] - self.beta * self.P[j][k])
                    self.Q[k][i] = self.Q[k][i] + self.alpha * (2 * eij * self.P[j][k] - self.beta * self.Q[k][i])
            pR = np.dot(self.P, self.Q).T
            e = 0
            for i,j in zip(row,col):
                e = e + pow(R[i,j] - pR[i,j], 2)
                for k in range(self.K):       # add regularization
                    e = e + (self.beta/2) * (pow(self.P[j][k], 2) + pow(self.Q[k][i], 2))
            if e < self.threshold:
                break
        self.Q = self.Q.T
        return np.dot(self.P, self.Q.T).T


    def calculate_average_RMSE(self, oRate, pRate, start, end):
        user_num = oRate.shape[1]
        e = 0
        cnt_of_rate = 0
        for i in range(start, end+1):
            for j in range(user_num):
                if oRate[i][j] > 0:
                    e = e + pow(oRate[i][j] - pRate[i][j])
                    cnt_of_rate = cnt_of_rate + 1
        return e/cnt_of_rate