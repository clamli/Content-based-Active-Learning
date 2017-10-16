import numpy as np
import math
import matplotlib.pyplot as plt

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
        self.Q = np.random.rand(R.shape[0], self.K)         # item latent factor 
        print("MF iterations")
        error_list = []
        for step in range(self.steps):
            print(step)
            x = sparse.find(R)
            row = x[0]
            col = x[1]
            for i,j in zip(row, col):
                # eij_2 = pow(R[i,j] - np.dot(self.P[j, :], self.Q[i, :].T), 2) + (self.beta/2) * (sum(pow(self.P[j, :], 2)) + sum(pow(self.Q[i, :], 2)))
                eij = R[i,j] - np.dot(self.P[j, :], self.Q[i, :].T)
                self.P[j, :] = self.P[j, :] + self.alpha * (2 * eij * self.Q[i, :] - self.beta * self.P[j, :])
                self.Q[i, :] = self.Q[i, :] + self.alpha * (2 * eij * self.P[j, :] - self.beta * self.Q[i, :])
            pR = np.dot(self.Q, self.P.T)
            e = 0
            for i,j in zip(row,col):
                e = e + pow(R[i,j] - pR[i,j], 2)
            e = math.sqrt(e/len(row))
            error_list.append(e)
            if e < self.threshold:
                break

        #### Plot RMSE picture ####
        plt.figure(1) # 创建图表1
        plt.title('RMSE for each iteration')
        plt.xlabel('Iteration') 
        plt.ylabel('RMSE value')
        plt.plot(error_list)
        plt.show()

        return np.dot(self.Q, self.P.T)


    def approximate_matrix_factorization(self, R):
        self.P = np.random.rand(R.shape[1], self.K)         # user latent factor
        self.Q = np.random.rand(R.shape[0], self.K)         # item latent factor 
        print("MF iterations")
        error_list = []
        for step in range(self.steps):
            print(step)
            x = sparse.find(R)
            row = x[0]                 # item
            col = x[1]                 # user

            #### User Part ####
            denominatorV = 0
            for i in range(R.shape[0]):
                denominatorV = denominatorV + np.dot(self.Q[i, :], self.Q[i, :].T)
            numeratorVR = np.zeros((R.shape[1], self.K))
            for i,j in zip(row, col):
                numeratorVR[j] = numeratorVR[j] + self.Q[i, :]*R[i, j]
            self.P = numeratorVR / denominatorV

            #### Item Part ####
            denominatorU = 0
            for i in range(R.shape[1]):
                denominatorU = denominatorU + np.dot(self.P[i, :], self.P[i, :].T)
            numeratorUR = np.zeros((R.shape[0], self.K))
            for i,j in zip(row, col):
               numeratorUR[i] = numeratorUR[i] + self.P[j, :]*R[i, j]
            self.Q = numeratorUR / denominatorU

            #### Calculate Predicted Matrix ####
            pR = np.dot(self.Q, self.P.T)
            e = 0
            cnt = 0
            for i,j in zip(row,col):
                e = e + pow(R[i,j] - pR[i,j], 2)
                # for k in range(self.K):       # add regularization
                #     e = e + (self.beta/2) * (pow(self.P[j][k], 2) + pow(self.Q[i][k], 2))
                cnt = cnt + 1
            e = math.sqrt(e/cnt)
            error_list.append(e)

            #### Judge if small than threshold ###
            if e < self.threshold:
                break

        #### Plot RMSE picture ####
        plt.figure(1) # 创建图表1
        plt.title('RMSE for each iteration')
        plt.xlabel('Iteration') 
        plt.ylabel('RMSE value')
        plt.plot(error_list)
        plt.show()

        return np.dot(self.Q, self.P.T)


    def approximate_matrix_factorization_by_NMF(self, R):
        self.P = np.random.rand(R.shape[1], self.K)         # user latent factor
        self.Q = np.random.rand(R.shape[0], self.K)         # item latent factor 
        print("MF iterations")
        error_list = []
        for step in range(self.steps):
            print(step)
            x = sparse.find(R)
            row = x[0]                 # item
            col = x[1]                 # user

            #### User Part ####
            denominatorV = np.dot(np.dot(self.Q.T, self.Q), self.P.T).T + 1e-9   # u*k
            self.P = self.P * (R.T*self.Q) / denominatorV

            #### Item Part ####
            denominatorU = np.dot(np.dot(self.P.T, self.P), self.Q.T).T + 1e-9   # i*k
            self.Q = self.Q * (R*self.P) / denominatorU

            #### Calculate Predicted Matrix ####
            pR = np.dot(self.Q, self.P.T)
            e = 0
            cnt = 0
            for i,j in zip(row,col):
                e = e + pow(R[i,j] - pR[i,j], 2)
                # for k in range(self.K):       # add regularization
                #     e = e + (self.beta/2) * (pow(self.P[j][k], 2) + pow(self.Q[i][k], 2))
                cnt = cnt + 1
            e = math.sqrt(e/cnt)
            error_list.append(e)

            #### Judge if small than threshold ###
            if e < self.threshold:
                break

        #### Plot RMSE picture ####
        plt.figure(1) # 创建图表1
        plt.title('RMSE for each iteration')
        plt.xlabel('Iteration') 
        plt.ylabel('RMSE value')
        plt.plot(error_list)
        plt.show()

        return np.dot(self.Q, self.P.T)


    def calculate_average_RMSE(self, oRate, pRate, start, end):
        user_num = oRate.shape[1]
        e = 0
        cnt_of_rate = 0
        for i in range(start, end):
            for j in range(user_num):
                if oRate[i][j] > 0:
                    e = e + pow(oRate[i][j] - pRate[i][j], 2)
                    cnt_of_rate = cnt_of_rate + 1
        return math.sqrt(e/cnt_of_rate)
