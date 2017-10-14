from similarity import Similarity
from matrix_factorization import MatrixFactorization
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#################### test for Similarity ####################
# s = Similarity()
#### test data ####
# s.read_data("meta_Computers.json")
# print(s.data['price'])
# print(len(s.data['title']))
# plt.figure(1)
# plt.plot(np.array(s.data['price']).tolist())
# plt.show()
#### test for tf-idf ####
# data = [  
# 	'hello, world' 
# ]
# corpus = s.data_preprocess(data)
# res = s.cal_tf_idf(corpus)
# print(res)
#### test for dataframe ####

#################### test for MatrixFactorization ####################
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

M = MatrixFactorization(R, 2)
nP, nQ, m, e = M.matrix_factorization()
print("nP:\n", nP, "\n------------------")
print("nQ:\n", nQ, "\n------------------")
print("m:\n", m, "\n------------------")
print("e:\n", e, "\n------------------")


