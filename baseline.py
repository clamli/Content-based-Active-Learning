################################# 2017.10.14 #######################################
import pandas as pd
import numpy as np 
import math
import random
from subprocess import check_output
from scipy.sparse import coo_matrix
print(check_output(["dir", "."], shell=True).decode("utf8"))
import tool_function

####################################################################################
''' Step 1: Data Input '''
#### Load in ratings data & meta_item data ####
ratingsFrame = pd.read_csv('./user_ratings/ratings_Computers.csv')
itemsFrame = getDF('./item_metadata/meta_Computers.json')
####################################################################################

####################################################################################
''' Step 2: Data Process '''
#### Fill out all the missing value ####
itemsFrame['description'].fillna("", inplace=True)
itemsFrame['title'].fillna("", inplace=True)
itemsFrame['price'].fillna(0, inplace=True)
#### drop items without any ratings ####
for index, row in itemsFrame.iterrows():
    if row["asin"] not in ratingsFrame.loc[:,"item"].tolist():
        itemsFrame.drop(index, inplace = True)
itemsFrame.reset_index(drop=True, inplace = True)  
itemsFrame = itemsFrame.sample(frac = 1, random_state = 1)
####################################################################################

####################################################################################
''' Step 3: Construct Item Dictionary 
	items: 
		dic {
			'B00001': {
				'User0001': r1
				'User0002': r2
				...
			}
			'B00002': {
				'User0003': r3
				'User0001': r4
			}
			...
		}
'''
items = {}  
for num in range(itemsFrame.shape[0]):
    items[itemsFrame.iloc[num]["asin"]] = {}
for num in range(ratingsFrame.shape[0]):
    if ratingsFrame.iloc[num]["item"] in items:  
        items[ratingsFrame.iloc[num]["item"]][ratingsFrame.iloc[num]["user"]] = ratingsFrame.iloc[num]["rating"]
####################################################################################

####################################################################################
''' Step 4: Construct User-Item Sparse Matrix '''
user_list = []
item_list = []
row = []
col = []
rating_data = []
itemNum = 0
for item in items:
    item_list.append(item)
    for user in items[item]:
        if user not in user_list:
            user_list.append(user)
        row.append(itemNum)
        col.append(user_list.index(user))
        rating_data.append(items[item][user])
    itemNum += 1
rating_martix_coo = coo_matrix((rating_data, (row, col)), shape=(itemsFrame.shape[0], len(user_list)))
rating_martix_csc = rating_martix_coo.tocsc()
rating_martix_csr = rating_martix_coo.tocsr()
####################################################################################

####################################################################################
''' Step 5: Split Dataset into Training and Test'''
start = 0
end = int(rating_martix_csr.shape[0] * 0.7)
rMatrix_training = rating_martix_csr[start:end,]
#rMatrix_test set
####################################################################################

####################################################################################
''' Step 6: Cross Validation '''
simItem_k = 6             # top K similar item to the new item
topUser_k = 15            # top K users to recommender the new item for ratings
K = 20					  # length of user profile and item profile when doing Matrix Factorization
alpha = 1.0
beta = 1e-3
theta = -1.0
simClass = Similarity(alpha, beta, theta)
simClass.read_data(itemsFrame)
mfClass = MatrixFactorization(K)
model, optim_ind = call_CV(simClass, mfClass, simItem_k, topUser_k, rMatrix_training, item_list[start:end])
####################################################################################

####################################################################################
''' Step 7: Use Optimum Parameters on Test Datast''' 
start = int(rating_martix_csr.shape[0] * 0.7)
end = rating_martix_csr.shape[0]
alpha = model[optim_ind]['alpha']
beta = model[optim_ind]['beta']
theta = model[optim_ind]['theta']
simClass.change_parameters(alpha, beta, theta)
tRMSE = active_learning_process(simClass, mfClass, rating_martix_csr, simItem_k, topUser_k, item_list, start, end)
####################################################################################

####################################################################################
''' Step 8: Plot Result of Baseline '''

####################################################################################
