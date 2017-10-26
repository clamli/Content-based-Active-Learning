################################# 2017.10.14 #######################################
''' Step 0: Import Package '''
import pandas as pd
import numpy as np 
import math
import random
from subprocess import check_output
from scipy.sparse import coo_matrix
print(check_output(["dir", "."], shell=True).decode("utf8"))
import tool_function as tf
from similarity import Similarity
from matrix_factorization import MatrixFactorization
#### spark package ####
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
####################################################################################

####################################################################################
''' Step 1: Data Input '''
#### Load in ratings data & meta_item data ####
catagory = "Cell_Phones_and_Accessories"
item_data = './item_metadata/meta_' + catagory + '.csv'
rating_data = './user_ratings/' + catagory + '.csv'
ratingsFrame = pd.read_csv(rating_data, names = ["userID", "itemID", "rating"])
itemsFrame = pd.read_csv(item_data)
####################################################################################

####################################################################################
''' Step 2: Data Process '''
#### Fill out all the missing value ####
itemsFrame['description'].fillna("", inplace=True)
itemsFrame['title'].fillna("", inplace=True)
itemsFrame['price'].fillna(0, inplace=True)
####################################################################################

####################################################################################
''' Step 3: Construct Rating Dictionary '''
#construct rating dictionary to boost score calculation speed
user_groups = ratingsFrame.groupby("userID")
users_rating = {}
for name, group in user_groups:
    users_rating[name] = {}
    for index, row in group.loc[:,["itemID", "rating"]].iterrows():
        users_rating[name][row["itemID"]] = row["rating"]
print("Rating dict construction DONE")
####################################################################################

####################################################################################
''' Step 4: Split Dataset into Training and Test'''
#### Generate CV set ####	
ratingNum = 0
itemNum = 0
CV_end = int(itemsFrame.shape[0] * 0.7)
ratings = ratingsFrame.to_records(index=False).tolist()
while(ratings[ratingNum][1] <= CV_end):
    ratingNum += 1
ratingFrame_end = ratingNum
####################################################################################

####################################################################################
''' Step 5: Cross Validation '''
alpha = 1.0
beta = 1e-3
theta = -1.0
simClass = Similarity(alpha, beta, theta)
simClass.read_data(itemsFrame)

simItem_k = 6             # top K similar item to the new item
topUser_k = 15            # top K users to recommender the new item for ratings
steplen_alpha = 0
steplen_theta= 0
steplen_beta= 2
iteration1 = 20
iteration2 = 3
model, optim_ind = call_CV(simClass, simItem_k, topUser_k, \
                           itemsFrame[0 : CV_end], \
                           ratingsFrame[0 : ratingFrame_end], \
                           users_rating,\
                           iteration1, iteration2, \
                           steplen_alpha,steplen_beta, steplen_theta)  
####################################################################################

####################################################################################
''' Step 6: Use Optimum Parameters on Test Datast''' 
start = int(rating_martix_csr.shape[0] * 0.7)
end = rating_martix_csr.shape[0]
alpha = model[optim_ind]['alpha']
beta = model[optim_ind]['beta']
theta = model[optim_ind]['theta']
simClass.change_parameters(alpha, beta, theta)
tRMSE = active_learning_process(simClass, mfClass, rating_martix_csr, simItem_k, topUser_k, item_list, start, end)
####################################################################################

####################################################################################
''' Step 7: Plot Result of Baseline '''
pass
####################################################################################
