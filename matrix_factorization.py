import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import find
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

class MatrixFactorization:
    def __init__(self, maxIter=15, regParam=0.01):
        self.maxIter = maxIter
        self.regParam = regParam

    def matrix_factorization(self, train_lst, test_lst):
        train_df = spark.createDataFrame(train_lst, ["userID", "itemID", "rating"])
        test_df = spark.createDataFrame(test_lst, ["userID", "itemID", "rating"])

        als = ALS(maxIter=15, regParam=0.01, userCol="userID", itemCol="itemID", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(train_df)
        print("matrix factorization DONE")
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))
        return rmse



        
