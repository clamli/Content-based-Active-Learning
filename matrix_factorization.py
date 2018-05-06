import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import find
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession


class MatrixFactorization:
    def __init__(self, maxIter=15, regParam=0.01, rank=10):
        self.maxIter = maxIter
        self.regParam = regParam
        self.spark = SparkSession \
            .builder \
            .master("local[*]") \
            .appName("Example") \
            .getOrCreate()

            # SparkSession.builder.master("local[*]").appName("Example").getOrCreate()

    def matrix_factorization(self, train_lst, test_lst):
        train_df = self.spark.createDataFrame(train_lst, ["userID", "itemID", "rating"])
        test_df = self.spark.createDataFrame(test_lst, ["userID", "itemID", "rating"])

        als = ALS(maxIter=15, regParam=0.01, userCol="userID", itemCol="itemID", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(train_df)
        print("matrix factorization DONE")
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))
        return rmse



        
