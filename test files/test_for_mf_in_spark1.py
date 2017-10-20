from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
sc = SparkContext()

# Load and parse the data
data = sc.textFile("D:/GitCode/item_based_collaborative_filtering/test files/test.data")
ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
'''
Parameters:	
ratings – RDD of Rating or (userID, productID, rating) tuple.
rank – Rank of the feature matrices computed (number of features).
iterations – Number of iterations of ALS. (default: 5)
lambda – Regularization parameter. (default: 0.01)
blocks – Number of blocks used to parallelize the computation. A value of -1 will use an auto-configured number of blocks. (default: -1)
nonnegative – A value of True will solve least-squares with nonnegativity constraints. (default: False)
seed – Random seed for initial matrix factorization model. A value of None will use system time as the seed. (default: None)
'''
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")