from matrix_factorization import MatrixFactorization
import pandas as pd

def Score(users_rated_sims, users_rating, sim_set):    
    user_score = {}
    for userNum in users_rated_sims:         
        sum_rate = 0
        ratings_of_user = users_rating[userNum]
        for item_sim in sim_set:
            if item_sim in ratings_of_user:
                sum_rate += sim_set[item_sim] * ratings_of_user[item_sim]
        user_score[userNum] = sum_rate#/sum_sim
    return user_score

def call_CV(simClass, simItem_k, topUser_k, itemsFrame, ratingsFrame, users_rating,\
            iteration1=20, iteration2=3, \
            steplen_alpha=0.02, steplen_beta = 0, steplen_theta=-0.02):
    #### Initial parameters and class ####
    alpha,beta, theta = simClass.get_parameters()

    aggr_output_of_cv = {}
    #### parameter training iteration ####
    for i in range(iteration1):
        print("############################# %dth K-Cross Validation ###############################"%(i))

        #### Modify parameters for each iteration ####
        alpha = alpha + steplen_alpha
        beta = beta * steplen_beta
        theta = theta + steplen_theta
        simClass.change_parameters(alpha, beta, theta)

        start = 0
        end = int(itemsFrame.shape[0]/iteration2)
        RMSE = []
        #### CV iteration ####
        for j in range(iteration2):
            print("---------- %dth fold of curent CV, %dth Iteration ----------"%(j,i))
            #print("%d-fold> "%j)
            #### Calculate the RMSE for this iteration ####         
            RMSE.append(active_learning(simClass, start, end, simItem_k, topUser_k, itemsFrame, ratingsFrame, users_rating))
            start = end
            end = end + int(itemsFrame.shape[0]/iteration2)


        #### Caculate and record average RMSE for each iteration2 ####
        aggr_output_of_cv[i] = {'avg_RMSE': sum(RMSE)/len(RMSE), 'alpha': alpha,'beta':beta, 'theta': theta}
        print(aggr_output_of_cv[i])

    #### Find best RMSE and best parameters####
    avg_rmse_lst = [aggr_output_of_cv[i]['avg_RMSE'] for i in aggr_output_of_cv]
    index = avg_rmse_lst.index(min(avg_rmse_lst))

    return aggr_output_of_cv, index
	
def active_learning(simClass, start, end, simItem_k, topUser_k, itemsFrame, ratingsFrame, users_rating):
    #### Split training set and test set ####   
    ratings_train = ratingsFrame.to_records(index=False).tolist()
    ratings_test = []
    for ratingNum in range(ratingsFrame.shape[0]):
        if(ratings_train[ratingNum][1] == start):
            test_start = ratingNum
            while(ratingNum in range(len(ratings_train)) and ratings_train[ratingNum][1] <= end):
                ratings_test.append(ratings_train[ratingNum])
                ratingNum += 1
            test_end = ratingNum
            break
    ratings_train = ratings_train[0:test_start] + ratings_train[test_end:]
    print("pre-split DONE")
    
    #### find k similar items for each new item ####
    print("test interval: [" + str(start)  + ":" + str(end) + "]")
    sims = simClass.generate_topk_item_similarity(range(start,end), simItem_k, itemsFrame.loc[:,"asin"].tolist())
    print("Sims calculation DONE")
    
    #### Calculate Propability for each new item ####     
    ratings_set = ratingsFrame.loc[:,["userID","itemID"]].to_records(index=False).tolist()
    ratings_origin = ratingsFrame.to_records(index=False).tolist()
    item_groups = ratingsFrame.groupby("itemID")
            
    for item in sims:
        # first, find the set of users who have rated items within sim set.
        users_rated_sims = []
        for item_sim in sims[item]:
            users_rated_sims += item_groups.get_group(item_sim)["userID"].tolist()
        users_rated_sims = list(set(users_rated_sims))
        
        user_score = Score(users_rated_sims, users_rating, sims[item])
        user_score = sorted(user_score.items(), key=lambda d:d[1], reverse = True)
        
        for top in range(topUser_k):
            t = (user_score[top][0],item)
            if t in ratings_set:
                index = ratings_set.index(t)
                ratings_test.remove(ratings_origin[index])
                ratings_train.append(ratings_origin[index])
    print("active learning ALL DONE")

    ##### Caculate RMSE for each iteration2 #####
    mf = MatrixFactorization()
    return mf.matrix_factorization(ratings_train, ratings_test)
