from scipy import sparse
import pandas as pd

def parse(path):
	g = open(path, 'rb')
	for l in g:
		yield eval(l)

def getDF(path):
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
		i += 1
	return pd.DataFrame.from_dict(df, orient='index').loc[:, ["title", "description", "price", "asin"]]

def Score(user_num, sim_set, rMatrix):
	sum_sim = 0
	sum_rate = 0
	ratings_of_user = rMatrix[:,user_num].toarray().T.tolist()
	for item_sim in sim_set:
		sum_sim += sim_set[item_sim]
		sum_rate += sim_set[item_sim] * ratings_of_user[0][item_sim]
	return sum_rate/sum_sim

def Probability(score):
	return 1/(1 + math.exp(-score))

def call_CV(simClass, mfClass, simItem_k, topUser_k, rMatrix_training, item_list, iteration1=10, iteration2=10, steplen_alpha=0.02, steplen_beta=2e-4, steplen_theta=-0.02):

    #### Initial parameters and class ####
    alpha, beta, theta = simClass.get_parameters()

    aggr_output_of_cv = {}
    #### parameter training iteration ####
    for i in range(iteration1):
        print("############################# %dth K-Cross Validation ###############################"%(i))

        #### Modify parameters for each iteration ####
        alpha = alpha + steplen_alpha
        beta = beta + steplen_beta
        theta = theta + steplen_theta
        simClass.change_parameters(alpha, beta, theta)

        start = 0
        end = int(rMatrix_training.shape[0]/iteration2)
        RMSE = []
        #### CV iteration ####
        for j in range(iteration2):
            print("---------- %dth fold of curent CV, %dth Iteration ----------"%(j,i))
            #print("%d-fold> "%j)
            #### Calculate the RMSE for this iteration ####		    
            RMSE.append(active_learning_process(simClass, mfClass, rMatrix_training, simItem_k, topUser_k, item_list, start, end))
            print("RMSE: %f"%RMSE[-1])
            tmp = end
            end = end + (end - start)
            start = tmp


        #### Caculate and record average RMESE for each iteration2 ####
        aggr_output_of_cv[i] = {'avg_RMSE': sum(RMSE)/len(RMSE), 'alpha': alpha, 'beta': beta, 'theta': theta}
        print(aggr_output_of_cv[i])

    #### Find best RMSE and best parameters####
    avg_rmse_lst = [aggr_output_of_cv[i]['avg_RMSE'] for i in aggr_output_of_cv]
    index = avg_rmse_lst.index(max(avg_rmse_lst))

    return aggr_output_of_cv, index
	
def active_learning_process(simClass, mfClass, rMatrix, simItem_k, topUser_k, item_list, start, end):

    #### Split training set and validation set ####	
    rating_martix_expanded = rMatrix.tolil()
    rating_martix_expanded[start:end,] = 0
    print("rating_martix_expanded_CV DONE")

    #### find k similar items for each new item ####
    newuser_asin = item_list[start:end]		
    print("test interval")
    print(start)
    print(end)
    sims = simClass.generate_topk_item_similarity(newuser_asin, simItem_k, item_list)

    #### construct new-item similarity dictionary ####
    '''
        sims_indexed: {
            'ITEM0001': {
                'ITEM0005': s1
                'ITEM0006': s2
                ...
            }
            'ITEM0002': {
                'ITEM0008': s3
                ...
            }
            ...
        }
    '''
    sims_indexed = {}
    for item in sims:
        sims_indexed[item] = {}
        for item_sim in sims[item]:
            sims_indexed[item][item_list.index(item_sim)] = sims[item][item_sim]
    print("sims_indexed DONE")

    #### Calculate Propability for each new item #### 
    for item in sims_indexed:    
        user_probability = {}
        sim_items_current = ()
        for item_sim in sims_indexed[item]:
            sim_items_current += (item_sim, )
        users_rated_sims = sparse.find(rMatrix[sim_items_current,:])[1]
        users_rated_sims = list(set(users_rated_sims))
        for userNum in users_rated_sims:
            user_probability[userNum] = Probability(Score(userNum, sims_indexed[item],rMatrix )) 
        user_probability = sorted(user_probability.items(), key=lambda d:d[1], reverse = True)

        ### when related users are less than k, randomly fill users into top k users###
        random_fill = True
        if random_fill == True:
            if(topUser_k > len(user_probability)):
                while(len(user_probability) != topUser_k):
                    filler = (int(rMatrix.shape[1] * random.random()), 0)
                    for user_possible in user_probability:
                        if filler[0] == user_possible[0]:
                            filler = (int(rMatrix.shape[1] * random.random()), 0)
                    user_probability.append(filler)

        for top in range(topUser_k):    
            rating_martix_expanded[item_list.index(item), user_probability[top][0]] = \
                rMatrix[item_list.index(item), user_probability[top][0]]
    print("item sims ALL DONE")

    ##### Caculate RMSE for each iteration2 #####
    prMatrix = mfClass.matrix_factorization(rating_martix_expanded)
    print("matrix factorization DONE")
    RMSE = mfClass.calculate_average_RMSE(rMatrix.toarray(), prMatrix, start, end)
    print("rmse DONE")
    return RMSE