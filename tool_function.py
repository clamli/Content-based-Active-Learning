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

def call_CV(simItem_k, topUser_k, rMatrix_training, item_list, iteration1=10, iteration2=10):
	


	for # iteration1 #

		#### Modify parameters for each iteration ####


		for # iteration2 #

		

			#### Split training set and validation set ####
			start = 0
			end = int(len(item_list)/10)
		    rating_martix_lil_CV = rMatrix_training.tolil()
		    rating_martix_lil_CV[start:end,] = 0
		    print("rating_martix_lil_CV DONE")





		    #### find k similar items for each new item ####
		    s = Similarity()
		    s.read_data(itemsFrame)
		    sims = s.generate_topk_item_similarity(itemsFrame.iloc[start:end,:].loc[:,"asin"].tolist(), simItem_k)
		    
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
				for userNum in range(rMatrix_training.shape[1]):
					user_probability[userNum] = Probability(Score(userNum, sims_indexed[item]))
				user_probability = sorted(user_probability.items(), key=lambda d:d[1], reverse = True)
				for top in range(topUser_k):    
					rating_martix_lil_CV[item_list.index(item), user_probability[top][0]] = \
						rMatrix_training[item_list.index(item), user_probability[top][0]]


		    ##### Caculate RMESE for each iteration2 #####



		#### Caculate and record average RMESE for each iteration2 ####



	#### Find best RMSE and best parameters####


    return # Best parameters, Best RMSE #
