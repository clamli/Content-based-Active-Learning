import numpy as np
import nltk
import math
import string
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Similarity:
	def __init__(self, alpha=1.0, beta=1e-3, theta=-1.0):
		self.stemmer = PorterStemmer()
		self.alpha = alpha
		self.beta = beta
		self.theta = theta
		self.item_num = 0
		
	def change_parameters(alpha, beta, theta):
		'''change parameter when doing cross validation'''
		self.alpha = alpha
		self.beta = beta
		self.theta = theta

	def read_data(self, dataframe):
		'''read in the data and fill the NaN'''
		# self.data = dataframe
		# self.generate_item_vector()
		# print(self.data.head(3))
		self.data = dataframe
		# self.data['description'].fillna("", inplace=True)
		# self.data['title'].fillna("", inplace=True)
		# self.data['price'].fillna(0, inplace=True)
		self.generate_item_vector()

	def stem_tokens(self, tokens, stemmer):
		'''stemming'''
		stemmed = []
		for item in tokens:
			stemmed.append(stemmer.stem(item))
		return stemmed

	def tokenize(self, text):
		'''do stemming for each word'''
		tokens = nltk.word_tokenize(text)
		stems = self.stem_tokens(tokens, self.stemmer)
		return stems

	def data_preprocess(self, corpus):
		''' stop-word filtering and stemming
			input: every element represent a title/description of an item (after preprocessing)
			corpus = [  
	    		'This is the first document.',  
	    		'This is the second second document.',  
	    		'And the third one.',  
	    		'Is this the first document?',  
			]
		'''
		token_lst = []
		for ele in corpus:
			lowers = ele.lower()
			#### delete punctuation in string ####
			table = str.maketrans(dict.fromkeys(string.punctuation))
			no_punctuation = lowers.translate(table)
			token_lst.append(no_punctuation)
		return token_lst


	def cal_tf_idf(self, corpus):
		''' 
			output: tf-idf value vectors of each element
			tf-idf = [[ 0.          0.43877674  0.54197657  0.43877674  0.          0.			0.35872874  0.          0.43877674]
		 			  [ 0.          0.27230147  0.          0.27230147  0.          0.85322574	0.22262429  0.          0.27230147]
		 			  [ 0.55280532  0.          0.          0.          0.55280532  0.			0.28847675  0.55280532  0.        ]
		 			  [ 0.          0.43877674  0.54197657  0.43877674  0.          0.			0.35872874  0.          0.43877674]]
		'''
		# print self.token_lst
		tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
		tfs = tfidf.fit_transform(corpus)
		return tfs

	def generate_item_vector(self):
		''' Generate item vector for training data
			item vector: <et, ed, price>
			et: <tf-idf1, tf-idf2, ..., tf-idfk>
			ed: <tf-idf1, tf-idf2, ..., tf-idfk>
			price: normalized price
			self.item_vector_set:
				dict {
					'0001': {
						'title': []
						'description': []
						'price': float
					}	
					'0002': {
						...
					}
					...
				}
		'''
		#### original data ####
		price_list = np.array(self.data['price']).tolist()
		number_list = np.array(self.data['asin']).tolist()
		#### tf-idf result ####
		self.title_result = self.cal_tf_idf(self.data_preprocess(np.array(self.data['title']).tolist()))
		'''
		self.title_result:
				v1 v2 v3 ... vn
			v1
			v2
			v3
			...
			vn 
		'''
		self.title_result = cosine_similarity(self.title_result, self.title_result)
		self.description_result = self.cal_tf_idf(self.data_preprocess(np.array(self.data['description']).tolist()))
		'''
		self.description_result:
				v1 v2 v3 ... vn
			v1
			v2
			v3
			...
			vn 
		'''
		self.description_result = cosine_similarity(self.description_result, self.description_result)
		#### price result ####
		'''self.price_result: [p1, p2, p3, ... , pn]'''
		self.price_result = [(2*(1.0 / (1+math.exp(-self.beta*(price_list[i]))))-1) for i in range(len(price_list))]

		self.item_num = len(price_list)
		self.item_vector_set = {n:{'title':t, 'description':d, 'price':p} for n,t,d,p in zip(number_list, self.title_result, self.description_result, self.price_result)}

	# def generate_item_vector_for_newitem(self, new_item):
	# 	''' Generate item vector for new item
	# 		Input:
	# 			new_item: dataframe
	# 		Output:
	# 			newitem_title_result[0]: list [v1, v2, v3, ....]
	# 			newitem_description_result[0]: list [v1, v2, v3, ....]
	# 			price_result: float
	# 	'''
	# 	#### original data for new item ####
	# 	newitem_title_list = new_item['title']
	# 	newitem_description_list = new_item['description']
	# 	newitem_price_list = new_item['price']
	# 	#### tf-idf result for new item ####
	# 	newitem_title_corpus = self.data_preprocess(newitem_title_list)
	# 	newitem_title_result = self.cal_tf_idf(newitem_title_corpus)
	# 	newitem_description_corpus = self.data_preprocess(newitem_description_list)
	# 	newitem_description_result = self.cal_tf_idf(newitem_description_corpus)
	# 	#### price result for new item ####
	# 	newitem_price_list = new_item['price']
	# 	price_result = 2*(1.0 / (1+math.exp(-self.beta*(newitem_price_list[i]))))-1

	# 	return newitem_title_result[0], newitem_description_result[0], price_result


	# def cal_cos_similarity(self, vec1, vec2):
	# 	'''calculate cosine similarity between two vectors'''
	# 	dot_product = 0.0
	# 	normA = 0.0
	# 	normB = 0.0
	# 	for a, b in zip(vec1, vec2):
	# 		dot_product += a*b
	# 		normA += a**2
	# 		normB += b**2
	# 	if normA == 0.0 or normB == 0.0:
	# 		return 0.0
	# 	else:
	# 		return dot_product / ((normA*normB)**0.5)

	def generate_topk_item_similarity(self, new_asin, k):
		''' return cosine similarity matrix
		Input:
			k: int value
			new_asin: list
		Output:
			dict{ 
				new_item1 : {
					'00001': 3
					'00002': 2
					'00003': 1
				}
				new_item2 : {
					'00004': 7
					'00005': 6
					'00006': 5
				}	
				...
			}
		'''
		#### get the new&training dataset ####
		'''dict {
					'0001': 
						'title': []
						'description': []
						'price': float
					'0002':
						...
				}
		'''
		if k > self.item_num:
			k = self.item_num

		train_item_vec_set = {n:self.data.loc[self.data['asin']==n].index[0] for n in self.item_vector_set.keys() if n not in new_asin}
		new_item_vec_set = {n:self.item_vector_set[n] for n in new_asin}

		ret_k_similarities_dict = {}
		for new_key, new_value in new_item_vec_set.items():
			ret_k_similarities = []
			for train_key, train_value in train_item_vec_set.items():
				value = self.item_vector_set[new_key]['title'][train_value] \
							+ self.alpha*self.item_vector_set[new_key]['description'][train_value] \
							+ self.theta*(np.abs(self.price_result[train_value]-new_value['price']))
				ret_k_similarities.append([value, train_key])
			ret_k_similarities = sorted(ret_k_similarities, key=lambda x:x[0], reverse=True)
			ret_k_similarities_dict_one = {ret_k_similarities[i][1]:ret_k_similarities[i][0] for i in range(len(ret_k_similarities)) if i <= k-1}
			ret_k_similarities_dict[new_key] = ret_k_similarities_dict_one

		return ret_k_similarities_dict


		


	
				







		












