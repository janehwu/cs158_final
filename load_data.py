import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

'''
TF-IDF
'''
def tfidf(filename):
	with open(filename, 'r') as f:
		vectorizer = TfidfVectorizer()
		return vectorizer.fit_transform(f)

def labels(filename):
	y = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			y.append(int(line))
	return np.asarray(y)


def get_data():
	X = tfidf('final_data/all_tweets.txt')
	y = labels('final_data/labels.txt')
	print X.shape
	print y.shape
	return X,y