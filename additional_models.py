from string import punctuation

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes

from load_data import *
#from dataVisualization import *

def performance(y_true, y_pred, metric="accuracy") :
	"""
	Calculates the performance metric based on the agreement between the 
	true labels and the predicted labels.
	
	Parameters
	--------------------
		y_true -- numpy array of shape (n,), known labels
		y_pred -- numpy array of shape (n,), (continuous-valued) predictions
		metric -- string, option used to select the performance measure
				  options: 'accuracy', 'f1_score', 'auroc', 'precision',
						   'sensitivity', 'specificity'        
	
	Returns
	--------------------
		score  -- float, performance score
	"""
	# map continuous-valued predictions to binary labels
	y_label = np.sign(y_pred)
	y_label[y_label==0] = 1 # map points of hyperplane to +1
	
	### ========== TODO : START ========== ###
	# part 2a: compute classifier performance
	tp = 0 # True positive
	fp = 0 # False positive
	fn = 0 # False negative
	tn = 0 # True negative

	for i in range(len(y_true)):
		if y_true[i] == 1:
			if y_label[i] == 1:
				tp += 1
			else:
				fn += 1
		else:
			if y_label[i] == -1:
				tn += 1
			else:
				fp += 1
	score = 0
	precision = float(tp)/(tp+fp)
	recall = float(tp)/(tp+fn)

	if metric == 'accuracy':
		score = float(tp+tn)/len(y_true)
	elif metric == 'f1_score':
		score = (2*precision*recall)/(precision+recall)
	elif metric == 'auroc':
		fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
		score = metrics.auc(fpr, tpr)
	elif metric == 'precision':
		score = precision
	elif metric == 'sensitivity':
		score = recall
	elif metric == 'specificity':
		score = float(tn)/(tn+fp)
	else:
		print "Error"

	return score

def performance_CI(clf, X, y, metric="accuracy") :
	"""
	Estimates the performance of the classifier using the 95% CI.
	
	Parameters
	--------------------
		clf          -- classifier (instance of SVC or DummyClassifier)
						  [already fit to data]
		X            -- numpy array of shape (n,d), feature vectors of test set
						  n = number of examples
						  d = number of features
		y            -- numpy array of shape (n,), binary labels {1,-1} of test set
		metric       -- string, option used to select performance measure
	
	Returns
	--------------------
		score        -- float, classifier performance
		lower        -- float, lower limit of confidence interval
		upper        -- float, upper limit of confidence interval
	"""
	
	try :
		y_pred = clf.decision_function(X)
	except :
		y_pred = clf.predict(X)
	score = performance(y, y_pred, metric)
	
	### ========== TODO : START ========== ###
	# part 4b: use bootstrapping to compute 95% confidence interval
	# hint: use np.random.randint(...)
	n,d = X.shape
	t = 100
	performances = []

	for i in range(t):
		indices = [np.random.randint(0,n) for j in range(n)]
		X_test = [X[indices[j]] for j in range(n)]
		y_test = [y[indices[j]] for j in range(n)]
		y_pred = clf.predict(X_test)
		performances.append(performance(y_test, y_pred, metric=metric))

	performances.sort()

	lower = performances[25]
	upper = performances[98]

	return score, lower, upper

def plot_results(metrics, classifiers, *args):
	"""
	Make a results plot.
	
	Parameters
	--------------------
		metrics      -- list of strings, metrics
		classifiers  -- list of strings, classifiers
		args         -- variable length argument
						  results for baseline
						  results for classifier 1
						  results for classifier 2
						  ...
						each results is a tuple (score, lower, upper)
	"""
	
	num_metrics = len(metrics)
	num_classifiers = len(args) - 1
	
	ind = np.arange(num_metrics)  # the x locations for the groups
	width = 0.7 / num_classifiers # the width of the bars
	
	fig, ax = plt.subplots()
	
	# loop through classifiers
	rects_list = []
	for i in xrange(num_classifiers):
		results = args[i+1] # skip baseline
		means = [it[0] for it in results]
		errs = [(it[0] - it[1], it[2] - it[0]) for it in results]
		rects = ax.bar(ind + i * width, means, width, label=classifiers[i])
		ax.errorbar(ind + i * width, means, yerr=np.array(errs).T, fmt='none', ecolor='k')
		rects_list.append(rects)
	
	# baseline
	results = args[0]
	for i in xrange(num_metrics) :
		mean = results[i][0]
		err_low = results[i][1]
		err_high = results[i][2]
		xlim = (ind[i] - 0.8 * width, ind[i] + num_classifiers * width - 0.2 * width)
		plt.plot(xlim, [mean, mean], color='k', linestyle='-', linewidth=2)
		plt.plot(xlim, [err_low, err_low], color='k', linestyle='--', linewidth=2)
		plt.plot(xlim, [err_high, err_high], color='k', linestyle='--', linewidth=2)
	
	ax.set_ylabel('Score')
	ax.set_ylim(0, 1)
	ax.set_xticks(ind + width / num_classifiers)
	ax.set_xticklabels(metrics)
	ax.legend()
	
	def autolabel(rects):
		"""Attach a text label above each bar displaying its height"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
					'%.3f' % height, ha='center', va='bottom')
	
	for rects in rects_list:
		autolabel(rects)

	print "plot should be showing ip??"

	plt.show()

def read_vector_file(fname) :
	"""
	Reads and returns a vector from a file.
	
	Parameters
	--------------------
		fname  -- string, filename
	
	Returns
	--------------------
		labels -- numpy array of shape (n,)
					n is the number of non-blank lines in the text file
	"""
	return np.genfromtxt(fname)

def extract_words(input_string) :
	"""
	Processes the input_string, separating it into "words" based on the presence
	of spaces, and separating punctuation marks into their own words.
	
	Parameters
	--------------------
		input_string -- string of characters
	
	Returns
	--------------------
		words        -- list of lowercase "words"
	"""
	
	for c in punctuation :
		input_string = input_string.replace(c, ' ' + c + ' ')
	return input_string.lower().split()

def extract_dictionary(infile) :
	"""
	Given a filename, reads the text file and builds a dictionary of unique
	words/punctuations.
	
	Parameters
	--------------------
		infile    -- string, filename
	
	Returns
	--------------------
		word_list -- dictionary, (key, value) pairs are (word, index)
	"""
	
	word_list = {}
	with open(infile, 'rU') as fid :
		### ========== TODO : START ========== ###
		# part 1a: process each line to populate word_list
		all_lines = ''.join(fid.readlines())
		words = extract_words(all_lines)
		print "Number of total words: " + str(len(words))

		index = 0
		for w in words:
			if w not in word_list: # Only add unique words to dictionary
				word_list[w] = index
				index += 1

		### ========== TODO : END ========== ###

	return word_list


def extract_feature_vectors(infile, word_list) :
	"""
	Produces a bag-of-words representation of a text file specified by the
	filename infile based on the dictionary word_list.
	
	Parameters
	--------------------
		infile         -- string, filename
		word_list      -- dictionary, (key, value) pairs are (word, index)
	
	Returns
	--------------------
		feature_matrix -- numpy array of shape (n,d)
						  boolean (0,1) array indicating word presence in a string
							n is the number of non-blank lines in the text file
							d is the number of unique words in the text file
	"""
	
	num_lines = sum(1 for line in open(infile,'rU'))
	num_words = len(word_list)
	feature_matrix = np.zeros((num_lines, num_words))
	
	with open(infile, 'rU') as fid :
		### ========== TODO : START ========== ###
		# part 1b: process each line to populate feature_matrix
		for i in range(num_lines):
			line_words = extract_words(fid.readline())
			for word,index in word_list.iteritems():
				if word in line_words:
					feature_matrix[i][index] = 1
		### ========== TODO : END ========== ###
	
	return feature_matrix

def random_forest_classifier(features, target):
	"""
	To train the random forest classifier with features and target data
	:param features:
	:param target:
	:return: trained random forest classifier
	"""
	clf = RandomForestClassifier()
	clf.fit(features, target)
	return clf

def main():
	print "Hey"
	np.random.seed(1234)

	# read the tweets and its labels
	data_file = 'final_data/all_tweets.txt'
	label_file = 'final_data/labels.txt'
	#tweets = read_tweets(data_file)
	#print len(tweets)

	dictionary = extract_dictionary(data_file)
	print len(dictionary)
	#X = extract_feature_vectors(data_file, dictionary)
	# y = read_vector_file('final_data/labels.txt')
	X = tfidf('final_data/all_tweets.txt')
	print X.shape
	y = labels(label_file)
	print len(y)

	# shuffle data (since file has tweets ordered by author)
	X, y = shuffle(X, y)
	print X.shape

	X_train, X_test = X[:int(len(y) * 0.8)], X[int(len(y) * 0.8):]
	y_train, y_test = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

	trained_rf = random_forest_classifier(X_train, y_train)

	base_clf = DummyClassifier()
	base_clf.fit(X_train, y_train)

	logistic_clf = LogisticRegression()
	logistic_clf.fit(X_train, y_train)

	linear_clf = SVC(C=100, kernel='rbf')
	linear_clf.fit(X_train, y_train)

	nb_clf = naive_bayes.MultinomialNB()
	nb_clf.fit(X_train, y_train)

	dummy_results = []
	rf_results = []
	linear_results = []
	logistic_results = []
	nb_results = []

	print "Dummy"
	dummy_results.append(performance_CI(base_clf, X_test, y_test, "f1_score"))
	dummy_results.append(performance_CI(base_clf, X_test, y_test, "accuracy"))

	print "Linear SVN"
	linear_results.append(performance_CI(linear_clf, X_test, y_test, "f1_score"))
	linear_results.append(performance_CI(linear_clf, X_test, y_test, "accuracy"))

	print "Random Forest"
	rf_results.append(performance_CI(trained_rf, X_test, y_test, "f1_score"))
	rf_results.append(performance_CI(trained_rf, X_test, y_test, "accuracy"))

	print "Logistic Regression"
	logistic_results.append(performance_CI(logistic_clf, X_test, y_test, "f1_score"))
	logistic_results.append(performance_CI(logistic_clf, X_test, y_test, "accuracy"))

	print "now plotting"	
	plot_results(["accuracy", "f1 score"], ("Random Forest", "RBF SVM", "Logistic Regression"), dummy_results, rf_results, linear_results, logistic_results)

if __name__ == "__main__" :
	main()










