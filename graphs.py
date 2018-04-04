"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2018 Feb 14
Description : Twitter
"""

#Emily Dorsey and Julio Medina
#PS6

from string import punctuation

# numpy libraries
import numpy as np
import collections

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle

######################################################################
# functions -- input/output
######################################################################

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


######################################################################
# functions -- feature extraction
######################################################################

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
        lineList = fid.readlines()

        for i in range(len(lineList)):
            current_line_words = extract_words(lineList[i])
            #print(lineList[i])

            for word in current_line_words:
                if word not in word_list:
                    word_list[word] = len(word_list)
    pass
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

        lineList = fid.readlines()
        
        for i in range(len(lineList)):

            extracted_words = extract_words(lineList[i])

            # for each of this tweets words, we then set its index equal to 1
            for word in extracted_words:
                feature_matrix[i][word_list[word]] = 1

        pass
        ### ========== TODO : END ========== ###
    
    return feature_matrix


def test_extract_dictionary(dictionary) :
    err = "extract_dictionary implementation incorrect"
    
    assert len(dictionary) == 1811, err
    
    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0,100,10)]
    assert exp == act, err


def test_extract_feature_vectors(X) :
    err = "extract_features_vectors implementation incorrect"
    
    assert X.shape == (630, 1811), err
    
    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all(), err


######################################################################
# functions -- evaluation
######################################################################

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

    tn, fp, fn, tp =  confusion_matrix = metrics.confusion_matrix(y_true, y_label).ravel()

    score = 0

    if(metric == "accuracy"):
        score = metrics.accuracy_score(y_true, y_label)
    elif(metric == "f1_score"):
        score = metrics.f1_score(y_true, y_label)   
    elif(metric == "auroc"):
        score = metrics.roc_auc_score(y_true, y_pred)
    elif(metric == "precision"):
        score = metrics.precision_score(y_true, y_label)
    elif(metric == "sensitivity"):
        score = metrics.recall_score(y_true, y_label)
    elif(metric == "specificity"):
        score = float(tn) / float(tn + fp)
    else:
        score = 0

    return score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy") :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    scores = []
    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score) :
            scores.append(score)
    return np.array(scores).mean()


def select_param_linear(X, y, kf, metric="accuracy", plot=True) :
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure
        plot   -- boolean, make a plot
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation

    scores = [0 for _ in xrange(len(C_range))] # dummy values, feel free to change

    i = 0
    max_index = 0


    for curr_c in C_range:
        perf_total = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = SVC(C = curr_c, kernel = "linear")
            model.fit(X_train, y_train)
            predictions = model.decision_function(X_test)
            perf_total += performance(y_test, predictions, metric)

        perf = perf_total/kf.n_splits
        scores[i] = perf
        if perf > scores[max_index]:
            max_index = i

        i = i + 1

    if plot:
        lineplot(C_range, scores, metric)

    return C_range[max_index], scores
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy") :
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        C        -- float, optimal parameter value for an RBF-kernel SVM
        gamma    -- float, optimal parameter value for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = np.logspace(-9, 3, 13)
    grid = [[0 for _ in xrange(len(gamma_range))] for _ in xrange(len(C_range))]

    i = 0

    for curr_c in C_range:
        j = 0
        for gamma in gamma_range:
            perf_total = 0

            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = SVC(kernel = 'rbf', gamma = gamma)
                model.C = curr_c
                model.fit(X_train, y_train)
                predictions = model.decision_function(X_test)
                perf_total += performance(y_test, predictions, metric)

            grid[i][j] = perf_total/kf.n_splits
            j = j + 1

        i = i + 1


    # get the index of the max value in a FLATTENED grid
    maxxx_i = np.argmax(grid)

    #now we figure out the location of that in the 2D grid array
    gamma_index = maxxx_i % len(gamma_range)
    c_index = maxxx_i / len(gamma_range)


    return C_range[c_index], gamma_range[gamma_index] 
    ### ========== TODO : END ========== ###


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
    t = 1000
    n = len(y)
    boot_scores = []

    for i in range(t):
        samplesX = []
        samplesY = []
        for curr_sample in range(n):
            rand_index = np.random.randint(0, n)
            samplesX.append(X[rand_index])
            samplesY.append(y[rand_index])
        predictions = clf.predict(samplesX)
        newScore = performance(samplesY, predictions, metric)
        boot_scores.append(newScore)

    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)

    

    return score, lower, upper
    ### ========== TODO : END ========== ###


######################################################################
# functions -- plotting
######################################################################

def lineplot(x, y, label):
    """
    Make a line plot.
    
    Parameters
    --------------------
        x            -- list of doubles, x values
        y            -- list of doubles, y values
        label        -- string, label for legend
    """
    
    xx = range(len(x))
    plt.plot(xx, y, linestyle='-', linewidth=2, label=label)
    plt.xticks(xx, x)    
    #plt.show()


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
    
    plt.show()


######################################################################
# main
######################################################################
 
def main() :
    # read the tweets and its labels
    dictionary = extract_dictionary('data/alltweets.txt')
    X = extract_feature_vectors('data/alltweets.txt', dictionary)
    y = read_vector_file('data/labels.txt')


    kimDict = extract_dictionary('data/kim_tweets.txt')
    obamaDict = extract_dictionary('data/obama_tweets.txt')

    print(len(X))

    #print dictionary
    c = collections.Counter(kimDict)
    common =  c.most_common(10)


    keys = [x[0] for x in common]
    values = [x[1] for x in common]

    for i in range(len(keys)):
        keys[i] = unicode(keys[i], 'utf-8')
        keys[i] = keys[i].encode('unicode_escape')


    y_pos = np.arange(len(keys))

    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, keys)
    plt.ylabel('Word')
    plt.title('Kim Kardashian\'s Most Common Words')

    plt.show()
    
    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)
    
    # set random seed
    np.random.seed(1234)
    
    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    

    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 15)


    rbf_model = SVC(kernel = 'rbf', gamma = 0.01, C = 100)
    linear_model = SVC(C = 100, kernel = 'linear')
    dummy_model = DummyClassifier(strategy = 'uniform')

    rbf_model.fit(X_train, y_train)
    linear_model.fit(X_train, y_train)
    dummy_model.fit(X_train, y_train)

    linear_preds = linear_model.decision_function(X_test)
    rbf_preds = rbf_model.decision_function(X_test)

    print linear_preds

    error = np.sum(y_test != linear_preds)

   #  rbf_scores = []
   #  svc_scores = []
   #  dummy_scores = []
   #  for metric in metric_list:
   #      print "Calculating " + metric
   #      rbf_scores += [performance_CI(rbf_model, X, y, metric)]
   #      svc_scores += [performance_CI(linear_model, X, y, metric)]
   #      dummy_scores += [performance_CI(dummy_model, X, y, metric)]

   #  # part 4c: use bootstrapping to report performance on test data
   #  #          use plot_results(...) to make plot
   # # classifiers_list = ["Linear","RBF"]
   #  plot_results(metric_list, ["RBF", "Linear", "Dummy"], rbf_scores, svc_scores, dummy_scores)

    # part 5: identify important features

    sorted_index_coefs = np.argsort(linear_model.coef_[0])
    bottom_10 = sorted_index_coefs[:10]
    top_10 = sorted_index_coefs[:-10:-1]
        
     
    topWords = []
    for i in top_10:
         topWords.append(dictionary.keys()[dictionary.values().index(i)])    
    print topWords

    bottomWords = []
    for i in bottom_10:
        bottomWords.append(dictionary.keys()[dictionary.values().index(i)])        

    for i in range(len(topWords)):
        topWords[i] = unicode(topWords[i], 'utf-8')
        topWords[i] = topWords[i].encode('unicode_escape')

    y_pos = np.arange(len(topWords))

    plt.bar(y_pos, top_10, align='center', alpha=0.5)
    plt.xticks(y_pos, topWords)
    plt.ylabel('Word')
    plt.title('Words with Higest Coefficients (More Kim-like)')

    plt.show()


    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
