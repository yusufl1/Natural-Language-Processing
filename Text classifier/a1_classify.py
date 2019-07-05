import csv
import sklearn
from sklearn import ensemble
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import argparse
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    correct = 0
    for i in range(len(C)):
        correct += C[i][i]
    return correct / C.sum()

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recalls = []
    for i in range(len(C)):
        recalls.append(C[i][i] / C[i].sum())
    return recalls

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precs = []
    for i in range(len(C)):
        precs.append(C[i][i] / C[:, i].sum())
    return precs

def split(data):
    train, test = train_test_split(data['arr_0'], test_size=0.2)
    X_train, y_train = train[:, :173], train[:, -1]
    X_test, y_test = test[:, :173], test[:, -1]
    return X_train, y_train, X_test, y_test

def train_SVC_lk(X, y):
    model = sklearn.svm.LinearSVC(loss='hinge', max_iter=10000)
    return model.fit(X, y)

def train_SVC_rb(X, y):
    model = sklearn.svm.SVC(gamma=2, max_iter=10000)
    return model.fit(X, y)
def train_rf(X, y):
    model = sklearn.ensemble.RandomForestClassifier(max_depth=5)
    return model.fit(X, y)
def train_mlp(X, y):
    model = sklearn.neural_network.MLPClassifier(alpha=0.05)
    return model.fit(X, y)
def train_ada(X, y):
    model = sklearn.ensemble.AdaBoostClassifier()
    return model.fit(X, y)



def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training fgeatures
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    data = np.load(filename)

    X_train, y_train, X_test, y_test = split(data)       #use helper fn to split data


    svc_lk = train_SVC_lk(X_train, y_train)       #initialize and train models using helper fns
    svc_rb = train_SVC_rb(X_train, y_train)
    rf = train_rf(X_train, y_train)

    mlp = train_mlp(X_train, y_train)

    ada = train_ada(X_train, y_train)


    models = [svc_lk, svc_rb, rf, mlp, ada]
    output = []

    accuracies = []

    i = 1
    for model in models:       #use each model to predict, using predictions get confusion matrix, accuracy.
        prediction = model.predict(X_test)     #recall, precision
        confusion = sklearn.metrics.confusion_matrix(y_test, prediction)
        accuracies.append(accuracy(confusion))
        output.append([i, accuracy(confusion), recall(confusion), precision(confusion), list(confusion.flatten())])
        i += 1

    with open('a1_3.1.csv', mode='w') as out:       #write to csv
        outwriter = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output:
            outwriter.writerow(row)

    iBest = accuracies.index(max(accuracies)) + 1


    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    classifiers = {1: svm.SVC(kernel='linear'), 2: svm.SVC(gamma=2), 3: ensemble.RandomForestClassifier(max_depth=5),\
                   4: neural_network.MLPClassifier(alpha=0.05), 5: ensemble.AdaBoostClassifier()}

    sizes = [1000, 5000, 10000, 15000, 20000]

    acc = []

    for size in sizes:
        clsfr = classifiers[iBest]
        ind = np.random.choice(X_train.shape[0], size)
        if size == 1000:
            X_1k, y_1k = X_train[ind, :], y_train[ind]
        X, y = X_train[ind, :], y_train[ind]                 #take sample of data of given size, train on sample,
        clsfr.fit(X, y)                                      #make predictions, get confusion matrix and then accuracy
        pred = clsfr.predict(X_test)
        conf = confusion_matrix(y_test, pred)
        acc.append(accuracy(conf))

    with open('a1_3.2.csv', 'w') as out:
        write = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        write.writerow(acc)


    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''


    features = [5, 10, 20, 30, 40, 50]
    classifiers = {1: svm.SVC(kernel='linear'), 2: svm.SVC(gamma=2), 3: ensemble.RandomForestClassifier(max_depth=5), \
                   4: neural_network.MLPClassifier(alpha=0.05), 5: ensemble.AdaBoostClassifier()}

    pp1k = []
    with open('a1_3.3.csv', 'w') as f:
        write = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for s in features:
            selector = SelectKBest(f_classif, k=s)          #initialize selector with f_classif
            X_new = selector.fit_transform(X_train, y_train)     # fit using the s best features
            pp = list(selector.pvalues_)

            iobf = list(selector.get_support(True))           #get indices of best features

            ps = []
            for j in iobf:
                ps.append(pp[j])

            write.writerow([s] + ps)               #write p-value to csv

            selector1k = SelectKBest(f_classif, k=s)          # select best features for 1k case
            X_new1k = selector1k.fit_transform(X_1k, y_1k)
            ppp = selector1k.pvalues_

            iobfk = selector1k.get_support(True)



            pp1k = []
            for f in list(iobfk):
                pp1k.append(ppp[f])


            if s == 5:                     #get indices of top 5 features
                Xt = X_new                 #save the transformed data to make predictions
                f_id = selector.get_support(indices=True)
                Xt1k = X_new1k
                f_id1k = selector1k.get_support(indices=True)




        clsfr, clsfr1k = classifiers[i], classifiers[i]        # choose classifier at index iBest


        clsfr.fit(Xt, y_train), clsfr1k.fit(Xt1k, y_1k)
        X_test_new = np.take(X_test, f_id, axis=1)
        X_test_new1k = np.take(X_test, f_id1k, axis=1)      # get accuracy of predictions for full data set and 1k set

        y_pred1k = clsfr1k.predict(X_test_new1k)
        y_pred = clsfr.predict(X_test_new)

        C1k = confusion_matrix(y_test, y_pred1k)
        C = confusion_matrix(y_test, y_pred)

        a1k, a = accuracy(C1k), accuracy(C)
        write.writerow([a1k, a])                 #write to csv









def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    data = np.load(filename)["arr_0"]
    X = data[:, :173]
    y = data[:, -1]

    lin = svm.LinearSVC(loss='hinge', max_iter=10000)     #initialize models
    rbf = svm.SVC(gamma=2)
    rf = ensemble.RandomForestClassifier(max_depth=5)
    mlp = neural_network.MLPClassifier(alpha=0.05)
    ada = ensemble.AdaBoostClassifier()

    models = [lin, rbf, rf, mlp, ada]

    kf = KFold(n_splits=5, shuffle=True)
    out = []

    split = kf.split(X)

    for train_i, test_i in split:        #do 5 fold cross validation
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]

        accs = []
        for model in models: #train and predict all models,
            model.fit(X_train, y_train)    # get CV error for test_i
            pred = model.predict(X_test)
            C = confusion_matrix(y_test, pred)
            accs.append(accuracy(C))

        out.append(accs)

    out = np.array(out)

    pv = []

    for k in range(4):       #get
        S = stats.ttest_rel(out[:, k], out[:, i-1])
        pv.append(S)

    with open('a1_3.4.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in out:
            writer.writerow(row)
        writer.writerow(pv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    # X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    #
    # class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    # class34("out.npz", iBest)






