import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import os, fnmatch
import random
from operator import itemgetter
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))

def precompute_m(theta):
    M = theta.Sigma.shape[0]
    precomp = np.zeros(M)
    for m in range(M):
        a = np.sum((theta.mu[m] ** 2) / (theta.Sigma[m]))#**2))
        b = theta.Sigma.shape[1] * np.log(2 * np.pi)
        c = np.log(np.product(theta.Sigma[m]))
        precomp[m] = 0.5 * (a + b + c)
    return precomp

def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    if len(x.shape) > 1:
        a = - (np.sum((0.5 * (x**2) - (myTheta.mu[m] * x)) / myTheta.Sigma[m], axis=1))
    elif len(x.shape) == 1:
        a = - np.sum((0.5 * (x**2)/myTheta.Sigma[m]) - ((myTheta.mu[m] * x)/myTheta.Sigma[m]))
    return a - preComputedForM[m]


def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    prec = precompute_m(myTheta)
    lbs = []
    for i in range(myTheta.omega.shape[0]):
        lbs.append(log_b_m_x(i, x, myTheta, prec))

    lwm = np.log(myTheta.omega[m])
    lbm = lbs[m]

    if len(x.shape) > 1:

        lsum = logsumexp(np.log(myTheta.omega) + lbs, axis=0)       #maybe change to *
    elif len(x.shape) == 1:
        lsum = logsumexp(np.log(myTheta.omega) + lbs)
    return lwm + lbm - lsum

def logLik(log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''

    lo = np.log(myTheta.omega)
    a = np.sum(logsumexp(lo + log_Bs, axis=0))

    return a



def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])
    T = X.shape[0]
    for k in range(M):
        myTheta.mu[k] = X[np.random.randint(T)]
        myTheta.Sigma[k] = np.ones(X.shape[1])
    rand = np.random.random(M)
    myTheta.omega = (rand/rand.sum()).reshape(M, 1)
    i = 0
    prev_L, improvement = -float('inf'), float('inf')
    while i <= maxIter and improvement >= epsilon:

        lbs = np.zeros((M, T))
        lps = np.zeros((M, T))
        prec = precompute_m(myTheta)
        for m in range(M):
                                        #compute intermediate
            lbs[m] = log_b_m_x(m, X, myTheta, prec)
        for m in range(M):
            lps[m] = log_p_m_x(m, X, myTheta)

        L = logLik(lbs, myTheta)
        # update params

        pm = np.exp(lps)
        myTheta.omega = (np.sum(pm, axis=1) / T).reshape(M, 1)
        for j in range(M):
            myTheta.mu[j] = np.dot(pm[j], X) / np.sum(pm[j])
        for j in range(M):
            myTheta.Sigma[j] = (np.dot(pm[j], X**2) / np.sum(pm[j]))\
                                - myTheta.mu[j] ** 2
        improvement = L - prev_L
        prev_L = L
        i += 1


    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    M = models[0].mu.shape[0]
    lls = []
    for model in models:
        log_bs = []
        prec = precompute_m(model)
        for m in range(M):
            log_bs.append(log_b_m_x(m, mfcc, model, prec))

        ll = logLik(log_bs, model)

        if len(lls) < k:
            lls.append((model.name, ll))
        else:
            lls_sorted = sorted(lls, key=itemgetter(1), reverse=True)
            if ll > lls[-1][1]:
                lls[-1] = (model.name, ll)
                lls = sorted(lls, key=itemgetter(1), reverse=True)

    lls = sorted(lls, key=itemgetter(1), reverse=True)

    realID = models[correctID].name
    # print("Actual ID: " + realID)
    # for pair in lls:
    #     print(pair)
    # with open('gmmLiks.txt', 'w') as f:
    #     f.write("Actual ID: [" + realID +']\n')
    #     for pair in lls:
    #         f.write(str(pair) + '\n')



    bestModel = lls[0][0]



    return 1 if (bestModel == realID) else 0


if __name__ == "__main__":


    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    Ms = [3, 5, 8, 10]
    M = 8
    epsilon = 0
    maxIter = 25

    epsilons = [0.0, 0.001, 0.01, 0.1]
    maxIters = [5, 15, 25, 30]
    numSpeakers = [0, 5, 15, 20]
    output = []
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        # for M in Ms:
        #     for numSpeakers in numSpeakers:
        #         for maxIter in maxIters:
        #             for epsilon in epsilons:
        if len(dirs) > 0:
            trainThetas = []
            testMFCCs = []
 #           random.shuffle(dirs)
#            [dirs.pop() for i in range(numSpeakers)]
            for speaker in dirs:

                print( speaker )



                files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
                random.shuffle( files )

                testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
                testMFCCs.append( testMFCC )

                X = np.empty((0,d))
                for file in files:
                    myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                    X = np.append( X, myMFCC, axis=0)

                trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate
            numCorrect = 0;
            for i in range(0, len(testMFCCs)):
                numCorrect += test(testMFCCs[i], i, trainThetas)
            accuracy = 1.0*numCorrect/len(testMFCCs)
        print("M = {}, maxIter = {}, numSpeakers = {}, epsilon = {}, accuracy = {}".format(M, maxIter,
                                                                                      len(dirs),
                                                                                      epsilon,
                                                                                      accuracy))
        output.append("M = {}, maxIter = {}, numSpeakers = {}, epsilon = {}, accuracy = {}".format(M, maxIter, len(dirs), epsilon, accuracy))
        print(accuracy)
    #
    # with open('gmmDiscussion.txt', 'w') as f:
    #     for out in output:
    #         f.write(out + '\n')




