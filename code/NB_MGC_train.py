# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import naive_bayes_gaussian_classifier, vcol, Bayes_risk_min_cost, accuracy
from library import load

def Ksplit(D, L, seed=0, K=3):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    # Generate a random seed
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
        labels.append(L[idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
    return folds, labels

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    mu = vcol(D.mean(1))

    DC = D - mu

    C = 0
    dotDC = numpy.dot(D, D.T)
    C = (1 / float(D.shape[1])) * dotDC

    U, s, Vh = numpy.linalg.svd(C)

    m = 3
    num = 0
    den = 0

    for m in range(7, s.size + 1):
        print("PCA:", m)
        P = U[:, 0:m]

        DP = numpy.dot(P.T, D)
        
        LTEs = []
        scores = []
        K = 3
        folds, labels = Ksplit(DP, L, seed=0, K=K)
        LTE = []
        scores = []
        LPs = []
        for i in range(K):
            DTR = []
            LTR = []
            for j in range(K):
                if j!=i:
                    DTR.append(folds[j])
                    LTR.append(labels[j])
            DTE = folds[i]
            LTE.append(labels[i])
            DTR=numpy.hstack(DTR)
            LTR=numpy.hstack(LTR)
            s, LP = naive_bayes_gaussian_classifier(DTR, LTR, DTE, LTE, 0.5)
            scores.append(s)
            LPs.append(LP)
        scores=numpy.hstack(scores)
        scores = numpy.max(scores, axis = 0)
        LTE=numpy.hstack(LTE)
        labels = numpy.hstack(labels)
        LPs = numpy.hstack(LPs)
        accuracy(LPs, labels)
        print("min cost:", Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))