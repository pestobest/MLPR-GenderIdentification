# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import tied_naive_bayes_classier, Bayes_risk_min_cost, Z_norm, PCA, Ksplit
from library import load

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    print("Tied Naive Bayes model RAW features")
    priors = [0.5, 0.1, 0.9]
    for p in priors:
        print("Prior:", p)
        for m in range(7, 13):
            print("PCA:", m)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K)
            scores = []
            LTEs = []
            for i in range(K):
                DTR = []
                LTR = []
                for j in range(K):
                    if j!=i:
                        DTR.append(folds[j])
                        LTR.append(labels[j])
                DTE = folds[i]
                LTE = labels[i]
                DTR = numpy.hstack(DTR)
                LTR = numpy.hstack(LTR)
                s, LP = tied_naive_bayes_classier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
                scores.append(s)
                LTEs.append(LTE)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(LTEs)
            min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
            print("min cost: %.3f" %min_cost)
        print()
        
    D = Z_norm(D)
    print("Tied Naive Bayes model Z-norm features")
    for p in priors:
        print("Prior:", p)
        for m in range(7, 13):
            print("PCA:", m)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K)
            scores = []
            LTEs = []
            for i in range(K):
                DTR = []
                LTR = []
                for j in range(K):
                    if j!=i:
                        DTR.append(folds[j])
                        LTR.append(labels[j])
                DTE = folds[i]
                LTE = labels[i]
                DTR = numpy.hstack(DTR)
                LTR = numpy.hstack(LTR)
                s, LP = tied_naive_bayes_classier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
                scores.append(s)
                LTEs.append(LTE)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(LTEs)
            min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
            print("min cost: %.3f" %min_cost)
        print()