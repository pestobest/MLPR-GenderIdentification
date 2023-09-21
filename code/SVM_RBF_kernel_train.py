# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from library import load, RBF_kernel_SVM, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, plotDCFRBF

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    K = 1   
    vect_C = numpy.logspace(-3, 3, num=15)
    vect_gamma = [10**(-5), 10**(-4), 10**(-3)]
    priors = [0.5, 0.1, 0.9]
    print("RBF kernel SVM RAW features")
    for p in priors:
        print("Prior:", p)
        for m in range(10, 13):
            print("PCA:", m)
            minDCs = []
            for gamma in vect_gamma:
                print("gamma:", gamma)
                for C in vect_C:
                    print("C:", C)
                    P = PCA(D, m)
        
                    DP = numpy.dot(P.T, D)
                    
                    K_fold = 5
                    folds, labels = Ksplit(DP, L, seed=0, K=K_fold)
                    scores = []
                    LTEs = []
                    for i in range(K_fold):
                        DTR = []
                        LTR = []
                        for j in range(K_fold):
                            if j!=i:
                                DTR.append(folds[j])
                                LTR.append(labels[j])
                        DTE = folds[i]
                        LTE = labels[i]
                        DTR = numpy.hstack(DTR)
                        LTR = numpy.hstack(LTR)
                        s = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE)
                        scores.append(s)
                        LTEs.append(LTE)
                    scores = numpy.hstack(scores)
                    orderedLabels = numpy.hstack(LTEs)
                    min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                    print("min cost: %.3f" %min_cost)
                    minDCs.append(min_cost)
            plotDCFRBF(vect_C, minDCs, "C", m, p)
            
    D = Z_norm(D)
    print("RBF kernel SVM Z_norm features")
    for p in priors:
        print("Prior:", p)
        for m in range(10, 13):
            print("PCA:", m)
            minDCs = []
            for gamma in vect_gamma:
                print("gamma:", gamma)
                for C in vect_C:
                    print("C:", C)
                    P = PCA(D, m)
        
                    DP = numpy.dot(P.T, D)
                    
                    K_fold = 5
                    folds, labels = Ksplit(DP, L, seed=0, K=K_fold)
                    scores = []
                    LTEs = []
                    for i in range(K_fold):
                        DTR = []
                        LTR = []
                        for j in range(K_fold):
                            if j!=i:
                                DTR.append(folds[j])
                                LTR.append(labels[j])
                        DTE = folds[i]
                        LTE = labels[i]
                        DTR = numpy.hstack(DTR)
                        LTR = numpy.hstack(LTR)
                        s = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE)
                        scores.append(s)
                        LTEs.append(LTE)
                    scores = numpy.hstack(scores)
                    orderedLabels = numpy.hstack(LTEs)
                    min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                    print("min cost: %.3f" %min_cost)
                    minDCs.append(min_cost)
            plotDCFRBF(vect_C, minDCs, "C", m, p)