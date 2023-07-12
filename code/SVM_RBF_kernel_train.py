# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import RBF_kernel_SVM, Bayes_risk_min_cost, PCA, Ksplit, Z_norm
from library import load

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
       
    K_C_gamma = [[0, 1, 1], [0, 1, 10], [1, 1, 1], [1, 1, 10]]
    priors = [0.5, 0.1, 0.9]
    print("RBF kernel SVM RAW features")
    for p in priors:
        print("Prior:", p)
        for K, C, gamma in K_C_gamma:
            print("K:", K)
            print("C:", C)
            print("gamma:", gamma)
            for m in range(10, 13):
                print("PCA:", m)
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
            print()
            
    D = Z_norm(D)
    print("RBF kernel SVM Z_norm features")
    for p in priors:
        print("Prior:", p)
        for K, C, gamma in K_C_gamma:
            print("K:", K)
            print("C:", C)
            print("gamma:", gamma)
            for m in range(10, 13):
                print("PCA:", m)
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
            print()