# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import polynomial_kernel_SVM, Bayes_risk_min_cost, PCA, Ksplit, Z_norm
from library import load

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    K_C_d_c = [[0, 1, 2, 0], [1, 1, 2, 0], [0, 1, 2, 1], [1, 1, 2, 1]]
    priors = [0.5, 0.1, 0.9]
    print("Polynomial kernel SVM RAW features")
    for p in priors:
        print("Prior:", p)
        for K, C, d, c in K_C_d_c:
            print("K:", K)
            print("C:", C)
            print("d:", d)
            print("c:", c)
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
                    s = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE)
                    scores.append(s)
                    LTEs.append(LTE)
                scores = numpy.hstack(scores)
                orderedLabels = numpy.hstack(LTEs)
                min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                print("min cost: %.3f" %min_cost)
            print()
    
    D = Z_norm(D)
    print("Polynomial kernel SVM Z_norm features")
    for p in priors:
        print("Prior:", p)
        for K, C, d, c in K_C_d_c:
            print("K:", K)
            print("C:", C)
            print("d:", d)
            print("c:", c)
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
                    s = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE)
                    scores.append(s)
                    LTEs.append(LTE)
                scores = numpy.hstack(scores)
                orderedLabels = numpy.hstack(LTEs)
                min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                print("min cost: %.3f" %min_cost)
            print()