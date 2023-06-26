# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import dual_SVM, Bayes_risk_min_cost, accuracy_v2, PCA, Ksplit
from library import load

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    print("SVM")
    C_K = [[0.1, 1], [1.0, 1], [10.0, 1], [0.1, 10], [1.0, 10], [10.0, 10]]
    for C, K in C_K:
        print("C:", C, "K:", K)
        for m in range(10, 13):
            print("PCA:", m)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K_fold = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K_fold)
            acc = 0
            min_cost = 0
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
                w = dual_SVM(DTR, LTR, K, C)
                scores = numpy.dot(w.T, numpy.vstack((DTE, numpy.zeros(DTE.shape[1]) + K)))
                acc += accuracy_v2(scores, LTE)
                min_cost += Bayes_risk_min_cost(0.5, 1, 1, scores, LTE)
            print("Error rate %.1f" %(acc/K_fold), "%")
            print("min cost:", min_cost/K_fold)
            print()