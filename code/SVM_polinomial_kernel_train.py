# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import polynomial_kernel_SVM, Bayes_risk_min_cost, accuracy_v2, PCA, Ksplit
from library import load

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    print("Polynomial kernel SVM")
    
    K_C_d_c = [[0, 1, 2, 0], [1, 1, 2, 0], [0, 1, 2, 1], [1, 1, 2, 1]]
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
                scores = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, LTE, DTE)
                acc += accuracy_v2(scores, LTE)
                min_cost += Bayes_risk_min_cost(0.5, 1, 1, scores, LTE)
            print("Error rate %.1f" %(acc/K_fold), "%")
            print("min cost:", min_cost/K_fold)
            print()