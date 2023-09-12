# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import polynomial_kernel_SVM, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, plot_minDCF_svm_poly, plotDCFpoly
from library import load

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    priors = [0.5, 0.1, 0.9]
    K = 1
    vect_C = numpy.logspace(-5, -1, num=15)
    d = 2
    vect_c = [0, 1, 10, 30]
    print("Polynomial kernel SVM RAW features")
    for p in priors:
        print("Prior:", p)
        for m in range(10, 13):
            print("PCA:", m)
            minDCs = []
            for c in vect_c:
                print("c:", c)
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
                        s = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE)
                        scores.append(s)
                        LTEs.append(LTE)
                    scores = numpy.hstack(scores)
                    orderedLabels = numpy.hstack(LTEs)
                    min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                    print("min cost: %.3f" %min_cost)
                    minDCs.append(min_cost)
            plotDCFpoly(vect_C, minDCs, "C", m, p)
            #plot_minDCF_svm_poly(vect_C, minDCs[0:len(vect_C)], minDCs[len(vect_C):2*len(vect_C)], minDCs[2*len(vect_C):], filename="RAW_PCA" + str(m), title="RAW_MinDCF_PCA" + str(m))
              
    
    D = Z_norm(D)
    print("Polynomial kernel SVM Z_norm features")
    for p in priors:
        print("Prior:", p)
        for m in range(10, 13):
            print("PCA:", m)
            minDCs = []
            for c in vect_c:
                print("c:", c)
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
                        s = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE)
                        scores.append(s)
                        LTEs.append(LTE)
                    scores = numpy.hstack(scores)
                    orderedLabels = numpy.hstack(LTEs)
                    min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                    print("min cost: %.3f" %min_cost)
                    minDCs.append(min_cost)
            plotDCFpoly(vect_C, minDCs, "C", m, p)
            #plot_minDCF_svm_poly(vect_C, minDCs[0:len(vect_C)], minDCs[len(vect_C):2*len(vect_C)], minDCs[2*len(vect_C):], filename="RAW_PCA" + str(m), title="RAW_MinDCF_PCA" + str(m))
              
