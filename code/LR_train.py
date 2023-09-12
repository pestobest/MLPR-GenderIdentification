# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import logreg_obj_wrapper, logreg, Bayes_risk_min_cost, Z_norm, PCA, Ksplit, plot_minDCF_lr
from library import load
import scipy

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    l_vec = numpy.logspace(-5, 2, 30)
    print("Logistic Regression RAW feature")
    priors = [0.5, 0.1, 0.9]
    for m in range(10, 13):
        print("PCA:", m)
        minDCs = []
        for p in priors:
            print("Prior:", p)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K)
            
            for l in l_vec:
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
                    logreg_obj = logreg_obj_wrapper(DTR, LTR, l, p)
                    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                                            approx_grad=True)
                    s, LP = logreg(x, DTE)
                    scores.append(s)
                    LTEs.append(LTE)
                scores = numpy.hstack(scores)
                orderedLabels = numpy.hstack(LTEs)
                min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                print("min cost: %.3f" %min_cost, "lambda: ", l)
                minDCs.append(min_cost)
        plot_minDCF_lr(l_vec, minDCs[0:len(l_vec)], minDCs[len(l_vec):2*len(l_vec)], minDCs[2*len(l_vec):], filename="RAW_PCA" + str(m), title="RAW_MinDCF_PCA" + str(m))
        print()
    
    D = Z_norm(D)
    print("Logistic Regression Z-norm features")
    for m in range(10, 13):
        print("PCA:", m)
        minDCs = []
        for p in priors:
            print("Prior:", p)
            print("PCA:", m)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K)
            for l in l_vec:
                #print("lambda:", l)
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
                    logreg_obj = logreg_obj_wrapper(DTR, LTR, l, p)
                    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                           x0=numpy.zeros(DTR.shape[0] + 1),
                                                           approx_grad=True)
                    s, LP = logreg(x, DTE)
                    scores.append(s)
                    LTEs.append(LTE)
                scores = numpy.hstack(scores)
                orderedLabels = numpy.hstack(LTEs)
                min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                print("min cost: %.3f" %min_cost, "lambda: ", l)
                minDCs.append(min_cost)
        plot_minDCF_lr(l_vec, minDCs[0:len(l_vec)], minDCs[len(l_vec):2*len(l_vec)], minDCs[2*len(l_vec):], filename="Z_norm_PCA" + str(m), title="Z_norm_MinDCF_PCA" + str(m))
        print()