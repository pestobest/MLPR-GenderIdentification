# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import logreg_obj_wrap_prof, logreg, Bayes_risk_min_cost, accuracy, PCA, Ksplit
from library import load
import scipy

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    print("Logistic Regression")

    for m in range(7, 13):
        print("PCA:", m)
        P = PCA(D, m)

        DP = numpy.dot(P.T, D)
        
        K = 5
        folds, labels = Ksplit(DP, L, seed=0, K=K)
        acc = 0
        min_cost = 0
        l_vec=[10**(-5), 10**(-3), 10**(-1), 1.0]
        for l in l_vec:
            print("lambda:", l)
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
                logreg_obj = logreg_obj_wrap_prof(DTR, LTR, l)
                x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                       x0=numpy.zeros(DTR.shape[0] + 1),
                                                       approx_grad=True)
                s, LP = logreg(x, DTE)
                acc += accuracy(LP, LTE)
                min_cost += Bayes_risk_min_cost(0.5, 1, 1, s, LTE)
            print("Error rate %.3f" %(acc/K), "%")
            print("min cost: %.3f" %(min_cost/K))
            print()