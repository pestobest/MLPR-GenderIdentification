# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import dual_SVM, RBF_kernel_SVM, polynomial_kernel_SVM, Bayes_risk_min_cost, Ksplit, Z_norm
from library import load

def train(D, L, C):
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    K = 1
    for p in priors:
        print("prior:", p)
        K_fold = 5
        folds, labels = Ksplit(D, L, seed=0, K=K_fold)
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
            w = dual_SVM(DTR, LTR, K, C, priors=p)
            s = numpy.dot(w.T, numpy.vstack((DTE, numpy.zeros(DTE.shape[1]) + K)))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("minDCF 0.9: %.3f" % Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()

def train_RBF(D, L, C, gamma):
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    K = 1
    for p in priors:
        print("prior:", p)
        K_fold = 5
        folds, labels = Ksplit(D, L, seed=0, K=K_fold)
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
            s = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE, priors=p)
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("minDCF 0.9: %.3f" % Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()

def train_poly(D, L, C):
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    K = 1
    d=2
    c=1
    for p in priors:
        print("prior:", p) 
        K_fold = 5
        folds, labels = Ksplit(D, L, seed=0, K=K_fold)
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
            s = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE, priors=p)
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("minDCF 0.9: %.3f" % Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()

if __name__ == '__main__':

    [D, L] = load('../Train.txt')
        
    print("SVM RAW features")    
    train(D, L, C = 1)
    print()
    print("SVM Z-Norm features")
    train(Z_norm(D), L, C = 10)
    print()
    
    print("SVM RBF RAW features")    
    train_RBF(D, L, C = 10, gamma = 10**(-3))
    print()
    print("SVM RBF Z-Norm features")
    train_RBF(Z_norm(D), L, C = 1, gamma = 10**(-1))
    print()
    
    print("SVM Poly RAW features")    
    train_poly(D, L, C = 10**(-3))
    print()
    print("SVM Poly Z-Norm features")
    train_poly(Z_norm(D), L, C = 10**(-1))
    print()