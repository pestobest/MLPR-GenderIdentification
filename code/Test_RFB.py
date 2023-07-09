# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import RBF_kernel_SVM, Bayes_risk_min_cost, accuracy_v2,  PCA, Ksplit
from library import load
import scipy.optimize as scopt
from itertools import repeat

def LD_objectiveFunctionOfModifiedDualFormulation(alpha, H):
    grad = numpy.dot(H, alpha) - numpy.ones(H.shape[1])
    return ((1/2)*numpy.dot(numpy.dot(alpha.T, H), alpha)-numpy.dot(alpha.T, numpy.ones(H.shape[1])), grad)

def dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scopt.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    numpy.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    kernelFunction = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTE[:, j], gamma, K)
    S=numpy.sum(numpy.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunction), axis=0)
    # Compute the scores
    # S = np.sum(
    #     np.dot((x*LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DTE)+c)**d+ K), axis=0)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = numpy.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute dual loss
    dl = -f
    print("K=%d, C=%f, RBF (gamma=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, gamma, dl, errorRate))
    return errorRate


# def kernelPoly(DTR, LTR, DTE, LTE, K, C, d, c):
#     # Compute the H matrix exploiting broadcasting
#     kernelFunction = (numpy.dot(DTR.T, DTR)+c)**d+ K**2
#     # To compute zi*zj I need to reshape LTR as a matrix with one column/row
#     # and then do the dot product
#     zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
#     Hij = zizj*kernelFunction
#     # We want to maximize JD(alpha), but we can't use the same algorithm of the
#     # previous lab, so we can cast the problem as minimization of LD(alpha) defined
#     # as -JD(alpha)
#     dualLossErrorRatePoly(DTR, C, Hij, LTR, LTE, DTE, K, d, c)
#     return


def RBF(x1, x2, gamma, K):
    return numpy.exp(-gamma*(numpy.linalg.norm(x1-x2)**2))+K**2

def kernelRBF(DTR, LTR, DTE, LTE, K, C, gamma):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    return dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma)
    

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    print("RBF kernel SVM")   
    K_C_gamma = [[0, 1, 1], [0, 1, 10], [1, 1, 1], [1, 1, 10]]
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
                
                acc += kernelRBF(DTR, LTR, DTE, LTE, K, C, gamma)
            #     min_cost += Bayes_risk_min_cost(0.5, 1, 1, scores, LTE)
            print("Error rate %.1f" %(acc/K_fold), "%")
            # print("min cost:", min_cost/K_fold)
            # print()