# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:04:52 2023

@author: gianm
"""

import numpy
from Library_gianmarco import logreg_obj_wrapper, logreg, Bayes_risk_min_cost, Z_norm, PCA, Ksplit, plot_minDCF_lr
from library import load
import scipy
import matplotlib.pyplot as plt


def train(D, L):
    l_vec = numpy.logspace(-5, 5, 51)
    min_cost_05 = []
    min_cost_01 = []
    min_cost_09 = []
    K = 5
    p = 0.5
    for l in l_vec:
        scores = []
        LTEs = []
        folds, labels = Ksplit(D, L, seed=0, K=K)
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
        min_cost_05.append(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        min_cost_01.append(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        min_cost_09.append(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
    return min_cost_05, min_cost_01, min_cost_09

def plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, mode):
    fig = plt.figure()
    l_values = numpy.logspace(-5, 5, num=51)
    plt.plot(l_values, min_cost_05, label="minDCF($\\tilde{\pi} = 0.5$)", color='b')
    plt.plot(l_values, min_cost_01, label="minDCF($\\tilde{\pi} = 0.1$)", color='r')
    plt.plot(l_values, min_cost_09, label="minDCF($\\tilde{\pi} = 0.9$)", color='g')

    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlabel("λ")
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig('minDCF/lr_'+mode+'.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

if __name__ == '__main__':

    [D, L] = load('../Train.txt')
    # print("Logistic Regression RAW features")
    # min_cost_05, min_cost_01, min_cost_09 = train(D, L)
    # plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, "Raw")
    
    # print("Logistic Regression Z-norm features")
    # min_cost_05, min_cost_01, min_cost_09 = train(Z_norm(D), L)
    # plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, "Z_norm")
    
    # P = PCA(D, 12)
    # DP = numpy.dot(P.T, D)
    
    # print("Logistic Regression PCA 12 RAW features")
    # min_cost_05, min_cost_01, min_cost_09 = train(DP, L)
    # plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, "Raw_PCA12")
    
    DZ = Z_norm(D)
    P = PCA(DZ, 12)
    DP = numpy.dot(P.T, DZ)
    print("Logistic Regression PCA 12 Z-norm features")
    min_cost_05, min_cost_01, min_cost_09 = train(DP, L)
    plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, "Z_norm_PCA12")