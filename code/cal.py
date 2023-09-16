# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:03:27 2023

@author: gianm
"""
import numpy
from Library_gianmarco import Bayes_risk_Bdummy, Bayes_risk_min_cost, logreg_obj_wrap_prof, logreg, LBG_algorithm, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, logpdf_GMM, plot_min_cdf_error_gaussian_mixture_models
from library import load
import scipy


import matplotlib.pyplot as plt

# def build_conf_mat_uniform(prediction, L):
#     conf_mat = numpy.zeros((2, 2))
#     for i in range(2):
#         for j in range(2):
#             conf_mat[i][j] = (1 * numpy.bitwise_and(prediction == i, L == j)).sum()

#     return conf_mat

# def build_conf_mat(llr: numpy.ndarray,L: numpy.ndarray,pi:float, C_fn:float,C_fp:float):
#     t = -numpy.log(pi*C_fn/((1-pi)*C_fp))
#     predictions = 1*(llr > t)
#     return build_conf_mat_uniform(predictions,L)

# def compute_DCF(llr: numpy.ndarray, L: numpy.ndarray, pi: float, C_fn: float, C_fp: float):
#     conf_mat = build_conf_mat(llr, L, pi, C_fn, C_fp)
#     FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
#     FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
#     return pi * C_fn * FNR + (1-pi) * C_fp * FPR

# def compute_NDCF(llr: numpy.ndarray, L: numpy.ndarray, pi: float, C_fn: float, C_fp: float):
#     return compute_DCF(llr, L, pi, C_fn, C_fp) / min([pi*C_fn, (1-pi)*C_fp])

# def compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn):
#     FNR = conf_mat[0][1] / (conf_mat[0][1] + conf_mat[1][1])
#     FPR = conf_mat[1][0] / (conf_mat[1][0] + conf_mat[0][0])
#     return (pi * C_fn * FNR + (1-pi) * C_fp * FPR) / min([pi * C_fn, (1-pi) * C_fp])

# def compute_minimum_NDCF(llr, L, pi, C_fp, C_fn):
#     llr = llr.ravel()
#     tresholds = numpy.concatenate([numpy.array([-numpy.inf]), numpy.sort(llr), numpy.array([numpy.inf])])
#     DCF = numpy.zeros(tresholds.shape[0])
#     for (idx, t) in enumerate(tresholds):
#         pred = 1 * (llr > t)
#         conf_mat = build_conf_mat_uniform(pred, L)
#         DCF[idx] = compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn)
#     argmin = DCF.argmin()
#     return DCF[argmin], tresholds[argmin]

def plot_bayes_error(LLRs, model: str, title: str, labels=None, is_train=True):

    effPriorLogOdds = numpy.linspace(-4, 4, 100)
    dcf = []
    mindcf = []
    for e in effPriorLogOdds:
        pi = 1/(1+numpy.exp(-e))
        dcf.append(Bayes_risk_Bdummy(pi, 1, 1, LLRs, labels))
        mindcf.append(Bayes_risk_min_cost(pi, 1, 1, LLRs, labels))
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.ylabel("DCF")
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":


    [D, L] = load('../Train.txt')
    
    K = 5
    folds, labels = Ksplit(Z_norm(D), L, seed=0, K=K)
    delta = 10**-6
    
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
        D0 = DTR[:, LTR==0]
        D1 = DTR[:, LTR==1]
        mu_D0=D0.mean(1).reshape(12,1)
            
        mu_D1=D1.mean(1).reshape(12,1)
            
    
        n0=(LTR==0).sum()
        n1=(LTR==1).sum()
    
        cov_D0=((D0-mu_D0).dot((D0-mu_D0).T))/n0
    
        cov_D1=((D1-mu_D1).dot((D1-mu_D1).T))/n1
    
        l,r=numpy.shape(DTR)
    
        GMM_D0 = [[1, mu_D0, cov_D0]]
    
        GMM_D1 = [[1, mu_D1, cov_D1]]
        
        EM_GMM_D0 =LBG_algorithm(2, D0, GMM_D0)
        
        EM_GMM_D1 =LBG_algorithm(2, D1, GMM_D1)
    
        _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
    
        _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
        s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
        scores.append(s)
        LTEs.append(LTE)
    scores = numpy.hstack(scores)
    orderedLabels = numpy.hstack(LTEs)
    print("graphs")
    plot_bayes_error(scores, "gmm", "4 components - Z-Score calibrated", labels=orderedLabels)
    
    K = 5
    folds, labels = Ksplit(numpy.array([scores]), orderedLabels, seed=0, K=K)
    
    p = [0.5, 0.5]
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
        logreg_obj = logreg_obj_wrap_prof(DTR, LTR, 0)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                x0=numpy.zeros(DTR.shape[0] + 1),
                                                approx_grad=True)
        s, LP = logreg(x, DTE)
        scores.append(s)
        LTEs.append(LTE)
    scores = numpy.hstack(scores)
    orderedLabels = numpy.hstack(LTEs)
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    # print("Error rate for this training is " + str(round(err, 2)) + "%" )
    # cost_0_5 = str(round(compute_minimum_NDCF(scores, orderedLabels, 0.5, 1, 1)[0], 3))
    # cost_0_1 = str(round(compute_minimum_NDCF(scores, orderedLabels, 0.1, 1, 1)[0], 3))
    # cost_0_9 = str(round(compute_minimum_NDCF(scores, orderedLabels, 0.9, 1, 1)[0], 3))
    # print("minDCF with π=0.5 " +  cost_0_5)
    # print("minDCF with π=0.1 " + cost_0_1)
    # print("minDCF with π=0.9 " + cost_0_9)
    # cost_0_5_cal = str(round(compute_NDCF(scores, orderedLabels, 0.5, 1, 1), 3))
    # cost_0_1_cal = str(round(compute_NDCF(scores, orderedLabels, 0.1, 1, 1), 3))
    # cost_0_9_cal = str(round(compute_NDCF(scores, orderedLabels, 0.9, 1, 1), 3))
    # print("actDCF with π=0.5 " + cost_0_5_cal)
    # print("actDCF with π=0.1 " + cost_0_1_cal)
    # print("actDCF with π=0.9 " + cost_0_9_cal)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "gmm", "4 components - Z-Score calibrated", labels=orderedLabels)
    
    