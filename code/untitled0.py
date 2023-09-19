# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 05:53:28 2023

@author: gianm
"""
import matplotlib.pyplot as plt
import numpy
import scipy
from Library_gianmarco import tied_covariance_gaussian_classier, logreg, LBG_algorithm_tied, Ksplit, logpdf_GMM, RBF_kernel_SVM, logreg_obj_wrapper
from library import load

def plot_det(llr: list, L: numpy.array, labels: list, file_name: str):
    colors = ["r", "b", "g", "y"]
    for (idx, scores) in enumerate(llr):
        fpr, tpr = compute_det_points(scores, L)
        plt.plot(fpr, tpr, color=colors[idx], label=labels[idx])

    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
def compute_det_points(llr, L):
    threshold = numpy.concatenate([numpy.array([-numpy.inf]), numpy.sort(llr), numpy.array([numpy.inf])])
    FNR_points = numpy.zeros(L.shape[0] + 2)
    FPR_points = numpy.zeros(L.shape[0] + 2)
    for (idx, t) in enumerate(threshold):
        pred = 1 * (llr > t)
        FNR = 1 - (numpy.bitwise_and(pred == 1, L == 1).sum() / (L == 1).sum())
        FPR = numpy.bitwise_and(pred == 1, L == 0).sum() / (L == 1).sum()
        FNR_points[idx] = FNR
        FPR_points[idx] = FPR
    print(FNR_points)
    return FNR_points, FPR_points

def test_GMM(DTR, LTR, DTE, LTE):
    print("GMM tied 8 comp RAW")
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
    
    EM_GMM_D0 =LBG_algorithm_tied(2, D0, GMM_D0)
    
    EM_GMM_D1 =LBG_algorithm_tied(2, D1, GMM_D1)

    _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)

    _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
    scores=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
    return scores
        
def test_SVM(DTR, LTR, DTE, LTE):
    print("SVM RBF RAW features")
    K = 1
    p = [0.5, 0.5]
    C = 10
    gamma = 10**(-3)
    scores = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE, priors=p)
    return scores
     

def test_TMVG(DTR, LTR, DTE, LTE):
    print("Tied MVG RAW features")
    p = 0.5
    scores, LP = tied_covariance_gaussian_classier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
    return scores

def test_LR(DTR, LTR, DTE, LTE):
    print("LR lambda 0.1 p=0.9")
    p = 0.9
    logreg_obj = logreg_obj_wrapper(DTR, LTR, 0.1, p)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                            approx_grad=True)
    scores, LP = logreg(x, DTE)
    return scores

if __name__ == "__main__":

    [DTR, LTR] = load('Train.txt')
    [DTE, LTE] = load('Test.txt')
    
    sGMM = test_GMM(DTR, LTR, DTE, LTE)
    sRBSVM = test_SVM(DTR, LTR, DTE, LTE)
    sTMVG = test_TMVG(DTR, LTR, DTE, LTE)
    sLR = test_LR(DTR, LTR, DTE, LTE)
    plot_det([sGMM, sRBSVM, sTMVG, sLR],LTE, ["LTE", "LTE", "LTE", "LTE"], "det_test")
    