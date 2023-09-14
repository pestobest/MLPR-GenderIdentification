# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:03:27 2023

@author: gianm
"""
import numpy
from Library_gianmarco import LBG_algorithm_tied, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, logpdf_GMM, plot_min_cdf_error_gaussian_mixture_models
from library import load

if __name__ == "__main__":


    [D, L] = load('../Train.txt')
    
    K = 5
    folds, labels = Ksplit(D, L, seed=0, K=K)
    delta = 10**-6
    
    """min_cost_vec = []
    for it in range(7):
        print("num iterations:", 2**it)
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
            
            EM_GMM_D0 =LBG_algorithm_tied(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_tied(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()
    
    print("Z-norm")
    folds, labels = Ksplit(Z_norm(D), L, seed=0, K=K)
    delta = 10**-6
    
    min_cost_vec_znorm = []
    for it in range(7):
        print("num iterations:", 2**it)
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
            
            EM_GMM_D0 =LBG_algorithm_tied(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_tied(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec_znorm.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()
    
    plot_min_cdf_error_gaussian_mixture_models("GMM tied", min_cost_vec, min_cost_vec_znorm, "RAW", "Z-Norm")
    
    print("PCA 12")
    
    P = PCA(D, 12)

    DP = numpy.dot(P.T, D)
    folds, labels = Ksplit(DP, L, seed=0, K=K)
    delta = 10**-6
    
    min_cost_vec = []
    for it in range(7):
        print("num iterations:", 2**it)
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
            
            EM_GMM_D0 =LBG_algorithm_tied(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_tied(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()
    
    print("Z-norm")
    folds, labels = Ksplit(Z_norm(DP), L, seed=0, K=K)
    delta = 10**-6
    
    min_cost_vec_znorm = []
    for it in range(7):
        print("num iterations:", 2**it)
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
            
            EM_GMM_D0 =LBG_algorithm_tied(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_tied(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec_znorm.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()
    
    plot_min_cdf_error_gaussian_mixture_models("GMM tied PCA-12", min_cost_vec, min_cost_vec_znorm, "RAW", "Z-Norm")
"""
    
    print("PCA 11")
    
    P = PCA(D, 11)

    DP = numpy.dot(P.T, D)
    folds, labels = Ksplit(DP, L, seed=0, K=K)
    delta = 10**-6
    
    min_cost_vec = []
    for it in range(7):
        print("num iterations:", 2**it)
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
            mu_D0=D0.mean(1).reshape(11,1)
                
            mu_D1=D1.mean(1).reshape(11,1)
                
        
            n0=(LTR==0).sum()
            n1=(LTR==1).sum()
        
            cov_D0=((D0-mu_D0).dot((D0-mu_D0).T))/n0
        
            cov_D1=((D1-mu_D1).dot((D1-mu_D1).T))/n1
        
            l,r=numpy.shape(DTR)
        
            GMM_D0 = [[1, mu_D0, cov_D0]]
        
            GMM_D1 = [[1, mu_D1, cov_D1]]
            
            EM_GMM_D0 =LBG_algorithm_tied(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_tied(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()
    
    print("Z-norm")
    folds, labels = Ksplit(Z_norm(DP), L, seed=0, K=K)
    delta = 10**-6
    
    min_cost_vec_znorm = []
    for it in range(7):
        print("num iterations:", 2**it)
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
            mu_D0=D0.mean(1).reshape(11,1)
                
            mu_D1=D1.mean(1).reshape(11,1)
                
        
            n0=(LTR==0).sum()
            n1=(LTR==1).sum()
        
            cov_D0=((D0-mu_D0).dot((D0-mu_D0).T))/n0
        
            cov_D1=((D1-mu_D1).dot((D1-mu_D1).T))/n1
        
            l,r=numpy.shape(DTR)
        
            GMM_D0 = [[1, mu_D0, cov_D0]]
        
            GMM_D1 = [[1, mu_D1, cov_D1]]
            
            EM_GMM_D0 =LBG_algorithm_tied(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_tied(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec_znorm.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()
    
    plot_min_cdf_error_gaussian_mixture_models("GMM tied PCA-11", min_cost_vec, min_cost_vec_znorm, "RAW", "Z-Norm")
