import numpy
from library import load, tied_covariance_gaussian_classier, Bayes_risk_Bdummy, logreg_obj_wrap_prof, logreg, LBG_algorithm_tied, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, logpdf_GMM, RBF_kernel_SVM, logreg_obj_wrapper
import scipy
import matplotlib.pyplot as plt

def cal_GMM1(D, L):
    print("Calibration GMM tied 8 comp RAW")
    K = 5
    folds, labels = Ksplit(D, L, seed=0, K=K)
    
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
        
        EM_GMM_D0 =LBG_algorithm_tied(2, D0, GMM_D0)
        
        EM_GMM_D1 =LBG_algorithm_tied(2, D1, GMM_D1)
    
        _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
    
        _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
        s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
        scores.append(s)
        LTEs.append(LTE)
    scores = numpy.hstack(scores)
    orderedLabels = numpy.hstack(LTEs)
    print("No calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    plot_bayes_error(scores, "gmm tied", "8 components RAW", labels=orderedLabels)
    
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
    print("After calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "gmm tied", "8 components RAW calibrated", labels=orderedLabels)
    
def cal_GMM2(D, L):
    print("Calibration GMM tied 8 comp Z-norm PCA-12")
    K = 5
    D = Z_norm(D)
    P = PCA(D, 12)
    D = numpy.dot(P.T, D)
    folds, labels = Ksplit(D, L, seed=0, K=K)
    
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
        
        EM_GMM_D0 =LBG_algorithm_tied(2, D0, GMM_D0)
        
        EM_GMM_D1 =LBG_algorithm_tied(2, D1, GMM_D1)
    
        _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
    
        _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
        s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
        scores.append(s)
        LTEs.append(LTE)
    scores = numpy.hstack(scores)
    orderedLabels = numpy.hstack(LTEs)
    print("No calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    plot_bayes_error(scores, "gmm tied", "8 components PCA-12 - Z-Score", labels=orderedLabels)
    
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
    print("After calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "gmm tied", "8 components PCA-12 - Z-Score calibrated", labels=orderedLabels)
    
def cal_SVM(D, L):
    print("Calibration SVM RBF RAW features")
    K = 1
    p = [0.5, 0.5]
    K_fold = 5
    C = 10
    gamma = 10**(-3)
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
    print("No calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    plot_bayes_error(scores, "SVM RBF", "SVM RBF", labels=orderedLabels)
    
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
    print("After calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "SVM RBF", "SVM RBF calibrated", labels=orderedLabels)
     

def cal_TMVG(D, L):
    print("Calibration Tied MVG RAW features")
    K_fold = 5
    folds, labels = Ksplit(D, L, seed=0, K=K_fold)
    scores = []
    LTEs = []
    p = 0.5
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
        s, LP = tied_covariance_gaussian_classier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
        scores.append(s)
        LTEs.append(LTE)
    scores=numpy.hstack(scores)
    orderedLabels=numpy.hstack(LTEs)
    print("No calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    plot_bayes_error(scores, "TMVG", "Tied MVG", labels=orderedLabels)
    
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
    print("After calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "TMVG", "Tied MVG calibrated", labels=orderedLabels)

def cal_LR(D, L):
    print("Calibration LR lambda 0.1 p=0.9")
    K_fold = 5
    folds, labels = Ksplit(D, L, seed=0, K=K_fold)
    scores = []
    LTEs = []
    p = 0.9
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
        logreg_obj = logreg_obj_wrapper(DTR, LTR, 0.1, p)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                x0=numpy.zeros(DTR.shape[0] + 1),
                                                approx_grad=True)
        s, LP = logreg(x, DTE)
        scores.append(s)
        LTEs.append(LTE)
    scores = numpy.hstack(scores)
    orderedLabels = numpy.hstack(LTEs)
    print("No calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    plot_bayes_error(scores, "LR", "LR", labels=orderedLabels)
    
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
    print("After calibration")
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (orderedLabels == predicted_labels).sum() / orderedLabels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, orderedLabels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, orderedLabels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, orderedLabels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "TMVG", "LR calibrated", labels=orderedLabels)

def plot_bayes_error(LLRs, model: str, title: str, labels):

    effPriorLogOdds = numpy.linspace(-4, 4, 100)
    dcf = []
    mindcf = []
    for e in effPriorLogOdds:
        pi = 1/(1+numpy.exp(-e))
        dcf.append(Bayes_risk_Bdummy(pi, 1, 1, LLRs, labels))
        mindcf.append(Bayes_risk_min_cost(pi, 1, 1, LLRs, labels))
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', linestyle='dotted', color='r')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.ylabel("DCF")
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
    plt.legend()
    plt.savefig('cal/model'+title+'.jpg', dpi=300, bbox_inches='tight')
    plt.close()    
    

if __name__ == "__main__":


    [D, L] = load('Train.txt')
    cal_GMM1(D, L)
    cal_GMM2(D, L)
    cal_SVM(D, L)
    cal_TMVG(D, L)
    cal_LR(D, L)
    