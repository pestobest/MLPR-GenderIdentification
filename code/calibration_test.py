import numpy
from library import load, tied_covariance_gaussian_classier, Bayes_risk_Bdummy, logreg_obj_wrap_prof, logreg, LBG_algorithm_tied, Bayes_risk_min_cost, Ksplit, logpdf_GMM, RBF_kernel_SVM, logreg_obj_wrapper
import scipy
import matplotlib.pyplot as plt

def compute_min_act(scores, labels):
    predicted_labels = numpy.where(scores > 0, 1, 0)
    err = (1 - (labels == predicted_labels).sum() / labels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%" )
    cost_0_5 = str(round(Bayes_risk_min_cost(0.5, 1, 1, scores, labels), 3))
    cost_0_1 = str(round(Bayes_risk_min_cost(0.1, 1, 1, scores, labels), 3))
    cost_0_9 = str(round(Bayes_risk_min_cost(0.9, 1, 1, scores, labels), 3))
    print("minDCF with π=0.5 " +  cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)
    cost_0_5_cal = str(round(Bayes_risk_Bdummy(0.5, 1, 1, scores, labels), 3))
    cost_0_1_cal = str(round(Bayes_risk_Bdummy(0.1, 1, 1, scores, labels), 3))
    cost_0_9_cal = str(round(Bayes_risk_Bdummy(0.9, 1, 1, scores, labels), 3))
    print("actDCF with π=0.5 " + cost_0_5_cal)
    print("actDCF with π=0.1 " + cost_0_1_cal)
    print("actDCF with π=0.9 " + cost_0_9_cal)
    
def train_GMM1(D, L):
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
    return numpy.array([scores]), orderedLabels
    
def cal_GMM1(DTR, LTR, DTE, LTE):
    print("Calibration GMM tied 8 comp RAW")
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
    print("No calibration")
    compute_min_act(scores, LTE)
    plot_bayes_error(scores, "gmm tied", "8 components RAW", labels=LTE)
    
    DTE = numpy.array([scores])
    DTR, LTR = train_GMM1(DTR, LTR)
    
    p = [0.5, 0.5]
    logreg_obj = logreg_obj_wrap_prof(DTR, LTR, 0)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                            approx_grad=True)
    scores, LP = logreg(x, DTE)
    print("After calibration")
    compute_min_act(scores, LTE)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "gmm tied", "8 components RAW calibrated", labels=LTE)
    
def train_SVM(D, L):
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
    return numpy.array([scores]), orderedLabels

def cal_SVM(DTR, LTR, DTE, LTE):
    print("Calibration SVM RBF RAW features")
    K = 1
    p = [0.5, 0.5]
    C = 10
    gamma = 10**(-3)
    scores = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE, priors=p)
    print("No calibration")
    compute_min_act(scores, LTE)
    plot_bayes_error(scores, "SVM RBF", "SVM RBF", labels=LTE)
    
    DTE = numpy.array([scores])
    DTR, LTR = train_SVM(DTR, LTR)
    
    p = [0.5, 0.5]
    logreg_obj = logreg_obj_wrap_prof(DTR, LTR, 0)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                            approx_grad=True)
    scores, LP = logreg(x, DTE)
    print("After calibration")
    compute_min_act(scores, LTE)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "SVM RBF", "SVM RBF calibrated", labels=LTE)
     
def train_TMVG(D, L):
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
    return numpy.array([scores]), orderedLabels

def cal_TMVG(DTR, LTR, DTE, LTE):
    print("Calibration Tied MVG RAW features")
    p = 0.5
    scores, LP = tied_covariance_gaussian_classier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
    print("No calibration")
    compute_min_act(scores, LTE)
    plot_bayes_error(scores, "TMVG", "Tied MVG", labels=LTE)
    
    DTE = numpy.array([scores])
    DTR, LTR = train_TMVG(DTR, LTR)
    
    p = [0.5, 0.5]
    logreg_obj = logreg_obj_wrap_prof(DTR, LTR, 0)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                            approx_grad=True)
    scores, LP = logreg(x, DTE)
    print("After calibration")
    compute_min_act(scores, LTE)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "TMVG", "Tied MVG calibrated", labels=LTE)

def train_LR(D, L):
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
    return numpy.array([scores]), orderedLabels
    
def cal_LR(DTR, LTR, DTE, LTE):
    print("Calibration LR lambda 0.1 p=0.9")
    p = 0.9
    logreg_obj = logreg_obj_wrapper(DTR, LTR, 0.1, p)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                            approx_grad=True)
    scores, LP = logreg(x, DTE)
    print("No calibration")
    compute_min_act(scores, LTE)
    plot_bayes_error(scores, "LR", "LR", labels=LTE)
    DTE = numpy.array([scores])
    DTR, LTR = train_LR(DTR, LTR)
    
    p = [0.5, 0.5]
    logreg_obj = logreg_obj_wrap_prof(DTR, LTR, 0)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                            x0=numpy.zeros(DTR.shape[0] + 1),
                                            approx_grad=True)
    scores, LP = logreg(x, DTE)
    print("After calibration")
    compute_min_act(scores, LTE)
    calibrated_score = scores - numpy.log(p[1] / p[0])
    plot_bayes_error(calibrated_score, "TMVG", "LR calibrated", labels=LTE)

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
    plt.savefig('cal/test_model'+title+'.jpg', dpi=300, bbox_inches='tight')
    plt.close()    
    

if __name__ == "__main__":


    [DTR, LTR] = load('Train.txt')
    [DTE, LTE] = load('Test.txt')
    
    #cal_GMM1(DTR, LTR, DTE, LTE)
    #cal_SVM(DTR, LTR, DTE, LTE)
    cal_TMVG(DTR, LTR, DTE, LTE)
    cal_LR(DTR, LTR, DTE, LTE)
    