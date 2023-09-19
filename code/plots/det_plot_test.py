import sys
sys.path.append('../')

import numpy
from Library_gianmarco import tied_covariance_gaussian_classier, logreg, LBG_algorithm_tied, logpdf_GMM, RBF_kernel_SVM, logreg_obj_wrapper
from library import load
import scipy
import matplotlib.pyplot as plt


def plot_det(llr_list, labels_list, file_name):
    colors = ["r", "b", "g", "y"]
    models = ["GMM", "RBSVM", "TMVG", "LR"]
    idx = 0
    for llr in llr_list:
        LLRs_sorted = numpy.concatenate([numpy.array([-numpy.inf]), numpy.sort(llr), numpy.array([numpy.inf])])
        FNR = []
        FPR = []
        labels = labels_list[idx]
        for i in range(labels.shape[0]):
            conf_matrix = numpy.zeros((2,2), numpy.int32)
            t = LLRs_sorted[i]
            cont_p = 0
            for j in range(labels.shape[0]):
                p = 0
                if llr[j] > t:
                    cont_p += 1
                    p = 1
                conf_matrix[p, labels[j]] += 1
            FNR.append(conf_matrix[0, 1]/(conf_matrix[0, 1] + conf_matrix[1, 1]))
            FPR.append((conf_matrix[1, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0])))
        plt.plot(FPR, FNR, color=colors[idx], label=models[idx])
        idx += 1
    
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig("det/" + file_name + '.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
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

    [DTR, LTR] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')
    
    sGMM = test_GMM(DTR, LTR, DTE, LTE)
    sRBSVM = test_SVM(DTR, LTR, DTE, LTE)
    sTMVG = test_TMVG(DTR, LTR, DTE, LTE)
    sLR = test_LR(DTR, LTR, DTE, LTE)
    plot_det([sGMM, sRBSVM, sTMVG, sLR], [LTE, LTE, LTE, LTE], "det_test")
    