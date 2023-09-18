import sys
sys.path.append('../')

import numpy
from Library_gianmarco import RBF_kernel_SVM, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, vcol
from library import load
import matplotlib.pyplot as plt


def train(D, L, gamma):
    C_vec = numpy.logspace(-5, 5, num=31)
    K = 1
    min_cost = []
    for C in C_vec: 
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
            s = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE)
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost.append(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
    return min_cost

def test(DTR, LTR, DTE, LTE, gamma):
    C_vec = numpy.logspace(-5, 5, num=31)
    K = 1
    min_cost = []
    for C in C_vec: 
        scores = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE)
        min_cost.append(Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
    return min_cost

def plot_minDCF_SVM_test(scores, scores_t, mode):
    fig = plt.figure()
    c_values = numpy.logspace(-5, 5, num=31)
    plt.plot(c_values, scores[0], label="$log\gamma = -1$ [Val]", linestyle='dotted', color='b')
    plt.plot(c_values, scores[1], label="$log\gamma = -2$ [Val]", linestyle='dotted', color='r')
    plt.plot(c_values, scores[2], label="$log\gamma = -3$ [Val]", linestyle='dotted', color='g')
    
    plt.plot(c_values, scores_t[0], label="$log\gamma = -1$ [Eval]", color='b')
    plt.plot(c_values, scores_t[1], label="$log\gamma = -2$ [Eval]", color='r')
    plt.plot(c_values, scores_t[2], label="$log\gamma = -3$ [Eval]", color='g')
    
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()
    plt.savefig('minDCF/svm_test_'+mode+'.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

if __name__ == '__main__':
    [D, L] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')    
    print("SVM RBF RAW features")
    
    vec_gamma = [10**(-1), 10**(-2), 10**(-3)]
    scores = []
    scores_t = []
    for gamma in vec_gamma:
        scores.append(train(D, L, gamma))
        scores_t.append(test(D, L, DTE, LTE, gamma))
    plot_minDCF_SVM_test(scores, scores_t, "RBF_Raw")
    print("Z-Norm features")
    Dstd = vcol(numpy.std(D, axis=1))
    Dmean = vcol(D.mean(1))
    DTE = (DTE - Dmean) / Dstd
    scores = []
    scores_t = []
    for gamma in vec_gamma:
        scores.append(train(Z_norm(D), L, gamma))
        scores_t.append(test(Z_norm(D), L, DTE, LTE, gamma))
    plot_minDCF_SVM_test(scores, scores_t, "RBF_Z_norm")
    