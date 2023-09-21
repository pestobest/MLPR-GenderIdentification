import sys
sys.path.append('../')

import numpy
from library import load, RBF_kernel_SVM, Bayes_risk_min_cost, Ksplit, Z_norm
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

def plot_minDCF_SVM(scores, mode):
    fig = plt.figure()
    c_values = numpy.logspace(-5, 5, num=31)
    plt.plot(c_values, scores[0], label="$log\gamma = -1$", color='b')
    plt.plot(c_values, scores[1], label="$log\gamma = -2$", color='r')
    plt.plot(c_values, scores[2], label="$log\gamma = -3$", color='g')
    
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()
    plt.savefig('minDCF/svm_'+mode+'.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

if __name__ == '__main__':
    [D, L] = load('../Train.txt')
        
    print("SVM RAW features")
    
    vec_gamma = [10**(-1), 10**(-2), 10**(-3)]
    scores = []
    for gamma in vec_gamma:
        scores.append(train(D, L, gamma))
    plot_minDCF_SVM(scores, "RBF_Raw")
    print("Z-Norm features")
    scores = []
    for gamma in vec_gamma:
        scores.append(train(Z_norm(D), L, gamma))
    plot_minDCF_SVM(scores, "RBF_Znorm")
    