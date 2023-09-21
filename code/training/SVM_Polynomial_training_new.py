import sys
sys.path.append('../')

import numpy
from library import load, polynomial_kernel_SVM, Bayes_risk_min_cost, Ksplit, Z_norm
import matplotlib.pyplot as plt


def train(D, L):
    C_vec = numpy.logspace(-5, 5, num=31)
    K = 1
    d=2
    c=1
    min_cost_05 = []
    min_cost_01 = []
    min_cost_09 = []
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
            s = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE)
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
    c_values = numpy.logspace(-5, 5, num=31)
    plt.plot(c_values, min_cost_05, label="minDCF($\\tilde{\pi} = 0.5$)", color='b')
    plt.plot(c_values, min_cost_01, label="minDCF($\\tilde{\pi} = 0.1$)", color='r')
    plt.plot(c_values, min_cost_09, label="minDCF($\\tilde{\pi} = 0.9$)", color='g')


    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlabel("C")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()
    plt.savefig('minDCF/svm_'+mode+'.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

if __name__ == '__main__':

    [D, L] = load('../Train.txt')
        
    # print("SVM RAW features")    
    # min_cost_05, min_cost_01, min_cost_09 = train(D, L)
    # plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, "Poly_Raw")
    
    print("SVM Z-Norm features")
    min_cost_05, min_cost_01, min_cost_09 = train(Z_norm(D), L)
    plot_minDCF_SVM(min_cost_05, min_cost_01, min_cost_09, "Poly_Znorm")