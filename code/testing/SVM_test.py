import sys
sys.path.append('../')

import numpy
from library import load, dual_SVM, Bayes_risk_min_cost, Ksplit, Z_norm, vcol
import matplotlib.pyplot as plt


def train(D, L):
    C_vec = numpy.logspace(-5, 5, num=31)
    K = 1
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
            w = dual_SVM(DTR, LTR, K, C)
            s = numpy.dot(w.T, numpy.vstack((DTE, numpy.zeros(DTE.shape[1]) + K)))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost_05.append(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        min_cost_01.append(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        min_cost_09.append(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
    return min_cost_05, min_cost_01, min_cost_09

def test(DTR, LTR, DTE, LTE):
    C_vec = numpy.logspace(-5, 5, num=31)
    K = 1
    min_cost_05 = []
    min_cost_01 = []
    min_cost_09 = []
    for C in C_vec: 
        w = dual_SVM(DTR, LTR, K, C)
        scores = numpy.dot(w.T, numpy.vstack((DTE, numpy.zeros(DTE.shape[1]) + K)))
        min_cost_05.append(Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
        min_cost_01.append(Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        min_cost_09.append(Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
    return min_cost_05, min_cost_01, min_cost_09

def plot_minDCF_SVM_test(min_cost_05, min_cost_01, min_cost_09, min_cost_05_t, min_cost_01_t, min_cost_09_t, mode):
    fig = plt.figure()
    c_values = numpy.logspace(-5, 5, num=31)
    plt.plot(c_values, min_cost_05, label="minDCF($\\tilde{\pi} = 0.5$) [Val]", linestyle='dotted', color='b')
    plt.plot(c_values, min_cost_01, label="minDCF($\\tilde{\pi} = 0.1$) [Val]", linestyle='dotted', color='r')
    plt.plot(c_values, min_cost_09, label="minDCF($\\tilde{\pi} = 0.9$) [Val]", linestyle='dotted', color='g')

    plt.plot(c_values, min_cost_05_t, label="minDCF($\\tilde{\pi} = 0.5$) [Eval]", color='b')
    plt.plot(c_values, min_cost_01_t, label="minDCF($\\tilde{\pi} = 0.1$) [Eval]", color='r')
    plt.plot(c_values, min_cost_09_t, label="minDCF($\\tilde{\pi} = 0.9$) [Eval]", color='g')


    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlabel("C")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()
    plt.savefig('minDCF/svm_test_'+mode+'.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

if __name__ == '__main__':

    [D, L] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')  
        
    print("SVM RAW features")    
    min_cost_05, min_cost_01, min_cost_09 = train(D, L)
    min_cost_05_t, min_cost_01_t, min_cost_09_t = test(D, L, DTE, LTE)
    plot_minDCF_SVM_test(min_cost_05, min_cost_01, min_cost_09, min_cost_05_t, min_cost_01_t, min_cost_09_t, "Raw")
    
    print("SVM Z-Norm features")
    Dstd = vcol(numpy.std(D, axis=1))
    Dmean = vcol(D.mean(1))
    DTE = (DTE - Dmean) / Dstd
    min_cost_05, min_cost_01, min_cost_09 = train(Z_norm(D), L)
    min_cost_05_t, min_cost_01_t, min_cost_09_t = test(Z_norm(D), L, DTE, LTE)
    plot_minDCF_SVM_test(min_cost_05, min_cost_01, min_cost_09, min_cost_05_t, min_cost_01_t, min_cost_09_t, "Znorm")