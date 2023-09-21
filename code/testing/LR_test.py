import sys
sys.path.append('../')

import numpy
from library import load, logreg_obj_wrapper, logreg, Bayes_risk_min_cost, Z_norm, Ksplit, vcol
import scipy
import matplotlib.pyplot as plt


def train(D, L):
    l_vec = numpy.logspace(-5, 5, 51)
    min_cost_05 = []
    min_cost_01 = []
    min_cost_09 = []
    K = 5
    p = 0.5
    for l in l_vec:
        scores = []
        LTEs = []
        folds, labels = Ksplit(D, L, seed=0, K=K)
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
            logreg_obj = logreg_obj_wrapper(DTR, LTR, l, p)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                    x0=numpy.zeros(DTR.shape[0] + 1),
                                                    approx_grad=True)
            s, LP = logreg(x, DTE)
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost_05.append(Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        min_cost_01.append(Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        min_cost_09.append(Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
    return min_cost_05, min_cost_01, min_cost_09

def test(DTR, LTR, DTE, LTE):
    l_vec = numpy.logspace(-5, 5, 51)
    min_cost_05 = []
    min_cost_01 = []
    min_cost_09 = []
    p = 0.5
    for l in l_vec:
        logreg_obj = logreg_obj_wrapper(DTR, LTR, l, p)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                x0=numpy.zeros(DTR.shape[0] + 1),
                                                approx_grad=True)
        scores, LP = logreg(x, DTE)
        min_cost_05.append(Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
        min_cost_01.append(Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        min_cost_09.append(Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
    return min_cost_05, min_cost_01, min_cost_09

def plot_minDCF_LR_test(min_cost_05, min_cost_01, min_cost_09, min_cost_05_t, min_cost_01_t, min_cost_09_t, mode):
    fig = plt.figure()
    l_values = numpy.logspace(-5, 5, num=51)
    plt.plot(l_values, min_cost_05, label="minDCF($\\tilde{\pi} = 0.5$) [Val]", linestyle='dotted', color='b')
    plt.plot(l_values, min_cost_01, label="minDCF($\\tilde{\pi} = 0.1$) [Val]", linestyle='dotted', color='r')
    plt.plot(l_values, min_cost_09, label="minDCF($\\tilde{\pi} = 0.9$) [Val]", linestyle='dotted', color='g')

    plt.plot(l_values, min_cost_05_t, label="minDCF($\\tilde{\pi} = 0.5$) [Eval]", color='b')
    plt.plot(l_values, min_cost_01_t, label="minDCF($\\tilde{\pi} = 0.1$) [Eval]", color='r')
    plt.plot(l_values, min_cost_09_t, label="minDCF($\\tilde{\pi} = 0.9$) [Eval]", color='g')

    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlabel("λ")
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig('minDCF/lr_test_'+mode+'.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

if __name__ == '__main__':

    [DTR, LTR] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')
    
    print("Logistic Regression RAW features")
    min_cost_05, min_cost_01, min_cost_09 = train(DTR, LTR)
    min_cost_05_t, min_cost_01_t, min_cost_09_t = test(DTR, LTR, DTE, LTE)
    plot_minDCF_LR_test(min_cost_05, min_cost_01, min_cost_09, min_cost_05_t, min_cost_01_t, min_cost_09_t, "Raw")
    
    print("Logistic Regression Z-norm features")
    Dstd = vcol(numpy.std(DTR, axis=1))
    Dmean = vcol(DTR.mean(1))
    DTE = (DTE - Dmean) / Dstd
    min_cost_05, min_cost_01, min_cost_09 = train(Z_norm(DTR), LTR)
    min_cost_05_t, min_cost_01_t, min_cost_09_t = test(Z_norm(DTR), LTR, DTE, LTE)
    plot_minDCF_LR_test(min_cost_05, min_cost_01, min_cost_09, min_cost_05_t, min_cost_01_t, min_cost_09_t, "Z_norm")