import sys
sys.path.append('../')

import numpy
from Library_gianmarco import logreg_obj_wrapper, logreg, Bayes_risk_min_cost, Z_norm, PCA, Ksplit
from library import load
import scipy

def train(D, L, l):
    priors = [0.5, 0.1, 0.9]
    K = 5
    for p in priors:
        print("prior:", p)
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
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, orderedLabels))
        print("minDCF 0.9: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, orderedLabels))
        print()

if __name__ == '__main__':

    [D, L] = load('../Train.txt')
        
    print("LR RAW features lambda = 0.1")    
    train(D, L, l = 0.1)
    print()
    print("LR Z-Norm features lambda = 10**(-3)")
    train(Z_norm(D), L, l = 10**(-3))
    print()
    
    P = PCA(D, 12)
    DP = numpy.dot(P.T, D)
    print("LR PCA 12 RAW features lambda = 0.1",)    
    train(DP, L, l = 0.1)
    print()
    
    DZ = Z_norm(D)
    P = PCA(DZ, 12)
    DP = numpy.dot(P.T, DZ)
    print("LR PCA 12 Z-Norm features lambda = 10**(-3)")
    train(DP, L, l = 10**(-3))
    print()