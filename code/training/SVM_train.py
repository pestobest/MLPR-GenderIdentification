import sys
sys.path.append('../')

import numpy
from library import load, dual_SVM, Bayes_risk_min_cost, PCA, Ksplit, Z_norm

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    C_K = [[0.1, 1], [1.0, 1], [10.0, 1], [0.1, 10], [1.0, 10], [10.0, 10]]
    priors = [0.5, 0.1, 0.9]
    
    print("SVM RAW features")
    for p in priors:
        print("Prior:", p)
        for C, K in C_K:
            print("C:", C, "K:", K)
            for m in range(10, 13):
                print("PCA:", m)
                P = PCA(D, m)
        
                DP = numpy.dot(P.T, D)
                
                K_fold = 5
                folds, labels = Ksplit(DP, L, seed=0, K=K_fold)
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
                min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                print("min cost: %.3f" %min_cost)
            print()
    
    D = Z_norm(D)
    print("SVM Z_norm features")
    for p in priors:
        print("Prior:", p)
        for C, K in C_K:
            print("C:", C, "K:", K)
            for m in range(10, 13):
                print("PCA:", m)
                P = PCA(D, m)
        
                DP = numpy.dot(P.T, D)
                
                K_fold = 5
                folds, labels = Ksplit(DP, L, seed=0, K=K_fold)
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
                min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
                print("min cost: %.3f" %min_cost)
            print()