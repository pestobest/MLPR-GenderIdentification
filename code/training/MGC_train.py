import sys
sys.path.append('../')

import numpy
from Library_gianmarco import multivariete_gaussian_classifier, Bayes_risk_min_cost, PCA, Ksplit, Z_norm
from library import load


if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')

    print("Multivariate Gaussian model RAW features")
    priors = [0.5, 0.1, 0.9]
    for p in priors:
        print("Prior:", p)
        for m in range(7, 13):
            print("PCA:", m)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K)
            
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
                s, LP = multivariete_gaussian_classifier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
                scores.append(s)
                LTEs.append(LTE)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(LTEs)
            min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
            print("min cost: %.3f" %min_cost)

        print("RAW")
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
            s, LP = multivariete_gaussian_classifier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
        print("min cost: %.3f" %min_cost)

        print()
            
    D = Z_norm(D)

    print("Multivariate Gaussian model Z-norm features")
    for p in priors:
        print("Prior:", p)
        for m in range(7, 13):
            print("PCA:", m)
            P = PCA(D, m)
    
            DP = numpy.dot(P.T, D)
            
            K = 5
            folds, labels = Ksplit(DP, L, seed=0, K=K)
            acc = 0
            min_cost = 0
            
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
                s, LP = multivariete_gaussian_classifier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
                scores.append(s)
                LTEs.append(LTE)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(LTEs)
            min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
            print("min cost: %.3f" %min_cost)

        print("RAW")
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
            s, LP = multivariete_gaussian_classifier(DTR, LTR, DTE, LTE, numpy.vstack([p, 1 - p]))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(p, 1, 1, scores, orderedLabels)
        print("min cost: %.3f" %min_cost)
        
        print()