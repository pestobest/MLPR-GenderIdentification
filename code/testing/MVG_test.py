import sys
sys.path.append('../')

import numpy
from library import load, multivariete_gaussian_classifier, Bayes_risk_min_cost, PCA, Z_norm, vcol

def test(DTR, LTR, DTE, LTE):
    scores, _ = multivariete_gaussian_classifier(DTR, LTR, DTE, LTE, numpy.vstack([0.5, 0.5]))
    print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
    print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
    print("minDCF 0.9: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
    print()

if __name__ == '__main__':

    [DTR, LTR] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')
    
    Dz = Z_norm(DTR)
    Dstd = vcol(numpy.std(DTR, axis=1))
    Dmean = vcol(DTR.mean(1))
    DTEz = (DTE - Dmean) / Dstd
    
    print("Multivariate Covariance model")

    print("RAW features")
    test(DTR, LTR, DTE, LTE)

    print("Z-norm")
    test(Dz, LTR, DTEz, LTE)

    for m in range(11, 13):
        print("PCA:", m)
        P = PCA(DTR, m)

        DP = numpy.dot(P.T, DTR)
        DTE_P = numpy.dot(P.T, DTE)

        print("RAW features")
        test(DP, LTR, DTE_P, LTE)

    for m in range(11, 13):
        print("PCA:", m)
        P = PCA(Dz, m)

        DP = numpy.dot(P.T, Dz)
        DTE_P = numpy.dot(P.T, DTEz)

        print("Z-norm")
        test(DP, LTR, DTE_P, LTE)