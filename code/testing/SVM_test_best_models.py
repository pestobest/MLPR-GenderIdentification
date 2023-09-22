import sys
sys.path.append('../')

import numpy
from library import load, dual_SVM, RBF_kernel_SVM, polynomial_kernel_SVM, Bayes_risk_min_cost, vcol, Z_norm

def test(DTR, LTR, DTE, LTE, C):
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    K = 1
    for p in priors:
        print("prior:", p)
        w = dual_SVM(DTR, LTR, K, C, priors=p)
        scores = numpy.dot(w.T, numpy.vstack((DTE, numpy.zeros(DTE.shape[1]) + K)))
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        print("minDCF 0.9: %.3f" % Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
        print()
        
def test_RBF(DTR, LTR, DTE, LTE, C, gamma):
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    K = 1
    for p in priors:
        print("prior:", p)
        scores = RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE, priors=p)
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        print("minDCF 0.9: %.3f" % Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
        print()


def test_poly(DTR, LTR, DTE, LTE, C, c=1):
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    K = 1
    d=2
    for p in priors:
        print("prior:", p)
        scores = polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE, priors=p)
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        print("minDCF 0.9: %.3f" % Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
        print()

if __name__ == '__main__':

    [DTR, LTR] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')
    
    Dstd = vcol(numpy.std(DTR, axis=1))
    Dmean = vcol(DTR.mean(1))
    
    print("SVM RAW features")    
    test(DTR, LTR, DTE, LTE, C = 1)
    print()
    
    print("SVM RBF RAW features")    
    test_RBF(DTR, LTR, DTE, LTE, C = 10, gamma = 10**(-3))
    print()
    
    print("SVM Poly RAW features c=1")    
    test_poly(DTR, LTR, DTE, LTE, C = 10**(-3))
    print()
    
    print("SVM Poly RAW features c=0")    
    test_poly(DTR, LTR, DTE, LTE, C = 10**(-3), c=0)
    print()
    
    DTR = Z_norm(DTR)
    DTE = (DTE - Dmean) / Dstd
    print("SVM Z-Norm features")
    test(DTR, LTR, DTE, LTE, C = 10)
    print()
    
    print("SVM RBF Z-Norm features")
    test_RBF(DTR, LTR, DTE, LTE, C = 1, gamma = 10**(-1))
    print()
    
    print("SVM Poly Z-Norm features")
    test_poly(DTR, LTR, DTE, LTE, C = 10**(-1))
    print()