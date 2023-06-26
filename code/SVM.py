# SVM
import numpy
from library import vcol, load, split_db_2to1, dual_SVM, accuracy_v2 
from Library_gianmarco import Bayes_risk_min_cost

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    mu = vcol(D.mean(1))

    DC = D - mu

    C = 0
    dotDC = numpy.dot(D, D.T)
    C = (1 / float(D.shape[1])) * dotDC

    U, s, Vh = numpy.linalg.svd(C)

    m = 3
    num = 0
    den = 0

    for m in range(7, s.size + 1):
        P = U[:, 0:m]

        DP = numpy.dot(P.T, D)
        
        (DTR, LTR), (DTE, LTE) = split_db_2to1(DP, L)
        
        w = dual_SVM(DTR, LTR, 1, 1)
        scores = numpy.dot(w.T, numpy.vstack((DTE, numpy.zeros(DTE.shape[1]) + 1)))
        accuracy_v2(scores, LTE)
        print(Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))