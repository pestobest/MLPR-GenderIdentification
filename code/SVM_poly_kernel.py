# SVM (polynomial kernel)
import numpy
from library import vcol, load, split_db_2to1, polynomial_kernel_SVM, accuracy_v2 

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

        s = polynomial_kernel_SVM(DTR, LTR, 1, 1, 2, 1, LTE, DTE)
        accuracy_v2(s, LTE)