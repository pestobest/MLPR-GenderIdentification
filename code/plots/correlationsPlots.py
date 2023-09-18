import numpy
from library import vcol, load
from Library_gianmarco import correlationsPlot

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
    
    print("RAW data")
    correlationsPlot(D, L)
    mu = vcol(D.mean(1))

    DC = D - mu

    C = 0
    dotDC = numpy.dot(D, D.T)
    C = (1 / float(D.shape[1])) * dotDC

    U, s, Vh = numpy.linalg.svd(C)

    print("Normalized data")
    P = U[:, 0:s.size]

    DP = numpy.dot(P.T, D)
    correlationsPlot(DP, L, 'normalized')
    