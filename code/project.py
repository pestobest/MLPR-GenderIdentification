import numpy
import matplotlib.pyplot as plot
import scipy

def vcol(vect):
    return vect.reshape((vect.size, 1))

def load(fname):
    dataList = []
    labelsList = []
    with open(fname, mode = 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.strip().split(',')
            data = numpy.array(values[:12], dtype=numpy.float32).reshape((12, 1))
            label = int(values[12])
            dataList.append(data)
            labelsList.append(label)

    return numpy.hstack(dataList), numpy.array(labelsList, dtype=numpy.int32)

def hist(D, L, spath):

    M0 = (L==0)
    M1 = (L==1)

    D0 = D[:, M0]
    D1 = D[:, M1]

    for dIdx in range(12):
        plot.figure()
        plot.hist(D0[dIdx, :], bins = 30, density = True, alpha = 0.4, label = 'Male')
        plot.hist(D1[dIdx, :], bins = 30, density = True, alpha = 0.4, label = 'Female')
        
        plot.legend()
        plot.tight_layout()
        plot.savefig('%shist_%d.pdf' % (spath, dIdx))

def scatter(D, L, spath, m):

    M0 = (L==0)
    M1 = (L==1)

    D0 = D[:, M0]
    D1 = D[:, M1]
    
    for dIdx1 in range(m):
        for dIdx2 in range(m):
            if dIdx1 == dIdx2:
                    continue
            plot.figure()
            plot.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Male')
            plot.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Female')

            plot.legend()
            plot.tight_layout()
            plot.savefig('%sscatter_%d_%d.pdf' % (spath, dIdx1, dIdx2))

def computeCovarianceMatrices (D, L):
    
    Sw = 0
    Sb = 0
    mu = vcol(D.mean(1))

    for i in range(2):
        D_c = D[:, L==i]
        mu_c = vcol(D_c.mean(1))
        D_cC = D_c - mu_c
        Sb += D_cC.shape[1] * numpy.dot(mu_c - mu, numpy.transpose(mu_c - mu))
        Sw += numpy.dot(D_cC, numpy.transpose(D_cC))

    Sb = (1 / float(D.shape[1])) * Sb
    Sw = (1 / float(D.shape[1])) * Sw

    return Sb, Sw

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')

    #hist(D, L, './initialGraphs/')
    #scatter(D, L, './initialGraphs/')
    
    mu = vcol(D.mean(1))

    DC = D - mu

    hist(DC, L, './modifiedGraphs/')

    """C = 0
    dotDC = numpy.dot(DC, numpy.transpose(DC))
    C = (1 / float(DC.shape[1])) * dotDC

    U, s, Vh = numpy.linalg.svd(C)

    m = 3
    num = 0
    den = 0

    for m in range(2, s.size):
        P = U[:, 0:m]

        DP = numpy.dot(P.T, D)
        
        scatter(DP, L, "./modifiedGraphsPCA_%s/" % m, m)"""

    """[Sb, Sw] = computeCovarianceMatrices(D, L)

    s, U = scipy.linalg.eigh(Sb, Sw)
    m=2
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]

    U, s, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot(numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T)

    Sbt = numpy.dot(numpy.dot(P1, Sb), P1.T)

    s, V = scipy.linalg.eigh(Sbt)
    P2 = V[:, ::-1][:, 0:m]

    W = numpy.dot(P1.T, P2)

    DP = numpy.dot(W.T, D)

    scatter(DP, L, "./modifiedGraphsLDA_%s/" % m, m)"""