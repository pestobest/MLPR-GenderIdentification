import numpy
import matplotlib.pyplot as plot
import scipy
import math

def vcol(vect):
    return vect.reshape((vect.size, 1))

def vrow(vect):
    return vect.reshape((1, vect.size))

def load(filename):
    list_of_samples = []
    list_of_labels = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] != '\n':
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                data[-1] = int(data[-1].rstrip('\n'))
                # Now create a 1-dim array and reshape it as a column vector,
                # then append it to the appropriate list
                list_of_samples.append(vcol(numpy.array(data[0:-1])))
                # Append the value of the class to the appropriate list
                list_of_labels.append(data[-1])
    # We have column vectors, we need to create a matrix, so we have to
    # stack horizontally all the column vectors
    dataset_matrix = numpy.hstack(list_of_samples[:])
    # Create a 1-dim array with class labels
    class_label_array = numpy.array(list_of_labels)
    return dataset_matrix, class_label_array

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

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*3.0/5.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

def kFold(D, k):
    foldDim = int(D.shape[1]/k)
    idx = numpy.random.permutation(D.shape[1])
    M1 = numpy.full(foldDim, True)
    M2 = numpy.full(foldDim, False)
    train = []
    test = []
    for i in range(k):
        T1 = numpy.full((k, foldDim), M1)
        T1 = numpy.delete(T1, i, 0)
        T1 = numpy.insert(T1, i, M2, 0)
        T2 = numpy.full((k, foldDim), M2)
        T2 = numpy.delete(T2, i, 0)
        T2 = numpy.insert(T2, i, M1, 0)
        train.append(numpy.array(idx[numpy.hstack(T1).reshape(D.shape[1])]))
        test.append(idx[numpy.hstack(T2).reshape(D.shape[1])])
    
    return numpy.vstack(train), numpy.vstack(test)

def MVG(DTR, LTR):
    mu = []
    C = []

    for i in range(2):
        X = DTR[:, LTR == i]
        mu.append(X.mean(1))
        X = X - vcol(mu[i])
        C.append(float(1 / X.shape[1]) * numpy.dot(X, X.T))
    
    return mu, C

def NB(DTR, LTR):
    mu = []
    C = []

    for i in range(3):
        X = DTR[:, LTR == i]
        mu.append(X.mean(1))
        X = X - vcol(mu[i])
        C.append((1 / X.shape[1]) * numpy.dot(X, X.T) * numpy.identity(4))
    
    return mu, C

def tied_C(DTR, LTR, C):
    tC = numpy.zeros((len(C[0]), len(C[0])))

    for i in range(len(C)):
        tC += float(1 / DTR.shape[1]) * DTR[:, LTR == i].shape[1] * C[i]

    return tC

def gaussian_model_score(DTE, LTE, mu, C):
    correct = 0
    listS = []

    if len(C) == len(mu):
        for i in range(len(mu)):
            lDataList = []
            for j in range(DTE.shape[1]):
                lDataList.append(loglikelihood(vcol(DTE[:, j]), mu[i], C[i]))
            listS.append(numpy.hstack(lDataList))
        S = numpy.vstack(listS)
    else:
        for i in range(len(mu)):
            lDataList = []
            for j in range(DTE.shape[1]):
                lDataList.append(loglikelihood(vcol(DTE[:, j]), mu[i], C))
            listS.append(numpy.hstack(lDataList))
        S = numpy.vstack(listS)

    P_c = 1 / 3
    logSJoint = S + numpy.log(P_c)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)

    res = SPost.argmax(axis=0)
    correct = 0

    for i in range(res.shape[0]):
        if res[i] == LTE[i]:
            correct += 1

    return correct

def logpdf_GAU_ND_1D(x, mu, C):
    _, logDetC = numpy.linalg.slogdet(C)
    A = float(x.size) * math.log(2 * math.pi)
    B = numpy.dot((x - mu).T, numpy.dot(numpy.linalg.inv(C), (x - mu)))
    return (0.5 * (- A - logDetC - B))

def loglikelihood(X, mu, C):
    sumll = 0
    for i in range(X.shape[1]):
        sumll += logpdf_GAU_ND_1D(X[:, i], mu, C)
    return sumll

def dual_SVM(DTR, LTR, K, C):
    LTR = 2 * LTR - 1
    D = numpy.vstack((DTR, numpy.zeros(DTR.shape[1]) + K))
    G = numpy.dot(D.T, D)
    n = DTR.shape[1]
    H = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i, j] = LTR[i] * LTR[j] * G[i, j]
    bounds = numpy.full((n, 2), (0, C))
    ones = numpy.ones(DTR.shape[1])
    alpha, dual_loss, _d = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(DTR.shape[1]), args=(H, ones), bounds=bounds, factr=1.0)
    print("Dual loss", -dual_loss)
    w = numpy.sum(alpha * D * LTR, axis=1)
    t = 0
    D = D.T
    for i in range(n):
        t += max(0, 1 - LTR[i] * (numpy.dot(w.T, D[i])))
    primal_loss = 0.5 * (numpy.linalg.norm(w))**2 + C * t
    print("Primal loss:", primal_loss)
    print("Dual gap", abs(dual_loss + primal_loss))
    return w
        
def LD_alpha(alpha, H, ones):
    f = 0.5*numpy.dot(alpha.T, numpy.dot( H, alpha)) - numpy.dot(alpha.T, ones)
    grad = numpy.dot(H, alpha)- ones
    grad = grad.reshape(alpha.shape[0],)
    return f, grad

def accuracy_v2(v, LTE):
    n = v.shape[0]
    cont_true = 0
    for i in range(n):
        if((v[i] > 0 and LTE[i] == 1) or (v[i] <= 0 and LTE[i] == 0)):
            cont_true += 1
    print("Error rate %.1f" % ((1-(cont_true/n))*100), "%", sep="")

def polynomial_kernel_SVM(DTR, LTR, C, K, d, c, LTE, DTE):
    LTR = 2 * LTR - 1
    n = DTR.shape[1]
    H = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            kernel=(numpy.dot(DTR[:, i].T, DTR[:, j]) + c)**d + K**2
            H[i, j] = LTR[i] * LTR[j] * kernel
    bounds = numpy.full((n, 2), (0, C))
    ones = numpy.ones(n)
    alpha, dual_loss, p = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(n), args=(H, ones), bounds=bounds, factr=1.0)
    print("Dual loss", -dual_loss)
    s=numpy.zeros(len(LTE))
    for i in range(len(LTE)):
        for j in range(n):
            if(alpha[j] != 0):
                k = (numpy.dot(DTR[:, j].T, DTE[:, i]) + c)**d + K**2
                s[i] += alpha[j] * LTR[j] * k
    for i in range(len(LTE)):
        if s[i] > 0:
            s[i] = 1
        else:
            s[i] = 0
    return s

def RBF_kernel_SVM(DTR, LTR, C, K, gamma, LTE, DTE):
    LTR = 2 * LTR - 1
    n = DTR.shape[1]
    H = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            kernel=numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i] - DTR[:, j])**2) + K**2
            H[i, j] = LTR[i] * LTR[j] * kernel
    bounds = numpy.full((n, 2), (0, C))
    ones = numpy.ones(n)
    alpha, dual_loss, p = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(n), args=(H, ones), bounds=bounds, factr=1.0)
    print("Dual loss", -dual_loss)
    s=numpy.zeros(len(LTE))
    for i in range(len(LTE)):
        for j in range(n):
            if(alpha[j] != 0):
                k = numpy.exp(-gamma * numpy.linalg.norm(DTR[:, j]-DTE[:, i])**2) + K**2
                s[i] += alpha[j] * LTR[j] * k
    for i in range(len(LTE)):
        if s[i] > 0:
            s[i] = 1
        else:
            s[i] = 0
    return s