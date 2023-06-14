# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:09:26 2023

@author: gianm
"""

import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import json

def mcol(v):
    return v.reshape((v.size, 1))

def load(file):
    vectors = []
    labels = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }
    with open(file) as f:
            for line in f:
                w = line.split(",")
                attr = w[:-1]
                vectors.append(mcol(numpy.array([float(i) for i in attr])))
                labels.append(hLabels[w[-1].strip()])
    return (numpy.hstack(vectors), numpy.array(labels, dtype=numpy.int32))
    
def plot_hist(D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]
    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }
    for i in range(4):   
        plt.figure()
        plt.xlabel(hFea[i])
        plt.hist(D0[i, :], density=True, alpha = 0.4, label='Setosa')
        plt.hist(D1[i, :], density=True, alpha = 0.4, label='Versicolor')
        plt.hist(D2[i, :], density=True, alpha = 0.4, label='Virginica')
        plt.legend()
    plt.show()

def plot_scatter(D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]
    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }
    for i in range(3):
        for j in range(i + 1, 4):
            plt.figure()
            plt.xlabel(hFea[i])
            plt.ylabel(hFea[j])
            plt.scatter(D0[i, :], D0[j, :], label='Setosa')
            plt.scatter(D1[i, :], D1[j, :], label='Versicolor')
            plt.scatter(D2[i, :], D2[j, :], label='Virginica')
            plt.legend()
        plt.show()

def load2():
    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def PCA(D, m):
    
    mu = vcol(D.mean(1))
    #print("mean:", mu)
    C = (numpy.dot((D - mu), (D - mu).T))/D.shape[1]
    #print("C:", C)
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P

def PCA2(D, m):    
    
    #Alternative method to calculate P 
    mu = vcol(D.mean(1))
    C = (numpy.dot((D - mu), (D - mu).T))/D.shape[1]
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def SbSw(D, L):
    
    classes = set(L)
    mu = vcol(D.mean(1))
    SB = 0
    SW = 0
    N = D.shape[1]
    for c in classes:
        DCls = D[:, L==c]
        nCls = DCls.shape[1]
        muCls = vcol(DCls.mean(1))
        SW += numpy.dot((DCls-muCls), (DCls-muCls).T)
        SB += nCls*(numpy.dot((muCls-mu), (muCls-mu).T))
    SB = SB/N
    SW = SW/N
    #print("SB:\n", SB )
    #print("SW:\n", SW )
    return SB, SW

def LDA1(D, L, m):
    
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    #UW, _, _ = numpy.linalg.svd(W)
    #U = UW[:, 0:m]
    #print("Ortoganal")
    #print(U)
    return W

def LDA2(D, L, m):
    
    SB, SW = SbSw(D, L)
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot(U * vrow(1.0 / (s ** 0.5)), U.T)
    #P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )
    SBT = numpy.dot(P1, numpy.dot(SB, P1.T))
    U, s, Vh = numpy.linalg.svd(SBT)
    P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)
    return W

def logpdf_GAU_ND(X, mu, C):
    
    M = C.shape[0]
    N = X.shape[1]
    _, det = numpy.linalg.slogdet(C)
    inv = numpy.linalg.inv(C)
    const = - 0.5 * M * numpy.log(2*numpy.pi)
    res = []
    for i in range(N):
        x = vcol(X[:, i])
        xc = x - mu
        res.append(const - 0.5 * det - 0.5 * numpy.dot(xc.T, numpy.dot(inv, xc)).ravel())
    return numpy.array(res).ravel()

def logpdf_GAU_ND_fast(X, mu, C):
    
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * numpy.log(2*numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (XC*numpy.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

def loglikelihood(XND, m_ML, C_ML):
    return sum(logpdf_GAU_ND(XND, m_ML, C_ML))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def multivariete_gaussian_classifier(DTR, LTR, DTE, LTE, prior):
    S = []
    prior = vcol(numpy.ones(2)*prior)
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0 = vcol(D0.mean(1))
    C0 = (numpy.dot((D0 - mu0), (D0 - mu0).T))/D0.shape[1]
    mu1 = vcol(D1.mean(1))
    C1 = (numpy.dot((D1 - mu1), (D1 - mu1).T))/D1.shape[1]
    fc0 = logpdf_GAU_ND_fast(DTE, mu0, C0)
    S.append(vrow(numpy.exp(fc0)))
    fc1 = logpdf_GAU_ND_fast(DTE, mu1, C1)
    S.append(vrow(numpy.exp(fc1)))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    PL = P.argmax(0)
    return S, PL

def naive_bayes_gaussian_classifier(DTR,LTR,DTE,LTE, prior):
    S = []
    prior = vcol(numpy.ones(2)*prior)
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0 = vcol(D0.mean(1))
    C0 = (numpy.dot((D0 - mu0), (D0 - mu0).T))/D0.shape[1]
    mu1 = vcol(D1.mean(1))
    C1 = (numpy.dot((D1 - mu1), (D1 - mu1).T))/D1.shape[1]
    diagonal = numpy.identity(mu0.shape[0])
    C0_d = C0*diagonal
    C1_d = C1*diagonal
    fc0 = logpdf_GAU_ND_fast(DTE, mu0, C0_d)
    S.append(vrow(numpy.exp(fc0)))
    fc1 = logpdf_GAU_ND_fast(DTE, mu1, C1_d)
    S.append(vrow(numpy.exp(fc1)))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    PL = P.argmax(0)
    return S, PL

def tied_covariance_gaussian_classier(DTR,LTR,DTE,LTE, prior):
    S = []
    prior = vcol(numpy.ones(2)*prior)
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0 = vcol(D0.mean(1))
    C0 = (numpy.dot((D0 - mu0), (D0 - mu0).T))/D0.shape[1]
    mu1 = vcol(D1.mean(1))
    C1 = (numpy.dot((D1 - mu1), (D1 - mu1).T))/D1.shape[1]
    C_tied = (C0*D0.shape[1]+C1*D1.shape[1])/DTR.shape[1]
    fc0 = logpdf_GAU_ND_fast(DTE, mu0, C_tied)
    S.append(vrow(numpy.exp(fc0)))
    fc1 = logpdf_GAU_ND_fast(DTE, mu1, C_tied)
    S.append(vrow(numpy.exp(fc1)))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    PL = P.argmax(0)
    return S, PL

def tied_naive_bayes_classier(DTR,LTR,DTE,LTE):
    S = []
    prior = vcol(numpy.ones(3)/3.0)
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    D2 = DTR[:, LTR==2]
    mu0 = vcol(D0.mean(1))
    C0 = (numpy.dot((D0 - mu0), (D0 - mu0).T))/D0.shape[1]
    mu1 = vcol(D1.mean(1))
    C1 = (numpy.dot((D1 - mu1), (D1 - mu1).T))/D1.shape[1]
    mu2 = vcol(D2.mean(1))
    C2 = (numpy.dot((D2 - mu2), (D2 - mu2).T))/D2.shape[1]
    diagonal = numpy.identity(mu0.shape[0])
    C0_d = C0*diagonal
    C1_d = C1*diagonal
    C2_d = C2*diagonal
    C_tied = (C0_d*D0.shape[1]+C1_d*D1.shape[1]+C2_d*D2.shape[1])/DTR.shape[1]
    fc0 = logpdf_GAU_ND_fast(DTE, mu0, C_tied)
    S.append(vrow(numpy.exp(fc0)))
    fc1 = logpdf_GAU_ND_fast(DTE, mu1, C_tied)
    S.append(vrow(numpy.exp(fc1)))
    fc2 = logpdf_GAU_ND_fast(DTE, mu2, C_tied)
    S.append(vrow(numpy.exp(fc2)))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    PL = P.argmax(0)
    return S, PL
    

def k_fold_cross_validation(D, L, K):
    prediction_mgc = numpy.zeros(D.shape[1])
    S_mgc = numpy.zeros((3, D.shape[1]))
    prediction_nb = numpy.zeros(D.shape[1])
    S_nb = numpy.zeros((3, D.shape[1]))
    prediction_tc = numpy.zeros(D.shape[1])
    S_tc = numpy.zeros((3, D.shape[1]))
    prediction_tnb = numpy.zeros(D.shape[1])
    S_tnb = numpy.zeros((3, D.shape[1]))
    for i in range(D.shape[1]):
        DTR = numpy.delete(D,i,1)
        LTR = numpy.delete(L,i)
        DTE = vcol(D[:,i])
        LTE = L[i]
        S, pcarr_mgc= multivariete_gaussian_classifier(DTR,LTR,DTE,LTE)
        S_mgc[0][i] = S[0]
        S_mgc[1][i] = S[1]
        S_mgc[2][i] = S[2]
        prediction_mgc[i]=pcarr_mgc
        S, pcarr_nb= naive_bayes_gaussian_classifier(DTR,LTR,DTE,LTE)
        S_nb[0][i] = S[0]
        S_nb[1][i] = S[1]
        S_nb[2][i] = S[2]
        prediction_nb[i]=pcarr_nb
        S, pcarr_tc= tied_covariance_gaussian_classier(DTR,LTR,DTE,LTE)
        S_tc[0][i] = S[0]
        S_tc[1][i] = S[1]
        S_tc[2][i] = S[2]
        prediction_tc[i]=pcarr_tc
        S, pcarr_tnb= tied_naive_bayes_classier(DTR,LTR,DTE,LTE)
        S_tnb[0][i] = S[0]
        S_tnb[1][i] = S[1]
        S_tnb[2][i] = S[2]
        prediction_tnb[i]=pcarr_tnb
    print("----------------------")
    print("Multivariate Gaussian model")
    logSJoint = numpy.log(S_mgc)
    logSJoint_prof = numpy.load('Solution/LOO_logSJoint_MVG.npy')
    print(numpy.abs((logSJoint-logSJoint_prof)).max())
    cont_true = 0
    cont_false = 0
    for i in range(150):
        if(prediction_mgc[i]==L[i]):
            cont_true +=1
        else:
            cont_false+=1
    print("Accuracy", cont_true/150*100, "%")
    print("Error", cont_false/150*100, "%")
    
    print("----------------------")
    print("Naive Bayes Gaussian model")
    logSJoint = numpy.log(S_nb)
    logSJoint_prof = numpy.load('Solution/LOO_logSJoint_NaiveBayes.npy')
    print(numpy.abs((logSJoint-logSJoint_prof)).max())
    cont_true = 0
    cont_false = 0
    for i in range(150):
        if(prediction_nb[i]==L[i]):
            cont_true +=1
        else:
            cont_false+=1
    print("Accuracy", cont_true/150*100, "%")
    print("Error", cont_false/150*100, "%")
    
    print("----------------------")
    print("Tied Covariance Gaussian model")
    logSJoint = numpy.log(S_tc)
    logSJoint_prof = numpy.load('Solution/LOO_logSJoint_TiedMVG.npy')
    print(numpy.abs((logSJoint-logSJoint_prof)).max())
    cont_true = 0
    cont_false = 0
    for i in range(150):
        if(prediction_tc[i]==L[i]):
            cont_true +=1
        else:
            cont_false+=1
    print("Accuracy", cont_true/150*100, "%")
    print("Error", cont_false/150*100, "%")
    
    print("----------------------")
    print("Tied Naive Bayes model")
    logSJoint = numpy.log(S_tnb)
    logSJoint_prof = numpy.load('Solution/LOO_logSJoint_TiedNaiveBayes.npy')
    print(numpy.abs((logSJoint-logSJoint_prof)).max())
    cont_true = 0
    cont_false = 0
    for i in range(150):
        if(prediction_tnb[i]==L[i]):
            cont_true +=1
        else:
            cont_false+=1
    print("Accuracy", cont_true/150*100, "%")
    print("Error", cont_false/150*100, "%")
    
def f(x):
    y = x[0]
    z = x[1]
    return (y+3)**2+numpy.sin(y)+(z+1)**2

def f_grad(x):
    y = x[0]
    z = x[1]
    f = (y+3)**2+numpy.sin(y)+(z+1)**2
    grad = numpy.zeros((2,))
    grad[0] = 2*(y+3)+numpy.cos(y)
    grad[1] = 2*(z+1)
    return f, grad

def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]
    DTR = DTR.T
    n = DTR.shape[0]
    ris = 0
    for i in range(n):
        z = 2*LTR[i] - 1
        ris += numpy.logaddexp(0, -z*(numpy.dot(w.T, DTR[i])+b))
    return l/2 * numpy.linalg.norm(w)**2 + (ris/n)    
    
def logreg(v, DTE):
    w,b = v[0:-1], v[-1]
    n = DTE.shape[1]
    DTE = DTE.T
    s = numpy.zeros(n)
    for i in range(n):
        s[i]=numpy.dot(w.T, DTE[i])+b 
    LP = numpy.zeros(n)
    for i in range(n):
        if s[i] > 0:
            LP[i] = 1
        else:
            LP[i] = 0
    return s, LP

def accuracy(v, LTE):
    n = v.shape[0]
    cont_false = 0
    for i in range(n):
        if(v[i] != LTE[i]):
            cont_false += 1
    print("Error rate %.1f" %(cont_false/n*100), "%")
    
def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        n = DTR.shape[1]
        ris = 0
        for i in range(n):
            z = 2*LTR[i] - 1
            ris += numpy.logaddexp(0, -z*(numpy.dot(w.T, DTR.T[i])+b))
        return l/2 * numpy.linalg.norm(w)**2 + (ris/n)
    return logreg_obj

def logreg_obj_wrap_prof(DTR, LTR, l):
    dim = DTR.shape[0]
    ZTR = LTR * 2.0 - 1.0
    def logreg_obj(v):
        w = mcol(v[0:dim])
        b = v[-1]
        scores = numpy.dot(w.T, DTR) + b
        loss_per_sample = numpy.logaddexp(0, -ZTR * scores)
        loss = loss_per_sample.mean() + 0.5*l*numpy.linalg.norm(w)**2
        return loss
    return logreg_obj

class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
    def logreg_obj(self, v):
        # Compute and return the objective function value. You can
        #retrieve all required information from self.DTR, self.LTR,
        #self.l
        w, b = v[0:-1], v[-1]
        DTR = self.DTR.T
        n = DTR.shape[0]
        ris = 0
        for i in range(n):
            z = 2*self.LTR[i] - 1
            ris += numpy.logaddexp(0, -z*(numpy.dot(w.T, DTR[i])+b))
        return self.l/2 * numpy.linalg.norm(w)**2 + (ris/n)

def multiclass_logreg_obj(v, DTR, LTR, l):
    W, b = v[: -3].reshape(DTR.shape[0], 3), mcol(v[-3:])
    n = DTR.shape[1]
    S = numpy.dot(W.T, DTR) + b
    x = numpy.zeros((DTR.shape[1],1))
    for i in range(DTR.shape[1]):
        x[i] = numpy.log(numpy.sum(numpy.exp(S.T[i])))
    Ylog = S - x.T
    T = numpy.zeros((n, 3))
    for i in range(n):
        z = numpy.zeros((3,))
        z[LTR[i]] = 1
        T[i] = z
    T = T.T
    loss = (T*Ylog).sum()
    ris = l/2 * (W*W).sum() - loss/n
    return ris
    
def multiclass_logreg(v, DTE):
    W, b = v[: -3].reshape(DTE.shape[0], 3), v[-3:]
    n = DTE.shape[1]
    DTE = DTE.T
    LP = numpy.zeros(n)
    sc = numpy.zeros(3)
    for i in range(n):
        for k in range(3):
            sc[k]=numpy.dot(W.T[k], DTE[i])+b[k]
        LP[i] = numpy.argmax(sc)
    return LP

def optimal_Bayes_decision(pi1, Cfn, Cfp, LLRs, labels):
    t = -numpy.log((pi1*Cfn)/((1-pi1)*Cfp))
    conf_matrix = numpy.zeros((2,2), numpy.int32)
    for i in range(labels.shape[0]):
        p = 0
        if LLRs[i] > t:
            p = 1
        conf_matrix[p][labels[i]] += 1
    print(conf_matrix)
    
def Bayes_risk(pi1, Cfn, Cfp, LLRs, labels):
    t = -numpy.log((pi1*Cfn)/((1-pi1)*Cfp))
    conf_matrix = numpy.zeros((2,2), numpy.int32)
    for i in range(labels.shape[0]):
        p = 0
        if LLRs[i] > t:
            p = 1
        conf_matrix[p][labels[i]] += 1
    FNR = conf_matrix[0][1]/(conf_matrix[0][1] + conf_matrix[1][1])
    FPR = conf_matrix[1][0]/(conf_matrix[0][0] + conf_matrix[1][0])
    DCF = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    print(pi1, Cfn, Cfp, "\t", DCF)
    
def Bayes_risk_Bdummy(pi1, Cfn, Cfp, LLRs, labels):
    t = -numpy.log((pi1*Cfn)/((1-pi1)*Cfp))
    conf_matrix = numpy.zeros((2,2), numpy.int32)
    for i in range(labels.shape[0]):
        p = 0
        if LLRs[i] > t:
            p = 1
        conf_matrix[p][labels[i]] += 1
    FNR = conf_matrix[0][1]/(conf_matrix[0][1] + conf_matrix[1][1])
    FPR = conf_matrix[1][0]/(conf_matrix[0][0] + conf_matrix[1][0])
    DCF = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    Bdummy = min(pi1*Cfn, (1-pi1)*Cfp)
    return DCF/Bdummy

def Bayes_risk_min_cost(pi1, Cfn, Cfp, LLRs, labels):
    DCF = []
    for i in range(labels.shape[0]):
        conf_matrix = numpy.zeros((2,2), numpy.int32)
        t = LLRs[i]
        for j in range(labels.shape[0]):
            p = 0
            if LLRs[j] > t:
                p = 1
            conf_matrix[p][labels[j]] += 1
        FNR = conf_matrix[0][1]/(conf_matrix[0][1] + conf_matrix[1][1])
        FPR = conf_matrix[1][0]/(conf_matrix[0][0] + conf_matrix[1][0])
        Bdummy = min(pi1*Cfn, (1-pi1)*Cfp)
        DCF.append((pi1*Cfn*FNR+(1-pi1)*Cfp*FPR)/Bdummy)
    return min(DCF)

def ROC(LLRs, labels):
    LLRs_sorted = numpy.sort(LLRs)
    TPR = []
    FPR = []
    for i in range(labels.shape[0]):
        conf_matrix = numpy.zeros((2,2), numpy.int32)
        t = LLRs_sorted[i]
        cont_p = 0
        for j in range(labels.shape[0]):
            p = 0
            if LLRs[j] > t:
                cont_p += 1
                p = 1
            conf_matrix[p, labels[j]] += 1
        FNR = conf_matrix[0, 1]/(conf_matrix[0, 1] + conf_matrix[1, 1])
        FPR.append((conf_matrix[1, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0])))
        TPR.append(1-FNR)
    plt.figure()
    plt.plot(numpy.array(FPR), numpy.array(TPR))
    plt.show()
    
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

def bounds_creator(C, n):
    bounds = []
    for i in range(n):
        bounds.append([0,C])
    return bounds
    

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

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]

def logpdf_GMM(X, gmm):
    M = len(gmm)
    S = numpy.zeros((M, X.shape[1]))
    for m in range(M):
        S[m] = logpdf_GAU_ND_fast(X, gmm[m][1], gmm[m][2])
    for g in range(M):
        S[g, :] += numpy.log(gmm[g][0])
    logdens = scipy.special.logsumexp(S, axis=0)
    return S, logdens

def E_step(S, logdens):
    for i in range(S.shape[0]):
        S[i] = numpy.exp(S[i] - logdens)
    return S

def M_step(E, X):
    M, N = E.shape
    Zg = numpy.sum(E, axis=1)
    Fg = []
    for i in range(M):
        sm = 0
        for j in range(N):
            sm += E[i][j] * mcol(X[:, j])
        Fg.append(sm)
    Sg = []
    for i in range(M):
        sm = 0
        for j in range(N):
            x = mcol(X[:, j])
            sm += E[i][j] * numpy.dot(x, x.T)
        Sg.append(sm)
    new_GMM = []
    for i in range(M):
        mu = mcol(Fg[i] / Zg[i])  
        cov = Sg[i] / Zg[i] - numpy.dot(mu, mu.T)
        new_GMM.append([Zg[i] / sum(Zg), mu, cov])
    return new_GMM

def GMM_EM_estimation(X, gmm, delta):
    referenceGMM = gmm
    S, logdens = logpdf_GMM(X, gmm)
    E = E_step(S, logdens)
    new_GMM = M_step(E, X)
    S_new, logdens_new = logpdf_GMM(X, new_GMM)
    while((logdens_new.mean() - logdens.mean()) > delta):
        print("old logs avg", logdens.mean(), "new logs avg", logdens_new.mean())
        referenceGMM = new_GMM
        S, logdens = S_new, logdens_new
        E = E_step(S, logdens)
        new_GMM = M_step(E, X)
        S_new, logdens_new = logpdf_GMM(X, new_GMM)  
    return referenceGMM

def LBG_algorithm(gmm, alpha):
    new_gmm = []
    for e in gmm:
        Sigma_g = e[2]    
        U, s, Vh = numpy.linalg.svd(Sigma_g)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        line1 = []
        line2 = []
        mu = e[1]
        line1.append(e[0]/2)
        line1.append(mu-d)
        line1.append(Sigma_g)
        new_gmm.append(line1)
        line2.append(e[0]/2)
        line2.append(mu+d)
        line2.append(Sigma_g)
        new_gmm.append(line2)
    return new_gmm

def M_step_diagonal(E, X):
    psi = 0.01
    M, N = E.shape
    Zg = numpy.sum(E, axis=1)
    Fg = []
    for i in range(M):
        sm = 0
        for j in range(N):
            sm += E[i][j] * mcol(X[:, j])
        Fg.append(sm)
    Sg = []
    for i in range(M):
        sm = 0
        for j in range(N):
            x = mcol(X[:, j])
            sm += E[i][j] * numpy.dot(x, x.T)
        Sg.append(sm)
    new_GMM = []
    for i in range(M):
        mu = mcol(Fg[i] / Zg[i])  
        cov = Sg[i] / Zg[i] - numpy.dot(mu, mu.T)
        covNew = cov * numpy.eye(cov.shape[0])
        U, s, _ = numpy.linalg.svd(covNew)
        s[s<psi] = psi
        covNew = numpy.dot(U, mcol(s)*U.T)
        new_GMM.append([Zg[i] / sum(Zg), mu, covNew])
    return new_GMM
        
def GMM_EM_estimation_diagonal(X, gmm, delta):
    S, logdens = logpdf_GMM(X, gmm)
    while(1):
        E = E_step(S, logdens)
        new_GMM = M_step_diagonal(E, X)
        S_new, logdens_new=logpdf_GMM(X, new_GMM)
        if((logdens_new.mean() - logdens.mean()) < delta):
            referenceGMM = new_GMM
            print("Final average log-likelihood:", logdens_new.mean())
            break      
        referenceGMM = new_GMM
        S, logdens=S_new, logdens_new
    return referenceGMM

def M_step_tied(E, X):
    psi = 0.01
    M, N = E.shape
    Zg = numpy.sum(E, axis=1)
    Fg = []
    for i in range(M):
        sm = 0
        for j in range(N):
            sm += E[i][j] * mcol(X[:, j])
        Fg.append(sm)
    Sg = []
    for i in range(M):
        sm = 0
        for j in range(N):
            x = mcol(X[:, j])
            sm += E[i][j] * numpy.dot(x, x.T)
        Sg.append(sm)
    new_GMM = []
    covNew = 0
    for i in range(M):
        mu = mcol(Fg[i] / Zg[i])
        covNew += Zg[i] * (Sg[i] / Zg[i] - numpy.dot(mu, mu.T))
    covNew /= N
    U, s, _ = numpy.linalg.svd(covNew)
    s[s<psi] = psi
    covNew = numpy.dot(U, mcol(s)*U.T)
    for i in range(M):
        mu = mcol(Fg[i] / Zg[i])  
        new_GMM.append([Zg[i] / sum(Zg), mu, covNew])
    return new_GMM
        
def GMM_EM_estimation_tied(X, gmm, delta):
    S, logdens=logpdf_GMM(X, gmm)
    while(1):
        E = E_step(S, logdens)
        new_GMM = M_step_tied(E, X)
        S_new, logdens_new=logpdf_GMM(X, new_GMM)
        if((logdens_new.mean() - logdens.mean()) < delta):
            referenceGMM = new_GMM
            print("Final average log-likelihood:", logdens_new.mean())
            break      
        referenceGMM = new_GMM
        S, logdens=S_new, logdens_new
    return referenceGMM