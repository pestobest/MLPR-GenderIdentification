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

def Z_norm(D):
    Dstd = vcol(numpy.std(D, axis=1))
    Dmean = vcol(D.mean(1))
    D = (D - Dmean) / Dstd
    return D
        
def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def PCA(D, m):
    mu = vcol(D.mean(1))
    DC = D - mu
    C = 0
    dotDC = numpy.dot(DC, DC.T)
    C = (1 / float(D.shape[1])) * dotDC
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
    return fc1-fc0, PL

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
    return fc1-fc0, PL

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
    return fc1-fc0, PL

def tied_naive_bayes_classier(DTR,LTR,DTE,LTE, prior):
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
    C_tied = (C0_d*D0.shape[1]+C1_d*D1.shape[1])/DTR.shape[1]
    fc0 = logpdf_GAU_ND_fast(DTE, mu0, C_tied)
    S.append(vrow(numpy.exp(fc0)))
    fc1 = logpdf_GAU_ND_fast(DTE, mu1, C_tied)
    S.append(vrow(numpy.exp(fc1)))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    PL = P.argmax(0)
    return fc1-fc0, PL

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
    #print("Error rate %.1f" %(cont_false/n*100), "%")
    return cont_false/n*100

def Ksplit(D, L, seed=0, K=3):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    # Generate a random seed
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
        labels.append(L[idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
    return folds, labels
   
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
    alpha, dual_loss, _d = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(DTR.shape[1]), args=(H,), bounds=bounds, factr=1.0)
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
        
# def LD_alpha(alpha, H, ones):
#     f = 0.5*numpy.dot(alpha.T, numpy.dot( H, alpha)) - numpy.dot(alpha.T, ones)
#     grad = numpy.dot(H, alpha)- ones
#     grad = grad.reshape(alpha.shape[0],)
#     return f, grad

def LD_alpha(alpha, H):
    f = 0.5*numpy.dot(alpha.T, numpy.dot( H, alpha)) - numpy.dot(alpha.T, numpy.ones(alpha.shape))
    grad = numpy.dot(H, alpha)-  numpy.ones(alpha.shape)
    grad = grad.reshape(alpha.shape[0],)
    return f, grad

def accuracy_v2(v, LTE):
    n = v.shape[0]
    cont_true = 0
    for i in range(n):
        if((v[i] > 0 and LTE[i] == 1) or (v[i] <= 0 and LTE[i] == 0)):
            cont_true += 1
    #print("Error rate %.1f" % ((1-(cont_true/n))*100), "%", sep="")
    return (1-(cont_true/n))*100

def bounds_creator(C, n):
    bounds = []
    for i in range(n):
        bounds.append([0,C])
    return bounds
    

def polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE):
    LTR = 2 * LTR - 1
    n = DTR.shape[1]
    H = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            kernel=(numpy.dot(DTR[:, i].T, DTR[:, j]) + c)**d + K**2
            H[i, j] = LTR[i] * LTR[j] * kernel
    bounds = numpy.full((n, 2), (0, C))
    alpha, dual_loss, p = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(n), args=(H,), bounds=bounds, factr=1.0)
    #print("Dual loss", -dual_loss)
    kernel=numpy.zeros((n,DTE.shape[1]))
    for i in range(n):
        for j in range(DTE.shape[1]):
            kernel = (numpy.dot(DTR[:, i].T, DTE[:, j]) + c)**d + K**2
    return numpy.dot(alpha * LTR, kernel)

def KFunc_RBF(D1, D2, g, K):
    dist = vcol((D1**2).sum(0)) + vrow((D2**2).sum(0)) - 2 * numpy.dot(D1.T, D2)
    return numpy.exp(-g*dist) + K**2

def RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE):
    LTR = 2 * LTR - 1
    n = DTR.shape[1]
    H = numpy.zeros((n,n))
    kernel = KFunc_RBF(DTR, DTR, gamma, K)
    H = kernel * vcol(LTR) * vrow(LTR)
    bounds = numpy.full((n, 2), (0, C))
    alpha, dual_loss, p = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(n), args=(H,), bounds=bounds, factr=1.0)
    #print("Dual loss", -dual_loss)
    kernel = KFunc_RBF(DTR, DTE, gamma, K)
    return numpy.dot(alpha * LTR, kernel)

# def RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE):
#     LTR = 2 * LTR - 1
#     n = DTR.shape[1]
#     H = numpy.zeros((n,n))
#     kernel = numpy.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             kernel[i, j] = numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i] - DTR[:, j])**2) + K**2
#     H = kernel * vcol(LTR) * vrow(LTR)
#     bounds = numpy.full((n, 2), (0, C))
#     alpha, dual_loss, p = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(n), args=(H,), bounds=bounds, factr=1.0)
#     print("Dual loss", -dual_loss)
#     kernel = numpy.zeros((n, DTE.shape[1]))
#     for i in range(n):
#         for j in range(DTE.shape[1]):
#             kernel[i, j] = numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i]-DTE[:, j])**2) + K**2
#     return numpy.dot(alpha * LTR, kernel)

# def RBF_kernel_SVM(DTR, LTR, C, K, gamma, LTE, DTE):
#     LTR = 2 * LTR - 1
#     n = DTR.shape[1]
#     H = numpy.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             kernel=numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i] - DTR[:, j])**2) + K**2
#             H[i, j] = LTR[i] * LTR[j] * kernel
#     bounds = numpy.full((n, 2), (0, C))
#     ones = numpy.ones(n)
#     alpha, dual_loss, p = scipy.optimize.fmin_l_bfgs_b(LD_alpha, x0=numpy.zeros(n), args=(H, ones), bounds=bounds, factr=1.0)
#     print("Dual loss", -dual_loss)
#     s=numpy.zeros(len(LTE))
#     for i in range(len(LTE)):
#         for j in range(n):
#             if(alpha[j] != 0):
#                 k = numpy.exp(-gamma * numpy.linalg.norm(DTR[:, j]-DTE[:, i])**2) + K**2
#                 s[i] += alpha[j] * LTR[j] * k
#     for i in range(len(LTE)):
#         if s[i] > 0:
#             s[i] = 1
#         else:
#             s[i] = 0
#     return s

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

def correlationsPlot(D, L, desc = ''):
    cmap = ['Greys', 'Reds', 'Blues']
    corrCoeff = {
        0: numpy.abs(numpy.corrcoef(D)),
        1: numpy.abs(numpy.corrcoef(D[:, L == 0])),
        2: numpy.abs(numpy.corrcoef(D[:, L == 1]))
    }
    for i in range(3):
        plt.figure()
        plt.imshow(corrCoeff[i], cmap = cmap[i], interpolation = 'nearest')
        plt.savefig('heatmaps/' + desc + 'heatmap_%d.jpg' % i)