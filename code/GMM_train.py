# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:03:27 2023

@author: gianm
"""
import numpy
import scipy.optimize
import json
from Library_gianmarco import multivariete_gaussian_classifier, Bayes_risk_min_cost, PCA, Ksplit, Z_norm

from library import load

def v_col(x: numpy.ndarray) -> numpy.ndarray:
    return x.reshape((x.size, 1))


def v_row(x: numpy.ndarray) -> numpy.ndarray:
    return x.reshape((1, x.size))


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

def mcol(v):
    return v.reshape((v.size, 1))

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]

def logpdf_GAU_ND_fast(X, mu, C): 
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (XC * numpy.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

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
        new_GMM.append([Zg[i] / sum(Zg), mu, cov])
    U, s, _ = numpy.linalg.svd(new_GMM[0][2])
    s[s<psi] = psi
    new_GMM[0][2] = numpy.dot(U, mcol(s)*U.T)
    return new_GMM

def GMM_EM_estimation(X, gmm, delta, psi = 0.01):
    ll_new = None
    ll_old = None
    while ll_old is None or ll_new - ll_old > 1e-6:
        ll_old = ll_new
        s_joint = numpy.zeros((len(gmm), X.shape[1]))
        for g in range(len(gmm)):
            s_joint[g, :] = logpdf_GAU_ND_fast(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        s_marginal = scipy.special.logsumexp(s_joint, axis=0)
        ll_new = s_marginal.sum() / X.shape[1]
        P = numpy.exp(s_joint - s_marginal)
        gmm_new = []
        z_vec = numpy.zeros(len(gmm))
        for g in range(len(gmm)):
            gamma = P[g, :]
            zero_order = gamma.sum()
            z_vec[g] = zero_order
            first_order = (v_row(gamma) * X).sum(1)
            second_order = numpy.dot(X, (v_row(gamma) * X).T)
            w = zero_order / X.shape[1]
            mu = v_col(first_order / zero_order)
            sigma = second_order / zero_order - numpy.dot(mu, mu.T)
            gmm_new.append((w, mu, sigma))

        for i in range(len(gmm)):
            transformed_sigma = gmm_new[i][2]
            u, s, _ = numpy.linalg.svd(transformed_sigma)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], numpy.dot(u, v_col(s) * u.T))
        gmm = gmm_new
    return gmm


# def GMM_EM_estimation(X, gmm, delta):
#     referenceGMM = gmm
#     S, logdens = logpdf_GMM(X, gmm)
#     E = E_step(S, logdens)
#     new_GMM = M_step(E, X)
#     S_new, logdens_new = logpdf_GMM(X, new_GMM)
#     while((logdens_new.mean() - logdens.mean()) > delta):
#         print("old logs avg", logdens.mean(), "new logs avg", logdens_new.mean())
#         referenceGMM = new_GMM
#         S, logdens = S_new, logdens_new
#         E = E_step(S, logdens)
#         new_GMM = M_step(E, X)
#         S_new, logdens_new = logpdf_GMM(X, new_GMM)  
#     return referenceGMM

def LBG_algorithm(iterations, D, gmm, alpha=0.1, psi = 0.01):
    U, s, _ = numpy.linalg.svd(gmm[0][2])
    s[s<psi] = psi
    gmm[0][2] = numpy.dot(U, mcol(s)*U.T)
    start_gmm = GMM_EM_estimation(D, gmm, 10**-6)
    
    for i in range(iterations):
        new_gmm = []
        for e in start_gmm:
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
        start_gmm = GMM_EM_estimation(D, new_gmm, 10**-6)
    return start_gmm

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
            #print("Final average log-likelihood:", logdens_new.mean())
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

def NB(DTR, LTR):
    mu = []
    C = []

    for i in range(2):
        X = DTR[:, LTR == i]
        mu.append(X.mean(1))
        X = X - mcol(mu[i])
        C.append((1 / X.shape[1]) * numpy.dot(X, X.T) * numpy.identity(4))
    
    return mu, C
    

[D, L] = load('../Train.txt')
# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

K = 5
folds, labels = Ksplit(D, L, seed=0, K=K)
delta = 10**-6
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
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu_D0=D0.mean(1).reshape(12,1)
        
    mu_D1=D1.mean(1).reshape(12,1)
        

    n0=(LTR==0).sum()
    n1=(LTR==1).sum()

    cov_D0=((D0-mu_D0).dot((D0-mu_D0).T))/n0

    cov_D1=((D1-mu_D1).dot((D1-mu_D1).T))/n1

    l,r=numpy.shape(DTR)

    GMM_D0 = [[1, mu_D0, cov_D0]]

    GMM_D1 = [[1, mu_D1, cov_D1]]
    
    EM_GMM_D0 =LBG_algorithm(5, D0, GMM_D0)
    
    EM_GMM_D1 =LBG_algorithm(5, D1, GMM_D1)

    _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)

    _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
    s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
    scores.append(s)
    LTEs.append(LTE)
scores = numpy.hstack(scores)
orderedLabels = numpy.hstack(LTEs)
min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
print("min cost: %.3f" %min_cost)

    # GMM_8_D0 = LBG_algorithm(GMM_4_D0, 0.1)
    # #EM_GMM_D0 = GMM_EM_estimation(D0, GMM_8_D0, delta)

    # _, logdens_D0=logpdf_GMM(DTE, GMM_8_D0)

    # GMM_8_D1 = LBG_algorithm(GMM_4_D1, 0.1)
    # #EM_GMM_D1 = GMM_EM_estimation(D1, GMM_8_D1, delta)

    # _, logdens_D1=logpdf_GMM(DTE, GMM_8_D1)

    # #EM_GMM_D2 = GMM_EM_estimation(D2, GMM_8_D2, delta)

    # S=numpy.matrix([logdens_D0,logdens_D1])    
    # spost = S.sum(axis=0)
    # pc=S.argmax(axis=0)
    # pcarr=numpy.ravel(pc)

    # M,l = numpy.shape(DTE)
    # cont_true = 0
    # cont_false = 0
    # for i in range(l):
    #     if(pcarr[i]==LTE[i]):
    #         cont_true +=1
    #     else:
    #         cont_false+=1
    # print("Accuracy D8", cont_true/l*100, "%")
    # print("Error D8", cont_false/l*100, "%")


    # GMM_16_D0 = LBG_algorithm(GMM_8_D0, 0.1)
    # #EM_GMM_D0 = GMM_EM_estimation(D0, GMM_8_D0, delta)

    # _, logdens_D0=logpdf_GMM(DTE, GMM_16_D0)

    # GMM_16_D1 = LBG_algorithm(GMM_8_D1, 0.1)
    # #EM_GMM_D1 = GMM_EM_estimation(D1, GMM_8_D1, delta)

    # _, logdens_D1=logpdf_GMM(DTE, GMM_16_D1)
    # #EM_GMM_D2 = GMM_EM_estimation(D2, GMM_8_D2, delta)

    # S=numpy.matrix([logdens_D0,logdens_D1])    
    # spost = S.sum(axis=0)
    # pc=S.argmax(axis=0)
    # pcarr=numpy.ravel(pc)

    # M,l = numpy.shape(DTE)
    # cont_true = 0
    # cont_false = 0
    # for i in range(l):
    #     if(pcarr[i]==LTE[i]):
    #         cont_true +=1
    #     else:
    #         cont_false+=1
    # print("Accuracy D16", cont_true/l*100, "%")
    # print("Error D16", cont_false/l*100, "%")