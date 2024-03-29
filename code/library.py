import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import json

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
    w, b = v[0:-1], v[-1]
    n = DTE.shape[1]
    s = numpy.dot(w, DTE)+b 
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
   
def logreg_obj_wrapper(D, L, l, pi):
    Z = (L * 2) - 1
    M = D.shape[0]

    def logreg_obj(v):
        w, b = mcol(v[0:M]), v[-1]
        c1 = 0.5 * l * (numpy.linalg.norm(w) ** 2)
        c2 = ((pi) * (L[L == 1].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == 1] * (numpy.dot(w.T, D[:, L == 1]) + b)).sum()
        c3 = ((1 - pi) * (L[L == 0].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == -1] * (numpy.dot(w.T, D[:, L == 0]) + b)).sum()
        return c1 + c2 + c3
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
    
def compute_weights(C, LTR, prior):
    bounds = numpy.zeros((LTR.shape[0]))
    empirical_pi_t = (LTR == 1).sum() / LTR.shape[0]
    bounds[LTR == 1] = C * prior[1] / empirical_pi_t
    bounds[LTR == 0] = C * prior[0] / (1 - empirical_pi_t)
    return list(zip(numpy.zeros(LTR.shape[0]), bounds))
    
def dual_SVM(DTR, LTR, K, C, priors=[0.5, 0.5]):
    Z = 2 * LTR - 1
    D = numpy.vstack((DTR, numpy.zeros(DTR.shape[1]) + K))
    G = numpy.dot(D.T, D)
    H = vcol(Z) * vrow(Z) * G
    alpha, dual_loss, _d = scipy.optimize.fmin_l_bfgs_b(
        LD_alpha, 
        x0=numpy.zeros(DTR.shape[1]), 
        args=(H,), 
        bounds=compute_weights(C, LTR, priors), 
        factr=1.0)
    #print("Dual loss", -dual_loss)
    w = numpy.sum(alpha * D * Z, axis=1)
    # t = 0
    # D = D.T
    # for i in range(n):
    #     t += max(0, 1 - LTR[i] * (numpy.dot(w.T, D[i])))
    # primal_loss = 0.5 * (numpy.linalg.norm(w))**2 + C * t
    # print("Primal loss:", primal_loss)
    # print("Dual gap", abs(dual_loss + primal_loss))
    return w

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

def polynomial_kernel_SVM(DTR, LTR, C, K, d, c, DTE, priors=[0.5, 0.5]):
    Z = 2 * LTR - 1
    kernel = (numpy.dot(DTR.T, DTR) + c)**d + K**2
    H = kernel * vcol(Z) * vrow(Z)
    alpha, dual_loss, _d = scipy.optimize.fmin_l_bfgs_b(
        LD_alpha, 
        x0=numpy.zeros(DTR.shape[1]), 
        args=(H,), 
        bounds=compute_weights(C, LTR, priors), 
        factr=1.0)
    kernel = (numpy.dot(DTR.T, DTE) + c)**d + K**2
    return numpy.dot(alpha * Z, kernel)

def KFunc_RBF(D1, D2, g, K):
    dist = vcol((D1**2).sum(0)) + vrow((D2**2).sum(0)) - 2 * numpy.dot(D1.T, D2)
    return numpy.exp(-g*dist) + K**2

def RBF_kernel_SVM(DTR, LTR, C, K, gamma, DTE, priors=[0.5, 0.5]):
    Z = 2 * LTR - 1
    kernel = KFunc_RBF(DTR, DTR, gamma, K)
    H = kernel * vcol(Z) * vrow(Z)
    alpha, dual_loss, _d = scipy.optimize.fmin_l_bfgs_b(
        LD_alpha, 
        x0=numpy.zeros(DTR.shape[1]), 
        args=(H,), 
        bounds=compute_weights(C, LTR, priors), 
        factr=1.0)
    kernel = KFunc_RBF(DTR, DTE, gamma, K)
    return numpy.dot(alpha * Z, kernel)

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
        S[m, :] = logpdf_GAU_ND_fast(X, gmm[m][1], gmm[m][2])
    for g in range(M):
        S[g, :] += numpy.log(gmm[g][0])
    logdens = scipy.special.logsumexp(S, axis=0)
    return S, logdens

def E_step(S, logdens):
    return numpy.exp(S - logdens)

def M_step(E, X, psi = 0.01):
    M, N = E.shape
    new_GMM = []
    Zg_vec = numpy.zeros(M)
    for i in range(M):
        gamma = E[i, :]
        Zg = gamma.sum()
        Zg_vec[i] = Zg
        Fg = (vrow(gamma) * X).sum(1)
        Sg = numpy.dot(X, (vrow(gamma) * X).T)
        w = Zg / X.shape[1]
        mu = vcol(Fg / Zg)
        sigma = Sg / Zg - numpy.dot(mu, mu.T)
        U, s, _ = numpy.linalg.svd(sigma)
        s[s<psi] = psi
        covNew = numpy.dot(U, mcol(s)*U.T)
        new_GMM.append([w, mu, covNew])
    return new_GMM

def GMM_EM_estimation(X, gmm, delta):
    referenceGMM = gmm
    S, logdens = logpdf_GMM(X, gmm)
    E = E_step(S, logdens)
    new_GMM = M_step(E, X)
    S_new, logdens_new = logpdf_GMM(X, new_GMM)
    while((logdens_new.mean() - logdens.mean()) > delta):
        #print("old logs avg", logdens.mean(), "new logs avg", logdens_new.mean())
        referenceGMM = new_GMM
        S, logdens = S_new, logdens_new
        E = E_step(S, logdens)
        new_GMM = M_step(E, X)
        S_new, logdens_new = logpdf_GMM(X, new_GMM)  
    return referenceGMM

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

def M_step_diagonal(E, X, psi = 0.01):
    M, N = E.shape
    new_GMM = []
    Zg_vec = numpy.zeros(M)
    for i in range(M):
        gamma = E[i, :]
        Zg = gamma.sum()
        Zg_vec[i] = Zg
        Fg = (vrow(gamma) * X).sum(1)
        Sg = numpy.dot(X, (vrow(gamma) * X).T)
        w = Zg / X.shape[1]
        mu = vcol(Fg / Zg)
        sigma = Sg / Zg - numpy.dot(mu, mu.T)
        covNew = sigma * numpy.eye(sigma.shape[0])
        U, s, _ = numpy.linalg.svd(covNew)
        s[s<psi] = psi
        covNew = numpy.dot(U, mcol(s)*U.T)
        new_GMM.append([w, mu, covNew])
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

def LBG_algorithm_diagonal(iterations, D, gmm, alpha=0.1, psi = 0.01):
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
        start_gmm = GMM_EM_estimation_diagonal(D, new_gmm, 10**-6)
    return start_gmm


def M_step_tied(E, X, psi = 0.01):
    M, N = E.shape
    new_GMM = []
    Zg_vec = numpy.zeros(M)
    w_vec = numpy.zeros(M)
    mu_vec = []
    sigma = 0
    for i in range(M):
        gamma = E[i, :]
        Zg = gamma.sum()
        Zg_vec[i] = Zg
        Fg = (vrow(gamma) * X).sum(1)
        Sg = numpy.dot(X, (vrow(gamma) * X).T)
        w_vec[i] = Zg / X.shape[1]
        mu_vec.append(vcol(Fg / Zg))
        sigma += Zg * (Sg / Zg - numpy.dot(mu_vec[i], mu_vec[i].T))    
    
    sigma /= N
    U, s, _ = numpy.linalg.svd(sigma)
    s[s<psi] = psi
    covNew = numpy.dot(U, mcol(s)*U.T)
    for i in range(M):
        new_GMM.append([w_vec[i], mu_vec[i], covNew])
    return new_GMM
        
def GMM_EM_estimation_tied(X, gmm, delta):
    S, logdens=logpdf_GMM(X, gmm)
    while(1):
        E = E_step(S, logdens)
        new_GMM = M_step_tied(E, X)
        S_new, logdens_new=logpdf_GMM(X, new_GMM)
        if((logdens_new.mean() - logdens.mean()) < delta):
            referenceGMM = new_GMM
            #print("Final average log-likelihood:", logdens_new.mean())
            break      
        referenceGMM = new_GMM
        S, logdens=S_new, logdens_new
    return referenceGMM

def LBG_algorithm_tied(iterations, D, gmm, alpha=0.1, psi = 0.01):
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
        start_gmm = GMM_EM_estimation_tied(D, new_gmm, 10**-6)
    return start_gmm
    
def plot_min_cdf_error_gaussian_mixture_models(model_name, first_dcf, second_dcf, first_label, second_label):

    plt.figure()
    plt.title(model_name)
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    iterations = range(7)
    x_axis = numpy.arange(len(iterations))
    bounds = numpy.array(iterations)
    plt.bar(x_axis + 0.00, first_dcf, width=0.25, linewidth=1.0, edgecolor='black', color="Red", label=first_label)
    plt.bar(x_axis + 0.25, second_dcf, width=0.25, linewidth=1.0, edgecolor='black', color="Orange", label=second_label)

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("minDCF/" + model_name + "_" + first_label + "_" + second_label + ".png")
    plt.close()

def correlationsPlot(D, L, desc = ''):
    cmap = ['Greys', 'Blues', 'Reds']
    corrCoeff = {
        0: numpy.abs(numpy.corrcoef(D)),
        1: numpy.abs(numpy.corrcoef(D[:, L == 0])),
        2: numpy.abs(numpy.corrcoef(D[:, L == 1]))
    }
    for i in range(3):
        plt.figure()
        plt.imshow(corrCoeff[i], cmap = cmap[i], interpolation = 'nearest')
        plt.savefig('heatmaps/' + desc + 'heatmap_%d.jpg' % i)
        
def plot_minDCF_lr(l, y5, y1, y9, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    plt.plot(l, numpy.array(y5), label='minDCF(π~ = 0.5)', color='r')
    plt.plot(l, numpy.array(y1), label='minDCF(π~ = 0.1)', color='b')
    plt.plot(l, numpy.array(y9), label='minDCF(π~ = 0.9)', color='g')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'minDCF/lr_minDCF_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plotDCFpoly(x, y, xlabel, m, variant, prior=0.5):
    fig = plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=' + str(prior) + ' - c=0', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=' + str(prior) + ' - c=1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=' + str(prior) + ' - c=10', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=' + str(prior) + ' - c=30', color='m')


    plt.xlim([1e-5, 1e-1])
    plt.xscale("log")
    plt.legend(["min DCF prior=" + str(prior) + " - c=0", "min DCF prior=" + str(prior) + " - c=1", 
                'min DCF prior=' + str(prior) + ' - c=10', 'min DCF prior=' + str(prior) + ' - c=30'])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('minDCF/svm_poly_minDCF_%s_PCA%s_prior%s.jpg' % (variant, str(m), str(prior)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

def plot_minDCF_svm_poly(C, y5, y1, y9, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    plt.plot(C, numpy.array(y5), label='minDCF(π~ = 0.5)', color='r')
    plt.plot(C, numpy.array(y1), label='minDCF(π~ = 0.1)', color='b')
    plt.plot(C, numpy.array(y9), label='minDCF(π~ = 0.9)', color='g')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('c')
    plt.ylabel('minDCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'minDCF/svm_poly_minDCF_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plotDCFRBF(x, y, xlabel, m, prior=0.5):
    fig = plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=' + str(prior) + ' - logγ=-5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=' + str(prior) + ' - logγ=-4', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=' + str(prior) + ' - logγ=-3', color='g')
    
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=" + str(prior) + " - logγ=-5", "min DCF prior=" + str(prior) + " - logγ=-4", 
                "min DCF prior=" + str(prior) + " - logγ=-3"])
    
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('minDCF/svm_RBF_minDCF_%s.jpg' % (str(m) + str(prior)), dpi=300, bbox_inches='tight')
    plt.close(fig)