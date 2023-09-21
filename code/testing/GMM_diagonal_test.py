import sys
sys.path.append('../')

import numpy
from library import load, LBG_algorithm_diagonal, Bayes_risk_min_cost, PCA, Ksplit, Z_norm, logpdf_GMM, vcol
import matplotlib.pyplot as plt


def train(D, L):
    K = 5
    folds, labels = Ksplit(D, L, seed=0, K=K)    
    min_cost_vec = []
    for it in range(7):
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
            
            EM_GMM_D0 =LBG_algorithm_diagonal(it, D0, GMM_D0)
            
            EM_GMM_D1 =LBG_algorithm_diagonal(it, D1, GMM_D1)
        
            _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
        
            _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
            s=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
            scores.append(s)
            LTEs.append(LTE)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(LTEs)
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, orderedLabels)
        min_cost_vec.append(min_cost)
    return min_cost_vec
    
def test(DTR, LTR, DTE, LTE):
    print("Testing")
    min_cost_vec = []
    for it in range(7):
        print("num iterations:", 2**it)
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
        
        EM_GMM_D0 =LBG_algorithm_diagonal(it, D0, GMM_D0)
        
        EM_GMM_D1 =LBG_algorithm_diagonal(it, D1, GMM_D1)
    
        _, logdens_D0=logpdf_GMM(DTE, EM_GMM_D0)
    
        _, logdens_D1=logpdf_GMM(DTE, EM_GMM_D1)
        scores=numpy.log(numpy.exp(logdens_D1)/ numpy.exp(logdens_D0))
        min_cost = Bayes_risk_min_cost(0.5, 1, 1, scores, LTE)
        min_cost_vec.append(min_cost)
        print("prior:", 0.5, "min cost: %.3f" %min_cost)
        print("prior:", 0.1, "min cost: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        print("prior:", 0.9, "min cost: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
        print()
    return min_cost_vec
        
def plot_min_cdf_error_gaussian_mixture_models_test(model_name, dcf_train_first, dcf_test_first, dcf_train_second, dcf_test_second, first_label, second_label):
    
    plt.figure()
    plt.title(model_name)
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    iterations = range(7)
    x_axis = numpy.arange(len(iterations)) * 1.25
    bounds = numpy.array(iterations)
    plt.bar(x_axis + 0.00, dcf_train_first, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label=first_label+" [Eval]", linestyle="--", hatch="//")
    plt.bar(x_axis + 0.25, dcf_test_first, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label=first_label+" [Val]")

    plt.bar(x_axis + 0.50, dcf_train_second, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label=second_label+" [Val]", linestyle="--", hatch="//")

    plt.bar(x_axis + 0.75, dcf_test_second, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label=second_label+" [Eval]")

    plt.xticks([r * 1.25 + 0.375 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("minDCF/test" + model_name + "_" + first_label + "_" + second_label + '.jpg', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    print("GMM diagonal")
    [D, L] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')
    
    Dz = Z_norm(D)
    Dstd = vcol(numpy.std(D, axis=1))
    Dmean = vcol(D.mean(1))
    DTEz = (DTE - Dmean) / Dstd
    
    print("Raw")
    min_cost_vec = train(D, L)
    test_min_cost_vec = test(D, L, DTE, LTE)
    
    print("Z-norm")
    min_cost_vec_znorm = train(Dz, L)
    test_min_cost_vec_znorm = test(Dz, L, DTEz, LTE)
    
    plot_min_cdf_error_gaussian_mixture_models_test("GMM diagonal", min_cost_vec, test_min_cost_vec, min_cost_vec_znorm, test_min_cost_vec_znorm, "RAW", "Z-Norm")
    
    print("PCA 12")
    
    P = PCA(D, 12)
    D = numpy.dot(P.T, D)
    
    Pz = PCA(Dz, 12)
    Dz = numpy.dot(Pz.T, Dz)
    
    DTE = numpy.dot(P.T, DTE)
    DTEz = numpy.dot(Pz.T, DTEz)
    
    print("Raw")
    min_cost_vec = train(D, L)
    test_min_cost_vec = test(D, L, DTE, LTE)
    
    print("Z-norm")
    min_cost_vec_znorm = train(Dz, L)
    test_min_cost_vec_znorm = test(Dz, L, DTEz, LTE)
    plot_min_cdf_error_gaussian_mixture_models_test("GMM diagonal PCA-12", min_cost_vec, test_min_cost_vec, min_cost_vec_znorm, test_min_cost_vec_znorm, "RAW", "Z-Norm")
    