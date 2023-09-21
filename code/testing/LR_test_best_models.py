import sys
sys.path.append('../')

import numpy
from library import load, logreg_obj_wrapper, logreg, Bayes_risk_min_cost, Z_norm, vcol
import scipy

def test(DTR, LTR, DTE, LTE, l):
    priors = [0.5, 0.1, 0.9]
    for p in priors:
        print("prior:", p)
        logreg_obj = logreg_obj_wrapper(DTR, LTR, l, p)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, 
                                                x0=numpy.zeros(DTR.shape[0] + 1),
                                                approx_grad=True)
        scores, LP = logreg(x, DTE)
        print("minDCF 0.5: %.3f" %Bayes_risk_min_cost(0.5, 1, 1, scores, LTE))
        print("minDCF 0.1: %.3f" %Bayes_risk_min_cost(0.1, 1, 1, scores, LTE))
        print("minDCF 0.9: %.3f" %Bayes_risk_min_cost(0.9, 1, 1, scores, LTE))
        print()

if __name__ == '__main__':

    [DTR, LTR] = load('../Train.txt')
    [DTE, LTE] = load('../Test.txt')
        
    print("LR RAW features lambda = 0.1")    
    test(DTR, LTR, DTE, LTE, l = 0.1)
    print()
    
    Dstd = vcol(numpy.std(DTR, axis=1))
    Dmean = vcol(DTR.mean(1))
    DTE = (DTE - Dmean) / Dstd
    print("LR Z-Norm features lambda = 10**(-3)")
    test(Z_norm(DTR), LTR, DTE, LTE, l = 10**(-3))
    print()
    