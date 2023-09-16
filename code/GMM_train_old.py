import numpy
from Library_gianmarco import logpdf_GMM, Bayes_risk_min_cost, Z_norm, PCA, Ksplit, plot_minDCF_lr
from library import load
import scipy

if __name__ == '__main__':
    D = []
    L = []

    [D, L] = load('../Train.txt')
