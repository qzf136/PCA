import numpy as np
from math import *
import matplotlib.pyplot as plt


def generate_data():
    mean = [5, 5]
    cov = [[0.5, 0], [0, 6]]
    pre = np.random.multivariate_normal(mean, cov, 50)
    X = np.insert(pre, 2, 1, axis=1)
    transltion = np.mat([[1,0,0], [0, cos(pi/18), -sin(pi/18)], [0, sin(pi/18), cos(pi/18)]])
    X = X * transltion
    return pre, X


def PCA(X, d):
    mean = np.mean(X, axis=0)
    X_cen = X - mean
    cov = X_cen.T * X_cen
    w = eigenvector(cov, d)
    Z = X_cen * w
    X2 = Z * w.T + mean
    return Z, X2


def eigenvector(cov, d):
    e, v = np.linalg.eig(cov)
    sort = np.argsort(e)
    index = sort[:-(d+1):-1]
    w = np.asarray(v[:,index], float)
    return w


def show(X, Z):
    X = np.array(X)
    plt.scatter(X[:,0], X[:,1])
    Z = np.array(Z)
    plt.scatter(Z[:,0], Z[:,1])
    plt.show()


if __name__ == '__main__':
    pre, X = generate_data()
    Z, X2 = PCA(X, 2)
    show(pre, Z)
