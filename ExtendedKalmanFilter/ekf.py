#!/usr/bin/env python
# -*- coding: utf-8 -*-

# original : http://satomacoto.blogspot.jp/2011/06/python_22.html

import numpy as np
import scipy as sp
from scipy.optimize.slsqp import approx_jacobian
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def main():
    # 初期化
    T = 500 # 観測数
    p0 = (-100,-100); p1 = (100,0); p2 = (0,100) # 観測値の座標
    x = np.mat([[0],[0]]) # 初期位置
    X = [np.mat([[0],[0]])] # 状態
    Y = [np.mat([[0],[0],[0]])] # 観測

    # state x = A * x_ + B * u + w, w~N(0,Q)
    A = np.mat([[1,0],[0,1]])
    B = np.mat([[1,0],[0,1]])
    u = np.mat([[2],[2]])
    Q = np.mat([[1,0],[0,1]])
    # observation Y = h(x) + v, v~N(0,R)
    R = np.mat([[2,0,0],[0,2,0],[0,0,2]])
    R_bad = np.mat([[1000,0,0],[0,1000,0],[0,0,1000]])
    def h_(x,p):
        return ((x[0]-p[0])**2 + (x[1]-p[1])**2)**.5
    def h(x):
        x = (x[0,0],x[1,0])
        return np.mat([[h_(x,p0)],[h_(x,p1)],[h_(x,p2)]])
    def Jh0(x):
        """
        解析的に求めるh(x)のヤコビアン
        """
        x = (x[0,0],x[1,0])
        return np.mat([[(x[0]-p0[0])/h_(x,p0),(x[1]-p0[1])/h_(x,p0)],
                       [(x[0]-p1[0])/h_(x,p1),(x[1]-p1[1])/h_(x,p1)],
                       [(x[0]-p2[0])/h_(x,p2),(x[1]-p2[1])/h_(x,p2)]])
    def Jh1(x):
        """
        数値的に求めるh(x)のヤコビアン
        """
        x = (x[0,0],x[1,0])
        h = lambda x: np.asfarray([h_(x,p0),h_(x,p1),h_(x,p2)])
        return np.mat(approx_jacobian(x,h,np.sqrt(np.finfo(float).eps)))

    # 観測データの生成
    for i in range(T):
        x = A * x + B * u + np.random.multivariate_normal([0,0],Q,1).T
        X.append(x)
        if i > T / 8 * 5 and i < T / 8 * 6:
            y = h(x) + np.random.multivariate_normal([0,0,0],R_bad,1).T
        else:
            y = h(x) + np.random.multivariate_normal([0,0,0],R,1).T
        Y.append(y)

    # EKF
    mu = np.mat([[0],[0]])
    Sigma = np.mat([[10,0],[0,10]])
    M = [mu] # 推定
    SM = [Sigma]
    for i in range(T):
        # prediction
        mu_ = A * mu + B * u
        Sigma_ = Q + A * Sigma * A.T
        # update
        C = Jh0(mu_) # 解析的
        # C = Jh1(mu_) # 数値的
        yi = Y[i+1] - h(mu_)
        S = C * Sigma_ * C.T + R
        K = Sigma_ * C.T * S.I
        mu = mu_ + K * yi
        Sigma = Sigma_ - K * C * Sigma_
        M.append(mu)
        SM.append(Sigma)

    # 描画
    a,b = np.array(np.concatenate(X,axis=1))
    plt.plot(a,b,'rs-', label='Ground Trueth')
    a,b = np.array(np.concatenate(M,axis=1))
    plt.plot(a,b,'bo-', label='Estimated')
    plt.axis('equal')
    for i in range(len(a)):
        if i % 10 == 0:
            plot_cov_ellipse(SM[i], np.array([a[i], b[i]]), nstd=10, alpha=0.5, color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
