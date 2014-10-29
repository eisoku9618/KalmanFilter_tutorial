#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import *

def lkf(T, Z, U, x0, P0, F, H, Q, R):
    '''Linear Kalman Filter

    - 状態方程式
        x_ = F * x + u + w, w ~ N(0,Q)
    - 観測方程式
        z = H * x_ + v, v ~ N(0,R)

    Parameters
    ==========
    - T : ステップ数
    - Z : 観測列
    - U : 入力列
    - x0 : 初期状態推定値
    - P0 : 初期誤差共分散行列
    - F, H, Q, R : カルマンフィルタの係数

    Returns
    =======
    - X : 状態推定値列
    '''

    x = x0 # 初期状態推定値
    P = P0 # 初期誤差共分散行列

    X_est = [x] # 状態推定値列
    P_est = [P]

    for i in range(T):
        # 推定
        x_ = F * x + U[i]
        P_ = F * P * F.T + Q

        # 更新
        e = Z[i+1] - H * x_
        S = R + H * P_ * H.T
        K = P_ * H.T * S.I
        x = K * e + x_
        P = P_ - K * H * P_
        X_est.append(x)
        P_est.append(P)

    return X_est, P_est

def main():
    q = 0.5
    r = 5.0
    F = np.mat([0])
    H = np.mat([1])
    Q = np.mat([q])
    R = np.mat([r])

    # 観測のテストデータの生成
    T = 100 # 観測数
    x = np.mat([0]) # 初期位置
    X = [x] # 状態列
    Z = [x] # 観測列
    u = np.mat([0]) # 入力（一定）
    U = [u] # 入力列
    GT = [u]
    for i in range(T):
        x = F * x + GT[i] + np.random.multivariate_normal([0], Q, 1).T
        X.append(x)
        z = H * x + np.random.multivariate_normal([0], R, 1).T
        Z.append(z)
        if i > T / 4 * 1 and i < T / 4 * 3:
            GT.append(np.mat([10]))
        else:
            GT.append(u)
        U.append(u)

    # LKF
    x0 = np.mat([0]) # 初期状態推定値
    P0 = np.mat([0]) # 初期誤差共分散行列
    X_est, P_est = lkf(T, Z, U, x0, P0, F, H, Q, R)
    print P_est
    X_est_lo = []
    X_est_hi = []
    for i in range(len(X_est)):
        X_est_lo.append(X_est[i] - sqrt(P_est[i]))
        X_est_hi.append(X_est[i] + sqrt(P_est[i]))

    # 描画
    plt.plot(range(T+1), [tmp.base[0] for tmp in X_est], 'g-', label='estimated')
    plt.plot(range(T+1), [tmp.base[0] for tmp in X_est_lo], 'go')
    plt.plot(range(T+1), [tmp.base[0] for tmp in X_est_hi], 'go')
    plt.plot(range(T+1), [tmp.base[0] for tmp in GT], 'b-', label='ground truth')
    plt.plot(range(T+1), [tmp.base[0] for tmp in Z], 'r-', label='observed')
    plt.legend(loc=0)
    plt.title("Q : %s, R : %s" % (q, r))
    plt.show()

if __name__ == '__main__':
    main()
