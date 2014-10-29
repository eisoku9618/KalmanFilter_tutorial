#!/usr/bin/python
# -*- coding: utf-8 -*-

# http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Adaptive-Kalman-Filter-CV.ipynb
# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

# import sys,os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# print sys.path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from LinearKalmanFilter import *

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    print vals, theta
    # width, height = 2 * nstd * np.sqrt(vals)
    width, height = 2 * nstd * vals
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plotData(data_dict, vx, vy):
    f, axarr = plt.subplots(2, 2)

    l = len(data_dict["P"])
    axarr[0, 0].semilogy(range(l), [p[0, 0] for p in data_dict["P"]], label='$x$')
    axarr[0, 0].semilogy(range(l), [p[1, 1] for p in data_dict["P"]], label='$y$')
    axarr[0, 0].semilogy(range(l), [p[2, 2] for p in data_dict["P"]], label='$\dot x$')
    axarr[0, 0].semilogy(range(l), [p[3, 3] for p in data_dict["P"]], label='$\dot y$')
    axarr[0, 0].set_xlabel('Filter Step')
    axarr[0, 0].set_title('Uncertainty (Elements from Matrix $P$)')
    axarr[0, 0].legend(loc='best')

    l = len(data_dict["x"])
    axarr[0, 1].plot(range(l), [x[2, 0] for x in data_dict["x"]], label='$\dot x$')
    axarr[0, 1].plot(range(l), [x[3, 0] for x in data_dict["x"]], label='$\dot y$')
    axarr[0, 1].axhline(vx, color='#999999', label='$\dot x_{real}$')
    axarr[0, 1].axhline(vy, color='#999999', label='$\dot y_{real}$')
    axarr[0, 1].set_xlabel('Filter Step')
    axarr[0, 1].set_title('Estimate (Elements from State Vector $x$)')
    axarr[0, 1].legend(loc='best')
    axarr[0, 1].set_ylabel('Velocity')

    l = len(data_dict["R"])
    axarr[1, 0].semilogy(range(l), [r[0, 0] for r in data_dict["R"]], label='$\dot x$')
    axarr[1, 0].semilogy(range(l), [r[1, 1] for r in data_dict["R"]], label='$\dot y$')
    axarr[1, 0].set_xlabel('Filter Step')
    axarr[1, 0].set_ylabel('')
    axarr[1, 0].set_title('Measurement Uncertainty $R$ (Adaptive)')
    axarr[1, 0].legend(loc='best')

    l = len(data_dict["x"])
    axarr[1, 1].scatter([x[0, 0] for x in data_dict["x"]], [x[1, 0] for x in data_dict["x"]], s=20, label='State', c='k')
    axarr[1, 1].scatter(data_dict["x"][0][0, 0], data_dict["x"][0][1, 0], s=20, label='State', c='k')
    axarr[1, 1].scatter(data_dict["x"][-1][0, 0], data_dict["x"][-1][1, 0], s=20, label='Goal', c='r')
    axarr[1, 1].set_xlabel('X')
    axarr[1, 1].set_ylabel('Y')
    axarr[1, 1].set_title('Position')
    axarr[1, 1].legend(loc='best')

    for i in range(l):
        if i % 10 == 0:
            plot_cov_ellipse(data_dict["P"][i][0:2, 0:2], np.array([data_dict["x"][i][0, 0], data_dict["x"][i][1, 0]]), ax=axarr[1, 1],
                             nstd=3, alpha=0.5, color='green')

    plt.show()

if __name__ == '__main__':
    # parameter
    dt = 0.5
    ra = 1.0**2
    sv = 1.0
    num = 200
    vx = 20 # in X
    vy = 40 # in Y

    # initialize
    x0 = np.matrix([[0.0, 0.0, 0, 0]]).T
    P0 = 1.0 * np.eye(4)
    R = np.matrix([[ra, 0.0],
                   [0.0, ra]])
    G = np.matrix([[0.5*dt**2],
                   [0.5*dt**2],
                   [dt],
                   [dt]])
    Q = G * G.T * sv**2
    F = np.matrix([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    H = np.matrix([[0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    AKF = AdaptiveKalmanFilter(x0, P0, Q, R, F, H)

    # input measurement
    mx = np.array(vx + np.random.randn(num))
    my = np.array(vy + np.random.randn(num))
    # some different error somewhere in the measurements
    my[(2* num/4):(3 * num/4)] = np.array(vy + 50.0 * np.random.randn(num/4))
    measurements = np.vstack((mx, my))
    for i in range(len(measurements[0])):
        n = 10
        if i > n:
            R = np.matrix([[np.std(measurements[0, (i-n):i])**2, 0.0],
                           [0.0, np.std(measurements[1, (i-n):i])**2]])
        AKF.proc(Q, measurements[:, i].reshape(2, 1), R)
    plotData(AKF.getData(), vx, vy)
    # print AKF.getData()["x"][0]
