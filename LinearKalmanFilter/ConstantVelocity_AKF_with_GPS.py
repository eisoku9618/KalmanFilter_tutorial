#!/usr/bin/python
# -*- coding: utf-8 -*-

# http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Adaptive-Kalman-Filter-CV.ipynb
# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

import argparse
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

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plotData(data_dict, vx, vy, goal_x, goal_y):
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
    axarr[1, 0].semilogy(range(l), [r[0, 0] for r in data_dict["R"]], label='$x$')
    axarr[1, 0].semilogy(range(l), [r[1, 1] for r in data_dict["R"]], label='$y$')
    axarr[1, 0].semilogy(range(l), [r[2, 2] for r in data_dict["R"]], label='$\dot x$')
    axarr[1, 0].semilogy(range(l), [r[3, 3] for r in data_dict["R"]], label='$\dot y$')
    axarr[1, 0].set_xlabel('Filter Step')
    axarr[1, 0].set_ylabel('')
    axarr[1, 0].set_title('Measurement Uncertainty $R$ (Adaptive)')
    axarr[1, 0].legend(loc='best')

    l = len(data_dict["x"])
    axarr[1, 1].scatter([x[0, 0] for x in data_dict["x"]], [x[1, 0] for x in data_dict["x"]], s=20, label='State', c='k')
    axarr[1, 1].scatter(data_dict["x"][0][0, 0], data_dict["x"][0][1, 0], s=30, label='Start', c='b')
    axarr[1, 1].scatter(data_dict["x"][-1][0, 0], data_dict["x"][-1][1, 0], s=30, label='Goal', c='r')
    axarr[1, 1].scatter(goal_x, goal_y, s=30, label='True Goal', c='y')
    axarr[1, 1].set_xlabel('X')
    axarr[1, 1].set_ylabel('Y')
    axarr[1, 1].set_title('Position')
    axarr[1, 1].legend(loc='lower right')
    axarr[1, 1].set_aspect('equal')

    for i in range(l):
        if i % 50 == 0:
            plot_cov_ellipse(data_dict["P"][i][0:2, 0:2], np.array([data_dict["x"][i][0, 0], data_dict["x"][i][1, 0]]), ax=axarr[1, 1],
                             nstd=400, alpha=0.5, color='green')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive Kalman Filter sample')
    parser.add_argument('-t', type=float, default=0.5, help='time step')
    parser.add_argument('--sra', type=float, default=2.0, help='true GPS standard deviation of R which value is known only by God')
    parser.add_argument('--srb', type=float, default=1.0, help='true Speed Detector standard deviation of R which value is known only by God')
    parser.add_argument('--sraf', type=float, default=1.0, help='user setting GPS standard deviation of R which value is assumed right')
    parser.add_argument('--srbf', type=float, default=0.5, help='user setting Speed Detector standard deviation of R which value is assumed right')
    parser.add_argument('-N', type=int, default=500, help='number of trials')
    parser.add_argument('--vx', type=float, default=40, help='ground truth of velocity x')
    parser.add_argument('--vy', type=float, default=20, help='ground truth of velocity y')
    parser.add_argument('--noisea', type=float, default=100, help='unexpected observation noise of x')
    parser.add_argument('--noiseb', type=float, default=50, help='unexpected observation noise of velocity y')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--adaptive', dest='akf', action='store_true', help='use adaptive kalman filter')
    group.add_argument('--non-adaptive', dest='akf', action='store_false', help='do not use adaptive kalman filter')
    parser.set_defaults(akf=True)

    # parameter
    dt = parser.parse_args().t
    ra = parser.parse_args().sraf**2
    rb = parser.parse_args().srbf**2
    sv = 1.0
    num = parser.parse_args().N
    vx = parser.parse_args().vx # in X
    vy = parser.parse_args().vy # in Y

    # initialize
    x0 = np.matrix([[0.0, 0.0, 0, 0]]).T
    P0 = 1.0 * np.eye(4)
    R = np.matrix([[ra, 0.0, 0.0, 0.0],
                   [0.0, ra, 0.0, 0.0],
                   [0.0, 0.0, rb, 0.0],
                   [0.0, 0.0, 0.0, rb]])
    G = np.matrix([[0.5*dt**2],
                   [0.5*dt**2],
                   [dt],
                   [dt]])
    Q = G * G.T * sv**2
    F = np.matrix([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    H = 1.0 * np.eye(4)
    AKF = LinearKalmanFilter(x0, P0, Q, R, F, H)

    # input measurement
    mx = np.linspace(0.0 + vx * dt * 1, 0.0 + vx * dt * num, num) + parser.parse_args().sra * np.random.randn(num)
    my = np.linspace(0.0 + vy * dt * 1, 0.0 + vy * dt * num, num) + parser.parse_args().sra * np.random.randn(num)
    mvx = np.array(vx + parser.parse_args().srb * np.random.randn(num))
    mvy = np.array(vy + parser.parse_args().srb * np.random.randn(num))
    # some different error somewhere in the measurements
    mx[(2 * num/8):(3 * num/8)] = np.linspace(0.0 + vx * dt * (2 * num/8), 0.0 + vx * dt * (3 * num/8), num/8) + parser.parse_args().noisea * np.random.randn(num/8)
    mvy[(6* num/8):(7 * num/8)] = np.array(vy + parser.parse_args().noiseb * np.random.randn(num/8))
    measurements = np.vstack((mx, my, mvx, mvy))
    for i in range(len(measurements[0])):
        if parser.parse_args().akf:
            n = 10
            if i > n:
                R = np.matrix([[np.std(measurements[0, (i-n):i] - np.linspace(vx * dt * (i-n), vx * dt * i, n))**2, 0.0, 0.0, 0.0],
                               [0.0, np.std(measurements[1, (i-n):i] - np.linspace(vy * dt * (i-n), vy * dt * i, n))**2, 0.0, 0.0],
                               [0.0, 0.0, np.std(measurements[2, (i-n):i])**2, 0.0],
                               [0.0, 0.0, 0.0, np.std(measurements[3, (i-n):i])**2],
                           ])
        AKF.proc(Q, measurements[:, i].reshape(4, 1), R)
    plotData(AKF.getData(), vx, vy, vx*dt*num, vy*dt*num)
