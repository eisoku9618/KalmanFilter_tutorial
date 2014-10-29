# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np

class AdaptiveKalmanFilter(object):
    def __init__(self, x0, P0, Q0, R0, F, H, B=None):
        self.x = x0
        self.P = P0
        self.Q = Q0
        self.R = R0
        self.F = F
        self.H = H
        self.B = B
        self.data = {"x":[], "P":[], "Q":[], "R":[]}
        self.saveData()

    def saveData(self):
        self.data["x"].append(self.x)
        self.data["P"].append(self.P)
        self.data["Q"].append(self.Q)
        self.data["R"].append(self.R)

    def getData(self):
        return self.data

    def prediction(self, Q, u=None):
        if self.B is None:
            self.x = self.F * self.x
        else:
            self.x = self.F * self.x + self.B * u
        self.P = self.F * self.P * self.F.T + Q
        self.Q = Q

    def correction(self, Z, R):
        S = self.H * self.P * self.H.T + R
        K = (self.P * self.H.T) * np.linalg.pinv(S)
        y = Z - (self.H * self.x)
        self.x = self.x + (K * y)
        self.P = (np.eye(4) - (K * self.H)) * self.P
        self.R = R

    def proc(self, Q, Z, R, u=None):
        self.prediction(Q, u)
        self.correction(Z, R)
        self.saveData()
