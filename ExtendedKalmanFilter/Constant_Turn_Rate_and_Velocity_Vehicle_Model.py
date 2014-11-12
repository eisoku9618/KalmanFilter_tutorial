#!/usr/bin/python
# -*- coding: utf-8 -*-

# original : http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRV.ipynb

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
import urllib
import os.path

numstates = 5
dt = 1.0 / 50.0

P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

sGPS = 0.5*8.8*dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse = 0.1*dt # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity = 8.8*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw = 1.0*dt # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sYaw**2])

varGPS = 6.0 # Standard Deviation of GPS Measurement
varspeed = 1.0 # Variance of the speed measurement
varyaw = 0.1 # Variance of the yawrate measurement
R = np.matrix([[varGPS**2, 0.0, 0.0, 0.0],
               [0.0, varGPS**2, 0.0, 0.0],
               [0.0, 0.0, varspeed**2, 0.0],
               [0.0, 0.0, 0.0, varyaw**2]])


I = np.eye(numstates)

datafile = '2014-03-26-000-Data.csv'

if not os.path.exists(datafile):
    urllib.urlretrieve('https://raw.githubusercontent.com/balzer82/Kalman/master/2014-03-26-000-Data.csv', datafile)

date, \
time, \
millis, \
ax, \
ay, \
az, \
rollrate, \
pitchrate, \
yawrate, \
roll, \
pitch, \
yaw, \
speed, \
course, \
latitude, \
longitude, \
altitude, \
pdop, \
hdop, \
vdop, \
epe, \
fix, \
satellites_view, \
satellites_used, \
temp = np.loadtxt(datafile, delimiter=',', unpack=True,
                  converters={1: mdates.strpdate2num('%H%M%S%f'),
                              0: mdates.strpdate2num('%y%m%d')},
                  skiprows=1)

course =(-course+90.0)

RadiusEarth = 6378388.0 # m
arc= 2.0*np.pi*(RadiusEarth+altitude)/360.0 # m/°

dx = arc * np.cos(latitude*np.pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
dy = arc * np.hstack((0.0, np.diff(latitude))) # in m

mx = np.cumsum(dx)
my = np.cumsum(dy)

ds = np.sqrt(dx**2+dy**2)

GPS=np.hstack((True, (np.diff(ds)>0.0).astype('bool'))) # GPS Trigger for Kalman Filter


x = np.matrix([[mx[0], my[0], course[0]/180.0*np.pi, speed[0]/3.6+0.001, yawrate[0]/180.0*np.pi]]).T

U=float(np.cos(x[2])*x[3])
V=float(np.sin(x[2])*x[3])


measurements = np.vstack((mx, my, speed/3.6, yawrate/180.0*np.pi))
# Lenth of the measurement
m = measurements.shape[1]

x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Pddx=[]
Pddy=[]
Kx = []
Ky = []
Kdx= []
Kdy= []
Kddx=[]
dstate=[]
Qx=[]
Qy=[]
Qpsi=[]
Qv=[]
Qdpsi=[]

for filterstep in range(m):
    # if filterstep > m / 8 * 5 and filterstep < m / 8 * 7:
    #     Q = np.diag([(sGPS * 2)**2, (sGPS * 2)**2, sCourse**2, sVelocity**2, sYaw**2])
    # else:
    #     Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sYaw**2])

    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # see "Dynamic Matrix"
    if np.abs(yawrate[filterstep])<0.0001: # Driving straight
        x[0] = x[0] + x[3]*dt * np.cos(x[2])
        x[1] = x[1] + x[3]*dt * np.sin(x[2])
        x[2] = x[2]
        x[3] = x[3]
        x[4] = 0.0000001 # avoid numerical issues in Jacobians
        dstate.append(0)
    else: # otherwise
        x[0] = x[0] + (x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2]))
        x[1] = x[1] + (x[3]/x[4]) * (-np.cos(x[4]*dt+x[2])+ np.cos(x[2]))
        x[2] = (x[2] + x[4]*dt + np.pi) % (2.0*np.pi) - np.pi
        x[3] = x[3]
        x[4] = x[4]
        dstate.append(1)

    # Calculate the Jacobian of the Dynamic Matrix A
    # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
    a13 = float((x[3]/x[4]) * (np.cos(x[4]*dt+x[2]) - np.cos(x[2])))
    a14 = float((1.0/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a15 = float((dt*x[3]/x[4])*np.cos(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a23 = float((x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a24 = float((1.0/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    a25 = float((dt*x[3]/x[4])*np.sin(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    JA = np.matrix([[1.0, 0.0, a13, a14, a15],
                    [0.0, 1.0, a23, a24, a25],
                    [0.0, 0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])

    # Project the error covariance ahead
    P = JA*P*JA.T + Q

    # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    hx = np.matrix([[float(x[0])],
                    [float(x[1])],
                    [float(x[3])],
                    [float(x[4])]])

    if GPS[filterstep]: # with 10Hz, every 5th step
        JH = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])
    else: # every other step
        JH = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])

    S = JH*P*JH.T + R
    K = (P*JH.T) * np.linalg.inv(S)

    # Update the estimate via
    Z = measurements[:,filterstep].reshape(JH.shape[0],1)
    y = Z - (hx)                         # Innovation or Residual
    x = x + (K*y)

    # Update the error covariance
    P = (I - (K*JH))*P


    # Save states for Plotting
    x0.append(float(x[0]))
    x1.append(float(x[1]))
    x2.append(float(x[2]))
    x3.append(float(x[3]))
    x4.append(float(x[4]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Pddx.append(float(P[4,4]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))
    Kddx.append(float(K[4,0]))
    Qx.append(Q[0, 0])
    Qy.append(Q[1, 1])
    Qpsi.append(Q[2, 2])
    Qv.append(Q[3, 3])
    Qdpsi.append(Q[4, 4])


f, axarr = plt.subplots(3, 3)


axarr[0, 0].semilogy(range(m),Px, label='$x$')
axarr[0, 0].step(range(m),Py, label='$y$')
axarr[0, 0].step(range(m),Pdx, label='$\psi$')
axarr[0, 0].step(range(m),Pdy, label='$v$')
axarr[0, 0].step(range(m),Pddx, label='$\dot \psi$')
axarr[0, 0].set_xlabel('Filter Step')
axarr[0, 0].set_ylabel('')
axarr[0, 0].set_title('Uncertainty (Elements from Matrix $P$)')
axarr[0, 0].legend(loc='best',prop={'size':22})

axarr[0, 1].step(range(len(measurements[0])),Kx, label='$x$')
axarr[0, 1].step(range(len(measurements[0])),Ky, label='$y$')
axarr[0, 1].step(range(len(measurements[0])),Kdx, label='$\psi$')
axarr[0, 1].step(range(len(measurements[0])),Kdy, label='$v$')
axarr[0, 1].step(range(len(measurements[0])),Kddx, label='$\dot \psi$')
axarr[0, 1].set_xlabel('Filter Step')
axarr[0, 1].set_ylabel('')
axarr[0, 1].set_title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
axarr[0, 1].legend(prop={'size':18})
axarr[0, 1].set_ylim([-0.1,0.1])


axarr[0, 2].step(range(len(measurements[0])),x0-mx[0], label='$x$')
axarr[0, 2].step(range(len(measurements[0])),x1-my[0], label='$y$')
axarr[0, 2].set_title('Extended Kalman Filter State Estimates (State Vector $x$)')
axarr[0, 2].legend(loc='best',prop={'size':22})
axarr[0, 2].set_ylabel('Position (relative to start) [m]')

axarr[1, 0].step(range(len(measurements[0])),x2, label='$\psi$')
axarr[1, 0].step(range(len(measurements[0])),(course/180.0*np.pi+np.pi)%(2.0*np.pi) - np.pi, label='$\psi$ (from GPS as reference)')
axarr[1, 0].set_ylabel('Course')
axarr[1, 0].legend(loc='best',prop={'size':16})

axarr[1, 1].step(range(len(measurements[0])),x3, label='$v$')
axarr[1, 1].step(range(len(measurements[0])),speed/3.6, label='$v$ (from GPS as reference)')
axarr[1, 1].set_ylabel('Velocity')
axarr[1, 1].set_ylim([0, 30])
axarr[1, 1].legend(loc='best',prop={'size':16})

axarr[1, 2].step(range(len(measurements[0])),x4, label='$\dot \psi$')
axarr[1, 2].step(range(len(measurements[0])),yawrate/180.0*np.pi, label='$\dot \psi$ (from IMU as reference)')
axarr[1, 2].set_ylabel('Yaw Rate')
axarr[1, 2].set_ylim([-0.6, 0.6])
axarr[1, 2].legend(loc='best',prop={'size':16})
axarr[1, 2].set_xlabel('Filter Step')


axarr[2, 0].quiver(x0,x1,np.cos(x2), np.sin(x2), color='#94C600', units='xy', width=0.05, scale=0.5)
axarr[2, 0].plot(x0,x1, label='EKF Position')
# Measurements
axarr[2, 0].scatter(mx[::5],my[::5], s=50, label='GPS Measurements')
# Start/Goal
axarr[2, 0].scatter(x0[0],x1[0], s=60, label='Start', c='g')
axarr[2, 0].scatter(x0[-1],x1[-1], s=60, label='Goal', c='r')
axarr[2, 0].set_xlabel('X [m]')
axarr[2, 0].set_ylabel('Y [m]')
axarr[2, 0].set_title('Position')
axarr[2, 0].legend(loc='best')
axarr[2, 0].axis('equal')


axarr[2, 1].semilogy(range(m),Qx, label='$x$')
axarr[2, 1].step(range(m),Qy, label='$y$')
axarr[2, 1].step(range(m),Qpsi, label='$\psi$')
axarr[2, 1].step(range(m),Qv, label='$v$')
axarr[2, 1].step(range(m),Qdpsi, label='$\dot \psi$')
axarr[2, 1].set_xlabel('Filter Step')
axarr[2, 1].set_ylabel('')
axarr[2, 1].set_title('Uncertainty (Elements from Matrix $Q$)')
axarr[2, 1].legend(loc='best',prop={'size':22})


plt.show()
