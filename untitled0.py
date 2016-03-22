# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:44:45 2016

@author: en1m12
"""

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *


m1 = 10
m2 = 10
k1 = 1e8
k2 = 1e8
c1 = 1e3
c2 = 1e3

M = np.matrix([[m1, 0],
             [0, m2]])
K = np.matrix([[k1, -k1],
             [-k1, k1+k2]])
C = np.matrix([[c1, -c1], [c1, c1+c2]])
L = np.matrix([1, 0])
L = L.transpose()

## I =  np.eye(2)
## Z = np.zeros((2,2))
##A = np.hstack([np.zeros((2,2)),np.eye(2)])

Minv = M.I

A = np.concatenate((np.concatenate((np.zeros((2,2)),np.eye(2)),axis=1),
                    np.concatenate((-Minv*K,-Minv*C),axis=1)),axis=0)

B = np.concatenate((np.zeros((2,1)),Minv*L),axis=0)

C = np.concatenate((np.concatenate((np.eye(2),np.zeros((2,2))),axis=1),
                    np.concatenate((np.zeros((2,2)),np.eye(2)),axis=1),
                    np.concatenate((-Minv*K,-Minv*C),axis=1)),axis=0)

D = np.concatenate((np.zeros((2,1)),np.zeros((2,1)),Minv*L),axis=0)

sys1 = ss(A,B,C,D)
##print(sys1)

force = np.random.normal(0,1,10000)
t = np.linspace(0,10,force.size)
##print(t.shape)
x0 = np.matrix([0, 0, 0, 0])
x0 = x0.transpose()
T, xout, yout = lsim(sys1, force, t, x0)

plt.plot(t,yout[:,2])
plt.show()
plt.xlabel('time, s')
##plt.ylabel('force, N')

#print(sys1)